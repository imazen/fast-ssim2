//! `compute_ssimulacra2_maps*`: the map path never changes the score, the
//! maps describe the documented pyramid geometry, and they localize a
//! distortion where it was made.
use std::sync::atomic::{AtomicUsize, Ordering};

use enough::{Stop, StopReason};
use fast_ssim2::{
    PixelDescriptor, PixelSlice, Ssimulacra2Config, Ssimulacra2Error, Ssimulacra2Maps,
    compute_ssimulacra2_maps, compute_ssimulacra2_maps_with_config,
    compute_ssimulacra2_with_config,
};

/// Deterministic photo-like texture: smooth structure plus hashed noise.
fn textured(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3u32 {
                let hash = ((x as u32).wrapping_mul(73_856_093)
                    ^ (y as u32).wrapping_mul(19_349_663)
                    ^ c.wrapping_mul(83_492_791)
                    ^ seed)
                    .wrapping_mul(2_654_435_761);
                let smooth = ((x * 3 + y * 5 + c as usize * 40) % 160) as u32;
                out.push((smooth + (hash >> 27) * 3) as u8);
            }
        }
    }
    out
}

fn rgb(bytes: &[u8], w: usize, h: usize) -> PixelSlice<'_> {
    strided(bytes, w, h, w * 3, PixelDescriptor::RGB8_SRGB)
}

fn strided(bytes: &[u8], w: usize, h: usize, stride: usize, d: PixelDescriptor) -> PixelSlice<'_> {
    PixelSlice::new(bytes, w as u32, h as u32, stride, d).unwrap()
}

/// `distorted` with a `size × size` patch at `(px, py)` brightened.
fn with_patch(src: &[u8], w: usize, px: usize, py: usize, size: usize) -> Vec<u8> {
    let mut out = src.to_vec();
    for y in py..py + size {
        for x in px..px + size {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                out[i] = out[i].saturating_add(60);
            }
        }
    }
    out
}

fn assert_same_score(
    a: &PixelSlice<'_>,
    b: &PixelSlice<'_>,
    cfg: &Ssimulacra2Config<'_>,
) -> Ssimulacra2Maps {
    let score = compute_ssimulacra2_with_config(a, b, cfg).unwrap();
    let maps = compute_ssimulacra2_maps_with_config(a, b, cfg).unwrap();
    assert_eq!(
        maps.score.to_bits(),
        score.to_bits(),
        "map path score {} != score path {score}",
        maps.score
    );
    maps
}

/// Scale `s` is `ceil(w / 2^s) × ceil(h / 2^s)` of the working image, and
/// the pyramid stops where the score's does (the gate is the previous
/// scale being at least 8 on both axes).
fn expected_dims(w: usize, h: usize) -> Vec<(usize, usize)> {
    let (mut w, mut h) = (w.max(8), h.max(8));
    let (mut gw, mut gh) = (w, h);
    let mut out = Vec::new();
    for _ in 0..6 {
        if gw < 8 || gh < 8 {
            break;
        }
        out.push((w, h));
        (gw, gh) = (w, h);
        (w, h) = (w.div_ceil(2), h.div_ceil(2));
    }
    out
}

fn check_geometry(maps: &Ssimulacra2Maps, w: usize, h: usize) {
    let dims: Vec<_> = maps.scales.iter().map(|s| (s.width, s.height)).collect();
    assert_eq!(dims, expected_dims(w, h), "{w}x{h}");
    for s in &maps.scales {
        for plane in s.ssim.iter().chain(&s.artifact).chain(&s.detail_lost) {
            assert_eq!(plane.len(), s.width * s.height);
            assert!(plane.iter().all(|v| v.is_finite() && *v >= 0.0));
        }
    }
}

#[test]
fn map_path_score_is_bit_identical_across_inputs_and_kernels() {
    for (w, h) in [
        (64usize, 48usize),
        (37, 23),
        (8, 8),
        (5, 3),
        (1, 1),
        (130, 9),
    ] {
        let a = textured(w, h, 1);
        let b = textured(w, h, 2);
        let flat_a = vec![128u8; w * h * 3];
        let flat_b = vec![140u8; w * h * 3];
        for cfg in [Ssimulacra2Config::simd(), Ssimulacra2Config::scalar()] {
            for (r, d) in [(&a, &b), (&a, &a), (&flat_a, &flat_b), (&a, &flat_b)] {
                let maps = assert_same_score(&rgb(r, w, h), &rgb(d, w, h), &cfg);
                check_geometry(&maps, w, h);
                assert_eq!(maps.alpha_background, None);
            }
        }
    }
}

#[test]
fn identical_pairs_have_all_zero_maps() {
    let (w, h) = (41, 29);
    let a = textured(w, h, 7);
    let maps = compute_ssimulacra2_maps(&rgb(&a, w, h), &rgb(&a, w, h)).unwrap();
    assert_eq!(maps.score, 100.0);
    for s in &maps.scales {
        for plane in s.ssim.iter().chain(&s.artifact).chain(&s.detail_lost) {
            assert!(plane.iter().all(|&v| v == 0.0));
        }
    }
}

#[test]
fn strided_and_linear_inputs_keep_the_score() {
    let (w, h) = (33, 21);
    let a = textured(w, h, 3);
    let b = textured(w, h, 4);
    // Rows padded by three pixels; the padding bytes are garbage.
    let stride = (w + 3) * 3;
    let pad = |src: &[u8]| {
        let mut out = vec![0xA5u8; stride * h];
        for y in 0..h {
            out[y * stride..y * stride + w * 3].copy_from_slice(&src[y * w * 3..(y + 1) * w * 3]);
        }
        out
    };
    let (pa, pb) = (pad(&a), pad(&b));
    let cfg = Ssimulacra2Config::default();
    let packed = assert_same_score(&rgb(&a, w, h), &rgb(&b, w, h), &cfg);
    let padded = assert_same_score(
        &strided(&pa, w, h, stride, PixelDescriptor::RGB8_SRGB),
        &strided(&pb, w, h, stride, PixelDescriptor::RGB8_SRGB),
        &cfg,
    );
    assert_eq!(packed.score.to_bits(), padded.score.to_bits());
    for (p, q) in packed.scales.iter().zip(&padded.scales) {
        assert_eq!(p.ssim, q.ssim);
        assert_eq!(p.artifact, q.artifact);
        assert_eq!(p.detail_lost, q.detail_lost);
    }

    let lin = |src: &[u8]| -> Vec<f32> {
        src.iter()
            .map(|&v| {
                let x = v as f32 / 255.0;
                if x <= 0.04045 {
                    x / 12.92
                } else {
                    ((x + 0.055) / 1.055).powf(2.4)
                }
            })
            .collect()
    };
    let (la, lb) = (lin(&a), lin(&b));
    let maps = assert_same_score(
        &strided(
            bytemuck::cast_slice(&la),
            w,
            h,
            w * 12,
            PixelDescriptor::RGBF32_LINEAR,
        ),
        &strided(
            bytemuck::cast_slice(&lb),
            w,
            h,
            w * 12,
            PixelDescriptor::RGBF32_LINEAR,
        ),
        &cfg,
    );
    check_geometry(&maps, w, h);
}

#[test]
fn alpha_maps_come_from_the_lower_scoring_background() {
    let (w, h) = (24, 20);
    let mk = |seed| {
        let rgb = textured(w, h, seed);
        let mut out = Vec::with_capacity(w * h * 4);
        for (i, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
            out.extend_from_slice(p);
            out.push(((i * 37) % 256) as u8);
        }
        out
    };
    let (a, b) = (mk(5), mk(6));
    let maps = assert_same_score(
        &strided(&a, w, h, w * 4, PixelDescriptor::RGBA8_SRGB),
        &strided(&b, w, h, w * 4, PixelDescriptor::RGBA8_SRGB),
        &Ssimulacra2Config::default(),
    );
    check_geometry(&maps, w, h);
    let bg = maps
        .alpha_background
        .expect("alpha input names its background");
    assert!(bg == 0.1 || bg == 0.9, "{bg}");
}

#[test]
fn a_local_distortion_peaks_inside_its_patch() {
    let (w, h) = (96, 64);
    let a = textured(w, h, 11);
    let (px, py, size) = (60usize, 20usize, 8usize);
    let b = with_patch(&a, w, px, py, size);
    let maps = compute_ssimulacra2_maps(&rgb(&a, w, h), &rgb(&b, w, h)).unwrap();
    for (s, scale) in maps.scales.iter().enumerate().take(3) {
        let f = 1usize << s;
        // Sum the three channels' SSIM error and find the peak sample.
        let (mut best, mut at) = (f32::MIN, (0, 0));
        for y in 0..scale.height {
            for x in 0..scale.width {
                let i = y * scale.width + x;
                let v: f32 = (0..3).map(|c| scale.ssim[c][i]).sum();
                if v > best {
                    (best, at) = (v, (x, y));
                }
            }
        }
        // Sample (x, y) owns input [x·f, (x+1)·f); the blur reaches five
        // samples further.
        let (x0, x1) = (
            (at.0 as isize - 5) * f as isize,
            (at.0 as isize + 6) * f as isize,
        );
        let (y0, y1) = (
            (at.1 as isize - 5) * f as isize,
            (at.1 as isize + 6) * f as isize,
        );
        let (pxi, pyi, si) = (px as isize, py as isize, size as isize);
        assert!(
            x0 < pxi + si && pxi < x1 && y0 < pyi + si && pyi < y1,
            "scale {s}: peak at {at:?} does not reach the patch"
        );
        assert!(best > 0.0, "scale {s}: no error");
        // Before the patch's support (left of it, or above it) the
        // recursive blur has not yet read a differing sample: exactly 0.
        // After it (right of it, or below it) only the blur's rounding
        // residue remains, which is measured and reported.
        let before = |x: usize, y: usize| (x + 1 + 5) * f <= px || (y + 1 + 5) * f <= py;
        let after = |x: usize, y: usize| x * f >= px + size + 5 * f || y * f >= py + size + 5 * f;
        let mut tail = 0f32;
        for y in 0..scale.height {
            for x in 0..scale.width {
                let i = y * scale.width + x;
                for c in 0..3 {
                    let terms = [
                        scale.ssim[c][i],
                        scale.artifact[c][i],
                        scale.detail_lost[c][i],
                    ];
                    if before(x, y) {
                        assert_eq!(terms, [0.0; 3], "scale {s} ({x},{y}) c{c}");
                    } else if after(x, y) {
                        tail = terms.iter().fold(tail, |m, &v| m.max(v));
                    }
                }
            }
        }
        eprintln!("scale {s}: peak {best:e}, rounding tail beyond the support {tail:e}");
        assert!(tail < best * 1e-3, "scale {s}: tail {tail} vs peak {best}");
    }
}

/// Trips on the `limit`-th poll.
struct StopAfter {
    polls: AtomicUsize,
    limit: usize,
}

impl Stop for StopAfter {
    fn check(&self) -> Result<(), StopReason> {
        if self.polls.fetch_add(1, Ordering::Relaxed) + 1 >= self.limit {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_every_poll_is_an_error_never_a_wrong_result() {
    let (w, h) = (40, 36);
    let a = textured(w, h, 21);
    let b = textured(w, h, 22);
    let want = compute_ssimulacra2_maps(&rgb(&a, w, h), &rgb(&b, w, h)).unwrap();
    let mut completed = false;
    for limit in 1..4000 {
        let stop = StopAfter {
            polls: AtomicUsize::new(0),
            limit,
        };
        let cfg = Ssimulacra2Config::default().with_stop(&stop);
        match compute_ssimulacra2_maps_with_config(&rgb(&a, w, h), &rgb(&b, w, h), &cfg) {
            Err(Ssimulacra2Error::Cancelled(StopReason::Cancelled)) => {}
            Ok(got) => {
                assert_eq!(got.score.to_bits(), want.score.to_bits());
                for (p, q) in got.scales.iter().zip(&want.scales) {
                    assert_eq!(p.ssim, q.ssim);
                }
                completed = true;
                break;
            }
            Err(e) => panic!("limit {limit}: {e:?}"),
        }
    }
    assert!(completed, "never completed within the poll budget");
}

#[test]
fn strip_evaluation_is_refused_for_maps() {
    let a = textured(16, 16, 1);
    assert!(matches!(
        compute_ssimulacra2_maps_with_config(
            &rgb(&a, 16, 16),
            &rgb(&a, 16, 16),
            &Ssimulacra2Config::strips(8)
        ),
        Err(Ssimulacra2Error::InvalidConfiguration(_))
    ));
    assert_eq!(
        compute_ssimulacra2_maps(&rgb(&a, 16, 16), &rgb(&a[..8 * 16 * 3], 16, 8)).err(),
        Some(Ssimulacra2Error::NonMatchingImageDimensions)
    );
}
