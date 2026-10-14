//! Native per-pixel terms of the SSIMULACRA2 score.

/// The spatial terms one comparison pools into its score.
///
/// SSIMULACRA2 has no single per-pixel score map. At each pyramid scale
/// and positive-XYB channel it computes three per-pixel error terms
/// ([`Ssimulacra2ScaleMaps`]), takes the mean (L1) and the 4-norm (L4) of
/// each over the scale, and feeds those 108 aggregates through a weighted
/// sum and a nonlinear transform. These are the terms before that
/// pooling: each is `0` where the images agree and grows with the error.
///
/// Returned by [`crate::compute_ssimulacra2_maps`] and
/// [`crate::compute_ssimulacra2_maps_with_config`]; [`Self::score`] is
/// bit-identical to [`crate::compute_ssimulacra2_with_config`] on the same
/// pair and configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Ssimulacra2Maps {
    /// The SSIMULACRA2 score of the pair.
    pub score: f64,
    /// One entry per evaluated pyramid scale, finest first (1–6 scales;
    /// small images stop early, exactly as the score does).
    pub scales: Vec<Ssimulacra2ScaleMaps>,
    /// For inputs with alpha, the encoded-sRGB background (`0.1` or `0.9`)
    /// whose composite produced [`Self::score`] — the score is the lower of
    /// the two, and these maps are that composite's. `None` without alpha.
    pub alpha_background: Option<f32>,
}

/// Per-pixel terms at one pyramid scale, each `width × height`, row-major
/// and tightly packed, indexed `[X, Y, B]` by positive-XYB channel.
///
/// Scale `s` samples the working image in `2^s × 2^s` blocks: sample
/// `(x, y)` is the box mean of input pixels `[x·2^s, (x+1)·2^s) ×
/// [y·2^s, (y+1)·2^s)` (edge blocks repeat the last row/column). The
/// working image is the input, reflect-padded to at least 8×8 when it is
/// smaller (the input occupies the top-left corner). Each value also
/// depends on the samples within the 5-sample radius of the scale's
/// Gaussian blur.
///
/// Values are the score's f64 terms narrowed to `f32`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Ssimulacra2ScaleMaps {
    /// Samples per row at this scale.
    pub width: usize,
    /// Rows at this scale.
    pub height: usize,
    /// `max(0, 1 − SSIM)` of the blurred local statistics.
    pub ssim: [Vec<f32>; 3],
    /// Edge artifact: how much more local detail (`|x − blur(x)|`) the
    /// distorted image has than the reference — ringing, blocking, noise.
    pub artifact: [Vec<f32>; 3],
    /// Detail lost: how much less local detail the distorted image has —
    /// blur and smoothing.
    pub detail_lost: [Vec<f32>; 3],
}

#[cfg(test)]
mod tests {
    use crate::pipeline::gauss::{create_recursive_gaussian, multiply_planes};
    use crate::pipeline::maps::{edge_diff_map, ssim_map};
    use crate::pipeline::score::{ScaleAggregates, score};
    use crate::pipeline::{EncodedData, EncodedSrgb};
    use crate::pipeline::{blur_planes, linearize, planes_to_positive_xyb};
    use crate::{PixelDescriptor, PixelSlice, compute_ssimulacra2_maps};

    fn textured(w: usize, h: usize, seed: u32) -> Vec<u8> {
        (0..w * h * 3)
            .map(|i| {
                let hash = (i as u32 ^ seed).wrapping_mul(2_654_435_761);
                (((i / 3) % w * 3 + (i / 3) / w * 5) % 160 + (hash >> 27) as usize * 3) as u8
            })
            .collect()
    }

    fn l1_l4(plane: &[f32]) -> (f64, f64) {
        let opp = 1.0 / plane.len() as f64;
        let (mut s1, mut s4) = (0f64, 0f64);
        for &v in plane {
            let v = v as f64;
            s1 += v;
            s4 += (v * v) * (v * v);
        }
        (opp * s1, (opp * s4).sqrt().sqrt())
    }

    fn rel_close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
    }

    /// Scale 0 re-derived with the reference-order scalar helpers
    /// (`linearize` → `planes_to_positive_xyb` → `blur_planes` →
    /// `ssim_map`/`edge_diff_map`) agrees with the maps' own L1/L4 means
    /// to f32 storage precision; and the score rebuilt from the maps of all
    /// scales matches the reported score.
    #[test]
    fn maps_reproduce_the_independent_aggregates_and_the_score() {
        let (w, h) = (53, 37);
        let (a, b) = (textured(w, h, 1), textured(w, h, 99));
        let (sa, sb) = (
            PixelSlice::new(&a, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap(),
            PixelSlice::new(&b, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap(),
        );
        let maps = compute_ssimulacra2_maps(&sa, &sb).unwrap();

        let enc = |d: &[u8]| EncodedSrgb {
            width: w,
            height: h,
            data: EncodedData::U8(d.to_vec()),
            alpha: None,
        };
        let (mut x1, mut x2) = (linearize(&enc(&a), 0.5), linearize(&enc(&b), 0.5));
        planes_to_positive_xyb(&mut x1, w * h);
        planes_to_positive_xyb(&mut x2, w * h);
        let rg = create_recursive_gaussian(1.5);
        let blur_product = |p: &[Vec<f32>; 3], q: &[Vec<f32>; 3]| {
            let mut m = [vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]];
            multiply_planes(p, q, &mut m);
            blur_planes(&rg, &m, w, h)
        };
        let (s11, s22, s12) = (
            blur_product(&x1, &x1),
            blur_product(&x2, &x2),
            blur_product(&x1, &x2),
        );
        let (mu1, mu2) = (blur_planes(&rg, &x1, w, h), blur_planes(&rg, &x2, w, h));
        let want_ssim = ssim_map(&mu1, &mu2, &s11, &s22, &s12, w, h);
        let want_edge = edge_diff_map(&x1, &mu1, &x2, &mu2, w, h);

        let s0 = &maps.scales[0];
        for c in 0..3 {
            let (l1, l4) = l1_l4(&s0.ssim[c]);
            assert!(rel_close(l1, want_ssim[c * 2], 1e-6), "ssim L1 c{c}");
            assert!(rel_close(l4, want_ssim[c * 2 + 1], 1e-6), "ssim L4 c{c}");
            let (a1, a4) = l1_l4(&s0.artifact[c]);
            let (d1, d4) = l1_l4(&s0.detail_lost[c]);
            for (got, want) in [(a1, 0), (a4, 1), (d1, 2), (d4, 3)] {
                assert!(
                    rel_close(got, want_edge[c * 4 + want], 1e-6),
                    "edge {want} c{c}"
                );
            }
        }

        let rebuilt: Vec<ScaleAggregates> = maps
            .scales
            .iter()
            .map(|s| {
                let mut avg_ssim = [0f64; 6];
                let mut avg_edgediff = [0f64; 12];
                for c in 0..3 {
                    (avg_ssim[c * 2], avg_ssim[c * 2 + 1]) = l1_l4(&s.ssim[c]);
                    (avg_edgediff[c * 4], avg_edgediff[c * 4 + 1]) = l1_l4(&s.artifact[c]);
                    (avg_edgediff[c * 4 + 2], avg_edgediff[c * 4 + 3]) = l1_l4(&s.detail_lost[c]);
                }
                ScaleAggregates {
                    avg_ssim,
                    avg_edgediff,
                }
            })
            .collect();
        let from_maps = score(&rebuilt);
        eprintln!(
            "score {} rebuilt from f32 maps {from_maps} (|delta| {:e})",
            maps.score,
            (maps.score - from_maps).abs()
        );
        assert!((maps.score - from_maps).abs() < 1e-5);
    }
}
