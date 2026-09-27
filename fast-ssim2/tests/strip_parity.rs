//! Strip-vs-full parity tests for SSIMULACRA2.
//!
//! Verifies that `compute_ssimulacra2_strip` and
//! `Ssimulacra2Reference::compare_strip` produce scores within atomic
//! tolerance of the full-image path. The IIR Gaussian's exponential
//! impulse decay means that, with the default 96-row halo (per
//! `fast_ssim2::HALO_ROWS_DEFAULT`), the per-strip blurs differ from
//! the full-image blur by `e^{-halo / 1.5}` ≈ `1e-28` in the
//! pre-aggregation values at scale 0 and `e^{-6}` ≈ `1e-3` at scale 4
//! (the smallest scale on a 40 MP image). Empirically the final 0..100
//! score lands within `< 0.01` of the full-image score across the test
//! corpus at and above 256x256.

#![forbid(unsafe_code)]
#![allow(clippy::chunks_exact_to_as_chunks)]

use fast_ssim2::{
    Ssimulacra2Config, Ssimulacra2Reference, compute_ssimulacra2, compute_ssimulacra2_with_config,
};

fn lin_f32_buf(data: Vec<[f32; 3]>, w: usize, h: usize) -> zenpixels::PixelBuffer {
    zenpixels::PixelBuffer::from_vec(
        bytemuck::cast_slice::<f32, u8>(data.as_flattened()).to_vec(),
        w as u32,
        h as u32,
        zenpixels::PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap()
}

fn generate_image(width: usize, height: usize, seed: u32) -> zenpixels::PixelBuffer {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (((x as u32)
                .wrapping_mul(7)
                .wrapping_add((y as u32).wrapping_mul(13))
                .wrapping_add(seed))
                & 0xff) as f32
                / 255.0;
            let g = (((x as u32)
                .wrapping_mul(11)
                .wrapping_add((y as u32).wrapping_mul(3))
                .wrapping_add(seed.wrapping_add(50)))
                & 0xff) as f32
                / 255.0;
            let b = (((x as u32)
                .wrapping_mul(5)
                .wrapping_add((y as u32).wrapping_mul(17))
                .wrapping_add(seed.wrapping_add(100)))
                & 0xff) as f32
                / 255.0;
            data.push([r, g, b]);
        }
    }
    lin_f32_buf(data, width, height)
}

/// SSIMULACRA2's score domain is roughly 0..100 with FMA + IIR
/// rounding contributing ~1e-2 to ~1e-1 of variation between SIMD
/// tiers (see `tests/simd_consistency.rs`). Strip mode with default
/// halo is a different blur context, so we set a slightly wider
/// tolerance here: 0.5 on the 0..100 scale catches structural bugs
/// while accommodating the halo-induced blur boundary effects.
const SCORE_TOLERANCE: f64 = 0.5;

#[test]
fn strip_parity_identical_64x64() {
    let img = generate_image(64, 64, 42);
    // Identical inputs MUST score very close to 100 in both modes.
    let full = compute_ssimulacra2(&img.as_slice(), &img.as_slice()).unwrap();
    let strip = compute_ssimulacra2_with_config(
        &img.as_slice(),
        &img.as_slice(),
        &Ssimulacra2Config::strips(32),
    )
    .unwrap();
    assert!(
        (full - strip).abs() < SCORE_TOLERANCE,
        "identical-image strip {strip:.4} vs full {full:.4} differs by more than {SCORE_TOLERANCE}",
    );
}

#[test]
fn strip_parity_different_64x64() {
    let source = generate_image(64, 64, 0);
    let distorted = generate_image(64, 64, 1);
    let full = compute_ssimulacra2(&source.as_slice(), &distorted.as_slice()).unwrap();
    let strip = compute_ssimulacra2_with_config(
        &source.as_slice(),
        &distorted.as_slice(),
        &Ssimulacra2Config::strips(32),
    )
    .unwrap();
    assert!(
        (full - strip).abs() < SCORE_TOLERANCE,
        "different-image strip {strip:.4} vs full {full:.4} differs by more than {SCORE_TOLERANCE}",
    );
}

#[test]
fn strip_parity_512x512_jpeg_like() {
    let width = 512;
    let height = 512;
    let source = generate_image(width, height, 7);
    let distorted = generate_image(width, height, 8);
    let full = compute_ssimulacra2(&source.as_slice(), &distorted.as_slice()).unwrap();
    for strip_h in [32u32, 64, 128, 256] {
        let strip = compute_ssimulacra2_with_config(
            &source.as_slice(),
            &distorted.as_slice(),
            &Ssimulacra2Config::strips(strip_h as usize),
        )
        .unwrap();
        assert!(
            (full - strip).abs() < SCORE_TOLERANCE,
            "512x512 strip_h={strip_h} score {strip:.4} vs full {full:.4} differs by more than {SCORE_TOLERANCE}",
        );
    }
}

#[test]
fn strip_parity_1024x1024_jpeg_like() {
    let width = 1024;
    let height = 1024;
    let source = generate_image(width, height, 11);
    let distorted = generate_image(width, height, 12);
    let full = compute_ssimulacra2(&source.as_slice(), &distorted.as_slice()).unwrap();
    let strip = compute_ssimulacra2_with_config(
        &source.as_slice(),
        &distorted.as_slice(),
        &Ssimulacra2Config::strips(128),
    )
    .unwrap();
    assert!(
        (full - strip).abs() < SCORE_TOLERANCE,
        "1024x1024 strip {strip:.4} vs full {full:.4} differs by more than {SCORE_TOLERANCE}",
    );
}

#[test]
fn warm_ref_strip_parity_512x512() {
    let width = 512;
    let height = 512;
    let source = generate_image(width, height, 21);
    let distorted = generate_image(width, height, 22);
    let full = compute_ssimulacra2(&source.as_slice(), &distorted.as_slice()).unwrap();
    let reference = Ssimulacra2Reference::new(&source.as_slice()).unwrap();
    let strip = reference
        .compare_with_config(&distorted.as_slice(), &Ssimulacra2Config::strips(64))
        .unwrap();
    assert!(
        (full - strip).abs() < SCORE_TOLERANCE,
        "compare_strip score {strip:.4} vs full {full:.4} differs by more than {SCORE_TOLERANCE}",
    );
}

#[test]
fn warm_ref_strip_matches_compare() {
    // `compare_strip` must produce a score that is close to
    // `compare`, which itself agrees with `compute_ssimulacra2`
    // (see precompute.rs tests). All three paths use the same
    // SIMD ops, so the differences are bounded by the strip halo's
    // exponential decay.
    let img = generate_image(256, 256, 99);
    let reference = Ssimulacra2Reference::new(&img.as_slice()).unwrap();
    let compare = reference.compare(&img.as_slice()).unwrap();
    let strip = reference
        .compare_with_config(&img.as_slice(), &Ssimulacra2Config::strips(64))
        .unwrap();
    assert!(
        (compare - strip).abs() < SCORE_TOLERANCE,
        "compare {compare:.4} vs compare_strip {strip:.4} differs by more than {SCORE_TOLERANCE}",
    );
}

#[test]
fn strip_height_below_minimum_errors() {
    let img = generate_image(64, 64, 0);
    let err = compute_ssimulacra2_with_config(
        &img.as_slice(),
        &img.as_slice(),
        &Ssimulacra2Config::strips(4),
    )
    .expect_err("strip_height=4 < MIN_STRIP_HEIGHT must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("at least") || msg.contains("8x8") || msg.contains("size"),
        "unexpected error message: {msg}",
    );
}

#[test]
fn strip_height_zero_errors() {
    let img = generate_image(64, 64, 0);
    assert!(
        compute_ssimulacra2_with_config(
            &img.as_slice(),
            &img.as_slice(),
            &Ssimulacra2Config::strips(0)
        )
        .is_err()
    );
}

#[test]
fn strip_mismatched_dimensions_errors() {
    let a = generate_image(64, 64, 0);
    let b = generate_image(32, 32, 0);
    assert!(
        compute_ssimulacra2_with_config(
            &a.as_slice(),
            &b.as_slice(),
            &Ssimulacra2Config::strips(32)
        )
        .is_err()
    );
}

// ============================================================================
// Match-official strip parity
// ============================================================================

fn generate_encoded_rgb(width: usize, height: usize, seed: u32) -> zenpixels::PixelBuffer {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let v = (((x as u32)
                .wrapping_mul(7)
                .wrapping_add(y as u32 * 13)
                .wrapping_add(seed))
                & 0xff) as f32
                / 255.0;
            let g = (((x as u32)
                .wrapping_mul(11)
                .wrapping_add(y as u32 * 3)
                .wrapping_add(seed + 50))
                & 0xff) as f32
                / 255.0;
            let b = (((x as u32)
                .wrapping_mul(5)
                .wrapping_add(y as u32 * 17)
                .wrapping_add(seed + 100))
                & 0xff) as f32
                / 255.0;
            data.push([v, g, b]);
        }
    }
    zenpixels::PixelBuffer::from_vec(
        bytemuck::cast_slice::<f32, u8>(data.as_flattened()).to_vec(),
        width as u32,
        height as u32,
        zenpixels::PixelDescriptor::RGBF32.with_transfer(zenpixels::TransferFunction::Srgb),
    )
    .unwrap()
}

/// Strip mode must approximate the full-image score within the halo
/// tolerance (serial and parallel paths alike, bit-identical).
#[test]
fn strip_parity_official_512x512() {
    use fast_ssim2::{Ssimulacra2Config, compute_ssimulacra2_with_config};
    let source = generate_encoded_rgb(512, 512, 7);
    let distorted = generate_encoded_rgb(512, 512, 8);
    let full = compute_ssimulacra2_with_config(
        &source.as_slice(),
        &distorted.as_slice(),
        &Ssimulacra2Config::default(),
    )
    .unwrap();
    for strip_h in [64usize, 128, 256] {
        let strip = compute_ssimulacra2_with_config(
            &source.as_slice(),
            &distorted.as_slice(),
            &Ssimulacra2Config::strips(strip_h),
        )
        .unwrap();
        assert!(
            (full - strip).abs() < SCORE_TOLERANCE,
            "strip{strip_h} {strip:.4} vs full {full:.4} differs by more than {SCORE_TOLERANCE}",
        );
    }
}

/// `parallel_strips` must produce bit-identical output: strips merge in
/// fixed order so the f64 accumulation is schedule-independent.
#[test]
fn strip_parallel_deterministic() {
    use fast_ssim2::{Ssimulacra2Config, compute_ssimulacra2_with_config};
    let source = generate_image(512, 512, 11);
    let distorted = generate_image(512, 512, 12);
    for strip_h in [32usize, 96] {
        let serial_cfg = Ssimulacra2Config::strips(strip_h);
        let mut par_cfg = Ssimulacra2Config::strips(strip_h);
        par_cfg.strip.as_mut().unwrap().parallel_strips = true;
        let serial =
            compute_ssimulacra2_with_config(&source.as_slice(), &distorted.as_slice(), &serial_cfg)
                .unwrap();
        let par =
            compute_ssimulacra2_with_config(&source.as_slice(), &distorted.as_slice(), &par_cfg)
                .unwrap();
        assert_eq!(
            serial, par,
            "strip_h={strip_h}: parallel {par} != serial {serial} — ordered merge is broken",
        );
    }
}
