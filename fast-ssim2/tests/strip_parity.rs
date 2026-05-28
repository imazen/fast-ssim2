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

use fast_ssim2::{
    LinearRgbImage, Ssimulacra2Reference, compute_ssimulacra2, compute_ssimulacra2_strip,
};

fn generate_image(width: usize, height: usize, seed: u32) -> LinearRgbImage {
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
    LinearRgbImage::new(data, width, height)
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
    let full = compute_ssimulacra2(img.clone(), img.clone()).unwrap();
    let strip = compute_ssimulacra2_strip(img.clone(), img.clone(), 32).unwrap();
    assert!(
        (full - strip).abs() < SCORE_TOLERANCE,
        "identical-image strip {strip:.4} vs full {full:.4} differs by more than {SCORE_TOLERANCE}",
    );
}

#[test]
fn strip_parity_different_64x64() {
    let source = generate_image(64, 64, 0);
    let distorted = generate_image(64, 64, 1);
    let full = compute_ssimulacra2(source.clone(), distorted.clone()).unwrap();
    let strip = compute_ssimulacra2_strip(source, distorted, 32).unwrap();
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
    let full = compute_ssimulacra2(source.clone(), distorted.clone()).unwrap();
    for strip_h in [32u32, 64, 128, 256] {
        let strip = compute_ssimulacra2_strip(source.clone(), distorted.clone(), strip_h).unwrap();
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
    let full = compute_ssimulacra2(source.clone(), distorted.clone()).unwrap();
    let strip = compute_ssimulacra2_strip(source, distorted, 128).unwrap();
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
    let full = compute_ssimulacra2(source.clone(), distorted.clone()).unwrap();
    let reference = Ssimulacra2Reference::new(source).unwrap();
    let strip = reference.compare_strip(distorted, 64).unwrap();
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
    let reference = Ssimulacra2Reference::new(img.clone()).unwrap();
    let compare = reference.compare(img.clone()).unwrap();
    let strip = reference.compare_strip(img, 64).unwrap();
    assert!(
        (compare - strip).abs() < SCORE_TOLERANCE,
        "compare {compare:.4} vs compare_strip {strip:.4} differs by more than {SCORE_TOLERANCE}",
    );
}

#[test]
fn strip_height_below_minimum_errors() {
    let img = generate_image(64, 64, 0);
    let err = compute_ssimulacra2_strip(img.clone(), img, 4)
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
    assert!(compute_ssimulacra2_strip(img.clone(), img, 0).is_err());
}

#[test]
fn strip_mismatched_dimensions_errors() {
    let a = generate_image(64, 64, 0);
    let b = generate_image(32, 32, 0);
    assert!(compute_ssimulacra2_strip(a, b, 32).is_err());
}
