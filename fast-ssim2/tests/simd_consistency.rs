//! SIMD tier consistency tests for fast-ssim2.
//!
//! Runs SSIMULACRA2 computation under every archmage SIMD tier permutation
//! and verifies all produce matching scores within acceptable tolerance.
//! The multi-scale Gaussian blur and XYB conversion compound FMA rounding
//! differences across 6 downscale levels, so the tolerance is wider than
//! single-operation tests.

#![forbid(unsafe_code)]
#![allow(clippy::chunks_exact_to_as_chunks)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use fast_ssim2::{Ssimulacra2Config, compute_ssimulacra2_with_config};

fn lin_f32_buf(data: Vec<[f32; 3]>, w: usize, h: usize) -> zenpixels::PixelBuffer {
    zenpixels::PixelBuffer::from_vec(
        bytemuck::cast_slice::<f32, u8>(data.as_flattened()).to_vec(),
        w as u32,
        h as u32,
        zenpixels::PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap()
}

/// Generate a deterministic test image of varied linear RGB pixels.
fn generate_test_image(width: usize, height: usize) -> zenpixels::PixelBuffer {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 7 + y * 13) % 256) as f32 / 255.0;
            let g = ((x * 11 + y * 3 + 50) % 256) as f32 / 255.0;
            let b = ((x * 5 + y * 17 + 100) % 256) as f32 / 255.0;
            data.push([r, g, b]);
        }
    }
    lin_f32_buf(data, width, height)
}

/// Generate a slightly different image (simulated distortion).
fn generate_distorted_image(width: usize, height: usize) -> zenpixels::PixelBuffer {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (((x * 7 + y * 13) % 256) as f32 / 255.0 + 0.02).min(1.0);
            let g = (((x * 11 + y * 3 + 50) % 256) as f32 / 255.0 - 0.01).max(0.0);
            let b = (((x * 5 + y * 17 + 100) % 256) as f32 / 255.0 + 0.005).min(1.0);
            data.push([r, g, b]);
        }
    }
    lin_f32_buf(data, width, height)
}

#[test]
fn ssimulacra2_all_tiers_within_tolerance() {
    let source = generate_test_image(64, 64);
    let distorted = generate_distorted_image(64, 64);
    let mut reference_score: Option<f64> = None;

    // SSIMULACRA2 compounds FMA rounding differences across 6 downscale
    // levels of Gaussian blur + XYB conversion. A tolerance of 0.5 on the
    // 0-100 scale catches algorithmic bugs while allowing FMA divergence.
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let score = compute_ssimulacra2_with_config(
            &source.as_slice(),
            &distorted.as_slice(),
            &Ssimulacra2Config::simd(),
        )
        .expect("score");

        if let Some(ref_score) = reference_score {
            let diff = (score - ref_score).abs();
            assert!(
                diff < 0.5,
                "ssimulacra2 score differs under '{}': {score} vs {ref_score} (diff={diff})",
                perm.label,
            );
        } else {
            reference_score = Some(score);
        }
    });
}

#[test]
fn ssimulacra2_roundtrip_stability() {
    let source = generate_test_image(32, 32);
    let distorted = generate_distorted_image(32, 32);

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let score1 = compute_ssimulacra2_with_config(
            &source.as_slice(),
            &distorted.as_slice(),
            &Ssimulacra2Config::simd(),
        )
        .expect("score1");

        let score2 = compute_ssimulacra2_with_config(
            &source.as_slice(),
            &distorted.as_slice(),
            &Ssimulacra2Config::simd(),
        )
        .expect("score2");

        assert_eq!(
            score1.to_bits(),
            score2.to_bits(),
            "ssimulacra2 not deterministic under '{}': {score1} vs {score2}",
            perm.label,
        );
    });
}
