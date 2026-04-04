//! Tests that verify all SIMD implementations produce matching scores.
//!
//! This ensures Scalar and Simd (archmage) backends compute the same results.

use fast_ssim2::{Ssimulacra2Config, compute_ssimulacra2_with_config};
use image::ImageReader;
use std::path::PathBuf;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn test_data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("jpeg_quality")
}

fn load_image(filename: &str) -> Rgb {
    let path = test_data_path().join(filename);
    let img = ImageReader::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e))
        .decode()
        .unwrap_or_else(|e| panic!("Failed to decode {}: {}", path.display(), e))
        .to_rgb8();

    let (width, height) = img.dimensions();
    let data: Vec<[f32; 3]> = img
        .pixels()
        .map(|p| {
            [
                f32::from(p[0]) / 255.0,
                f32::from(p[1]) / 255.0,
                f32::from(p[2]) / 255.0,
            ]
        })
        .collect();

    Rgb::new(
        data,
        std::num::NonZeroUsize::new(width as usize).unwrap(),
        std::num::NonZeroUsize::new(height as usize).unwrap(),
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed to create Rgb")
}

/// Create synthetic gradient test images
fn create_synthetic_images(width: usize, height: usize) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let source_data: Vec<[f32; 3]> = (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            [x, y, (x + y) / 2.0]
        })
        .collect();

    let distorted_data: Vec<[f32; 3]> = source_data
        .iter()
        .map(|&[r, g, b]| [r * 0.95, g * 1.02, b * 0.98])
        .collect();

    (source_data, distorted_data)
}

fn compute_score_from_data(
    source_data: &[[f32; 3]],
    distorted_data: &[[f32; 3]],
    width: usize,
    height: usize,
    config: Ssimulacra2Config,
) -> f64 {
    let nz_width = std::num::NonZeroUsize::new(width).unwrap();
    let nz_height = std::num::NonZeroUsize::new(height).unwrap();
    let source = Rgb::new(
        source_data.to_vec(),
        nz_width,
        nz_height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let distorted = Rgb::new(
        distorted_data.to_vec(),
        nz_width,
        nz_height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_ssimulacra2_with_config(source, distorted, config).unwrap()
}

// ============================================================================
// Exact match tests - identical images must score exactly 100.0
// ============================================================================

#[test]
fn test_identical_images_exact_score_scalar() {
    let source = load_image("source.png");
    let score =
        compute_ssimulacra2_with_config(source.clone(), source, Ssimulacra2Config::scalar())
            .unwrap();
    assert_eq!(
        score, 100.0,
        "Scalar: identical images must score exactly 100.0, got {}",
        score
    );
}

#[test]
fn test_identical_images_exact_score_simd() {
    let source = load_image("source.png");
    let score =
        compute_ssimulacra2_with_config(source.clone(), source, Ssimulacra2Config::simd()).unwrap();
    assert_eq!(
        score, 100.0,
        "SIMD: identical images must score exactly 100.0, got {}",
        score
    );
}

// ============================================================================
// Real JPEG artifact tests - pinned expected values for regression detection
// ============================================================================

/// Test cases with real JPEG compression artifacts.
/// Expected values are pinned from SIMD implementation - if these change,
/// it indicates a regression or intentional algorithm change.
struct RealImageTestCase {
    name: &'static str,
    distorted_file: &'static str,
    /// Expected score from SIMD implementation (pinned value, x86_64 only)
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    expected_simd: f64,
}

const REAL_IMAGE_CASES: &[RealImageTestCase] = &[
    RealImageTestCase {
        name: "JPEG Q20",
        distorted_file: "q20.jpg",
        expected_simd: 57.093473, // Pinned SIMD value (f32 Halley cbrt, captured 2026-04-04)
    },
    RealImageTestCase {
        name: "JPEG Q45",
        distorted_file: "q45.jpg",
        expected_simd: 68.675775, // Pinned SIMD value (f32 Halley cbrt, captured 2026-04-04)
    },
    RealImageTestCase {
        name: "JPEG Q70",
        distorted_file: "q70.jpg",
        expected_simd: 79.491173, // Pinned SIMD value (f32 Halley cbrt, captured 2026-04-04)
    },
    RealImageTestCase {
        name: "JPEG Q90",
        distorted_file: "q90.jpg",
        expected_simd: 90.834538, // Pinned SIMD value (f32 Halley cbrt, captured 2026-04-04)
    },
];

// Only run on x86_64 since pinned values were captured on that platform.
// ARM may produce slightly different results due to FP implementation differences.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_simd_scores_pinned_real_images() {
    let source = load_image("source.png");

    for case in REAL_IMAGE_CASES {
        let distorted = load_image(case.distorted_file);
        let score =
            compute_ssimulacra2_with_config(source.clone(), distorted, Ssimulacra2Config::simd())
                .unwrap();

        // Exact match - any deviation indicates a regression
        assert!(
            (score - case.expected_simd).abs() < 1e-5,
            "{}: SIMD score changed! expected={:.6}, got={:.6}. \
             If intentional, update expected_simd in test.",
            case.name,
            case.expected_simd,
            score
        );
    }
}

#[test]
fn test_scalar_vs_simd_real_images() {
    let source = load_image("source.png");

    for case in REAL_IMAGE_CASES {
        let distorted = load_image(case.distorted_file);

        let scalar_score = compute_ssimulacra2_with_config(
            source.clone(),
            distorted.clone(),
            Ssimulacra2Config::scalar(),
        )
        .unwrap();

        let simd_score =
            compute_ssimulacra2_with_config(source.clone(), distorted, Ssimulacra2Config::simd())
                .unwrap();

        let diff = (scalar_score - simd_score).abs();
        // 1% relative tolerance for FP differences between f64 scalar and f32 SIMD
        let tolerance = simd_score.abs() * 0.01;

        assert!(
            diff < tolerance,
            "{}: Scalar vs SIMD mismatch. scalar={:.6}, simd={:.6}, diff={:.6}, tolerance={:.6}",
            case.name,
            scalar_score,
            simd_score,
            diff,
            tolerance
        );
    }
}

// ============================================================================
// Synthetic image tests - for broader coverage
// ============================================================================

#[test]
fn test_scalar_vs_simd_synthetic() {
    let sizes = [(64, 64), (256, 256), (512, 512)];

    for (width, height) in sizes {
        let (source_data, distorted_data) = create_synthetic_images(width, height);

        let scalar_score = compute_score_from_data(
            &source_data,
            &distorted_data,
            width,
            height,
            Ssimulacra2Config::scalar(),
        );
        let simd_score = compute_score_from_data(
            &source_data,
            &distorted_data,
            width,
            height,
            Ssimulacra2Config::simd(),
        );

        let diff = (scalar_score - simd_score).abs();
        let tolerance = scalar_score.abs() * 0.01;

        assert!(
            diff < tolerance,
            "{}x{}: Scalar vs SIMD mismatch. scalar={:.6}, simd={:.6}, diff={:.6}",
            width,
            height,
            scalar_score,
            simd_score,
            diff
        );
    }
}

// ============================================================================
// Quality ordering test - higher quality = higher score
// ============================================================================

#[test]
fn test_jpeg_quality_ordering_preserved() {
    let source = load_image("source.png");
    let files = ["q20.jpg", "q45.jpg", "q70.jpg", "q90.jpg"];

    let mut prev_score = f64::NEG_INFINITY;

    for file in files {
        let distorted = load_image(file);
        let score =
            compute_ssimulacra2_with_config(source.clone(), distorted, Ssimulacra2Config::simd())
                .unwrap();

        assert!(
            score > prev_score,
            "{} score ({:.6}) should be > previous ({:.6})",
            file,
            score,
            prev_score
        );
        prev_score = score;
    }
}
