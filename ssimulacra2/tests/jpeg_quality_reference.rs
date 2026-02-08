//! JPEG quality reference tests with C++ ssimulacra2 verified scores.
//!
//! These tests use real JPEG-compressed images at various quality levels
//! with scores verified against the C++ ssimulacra2 implementation.
//!
//! Test images: 256x256 crop from tank_source.png
//! Total size: ~116KB
//!
//! C++ reference binary: libjxl/build/tools/ssimulacra2
//! Captured: 2026-01-04

use fast_ssim2::{Ssimulacra2Config, compute_frame_ssimulacra2};
use image::ImageReader;
use std::path::PathBuf;
use yuvxyb::Rgb;

/// JPEG quality test case with C++ verified score
struct JpegQualityCase {
    name: &'static str,
    quality: u8,
    filename: &'static str,
    /// Score from C++ ssimulacra2 binary
    cpp_score: f64,
}

const JPEG_QUALITY_CASES: &[JpegQualityCase] = &[
    JpegQualityCase {
        name: "JPEG Q20 (low quality)",
        quality: 20,
        filename: "q20.jpg",
        cpp_score: 57.14559032,
    },
    JpegQualityCase {
        name: "JPEG Q45 (medium-low quality)",
        quality: 45,
        filename: "q45.jpg",
        cpp_score: 68.62747595,
    },
    JpegQualityCase {
        name: "JPEG Q70 (medium-high quality)",
        quality: 70,
        filename: "q70.jpg",
        cpp_score: 79.38805044,
    },
    JpegQualityCase {
        name: "JPEG Q90 (high quality)",
        quality: 90,
        filename: "q90.jpg",
        cpp_score: 90.85152474,
    },
];

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
        width as usize,
        height as usize,
        yuvxyb::TransferCharacteristic::SRGB,
        yuvxyb::ColorPrimaries::BT709,
    )
    .expect("Failed to create Rgb")
}

#[test]
fn test_jpeg_quality_vs_cpp_reference() {
    let source = load_image("source.png");

    // Maximum allowed deviation from C++ reference
    const MAX_ERROR: f64 = 1.5;

    for case in JPEG_QUALITY_CASES {
        let distorted = load_image(case.filename);

        let score = compute_frame_ssimulacra2(source.clone(), distorted)
            .expect("SSIMULACRA2 computation failed");

        let error = (score - case.cpp_score).abs();

        println!(
            "{}: Rust={:.6}, C++={:.6}, error={:.6}",
            case.name, score, case.cpp_score, error
        );

        assert!(
            error <= MAX_ERROR,
            "{}: score {} differs from C++ reference {} by {} (max allowed: {})",
            case.name,
            score,
            case.cpp_score,
            error,
            MAX_ERROR
        );
    }
}

#[test]
fn test_jpeg_quality_ordering() {
    // Verify that higher JPEG quality = higher SSIMULACRA2 score
    let source = load_image("source.png");

    let mut prev_score = f64::NEG_INFINITY;
    let mut prev_quality = 0;

    for case in JPEG_QUALITY_CASES {
        let distorted = load_image(case.filename);
        let score = compute_frame_ssimulacra2(source.clone(), distorted)
            .expect("SSIMULACRA2 computation failed");

        assert!(
            score > prev_score,
            "Q{} score ({}) should be higher than Q{} score ({})",
            case.quality,
            score,
            prev_quality,
            prev_score
        );

        prev_score = score;
        prev_quality = case.quality;
    }
}

#[test]
fn test_jpeg_quality_with_configs() {
    use fast_ssim2::compute_frame_ssimulacra2_with_config;

    let source = load_image("source.png");
    let distorted = load_image("q70.jpg");
    let cpp_score = 79.38805044;

    // Test all configurations produce similar results
    let configs = [
        ("scalar", Ssimulacra2Config::scalar()),
        ("simd", Ssimulacra2Config::simd()),
    ];

    for (name, config) in configs {
        let score =
            compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config)
                .expect("SSIMULACRA2 computation failed");

        let error = (score - cpp_score).abs();
        println!("{}: score={:.6}, error from C++={:.6}", name, score, error);

        assert!(
            error <= 1.5,
            "{} config: score {} differs from C++ reference {} by {}",
            name,
            score,
            cpp_score,
            error
        );
    }
}
