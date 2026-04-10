//! Profile SIMD implementation performance
//!
//! Run with: cargo run --release --example profile_simd

#![allow(deprecated)]

use fast_ssim2::{Blur, SimdImpl, Ssimulacra2Config, compute_frame_ssimulacra2_with_config};
use std::time::Instant;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn create_test_images(width: usize, height: usize) -> (Rgb, Rgb) {
    let size = width * height;

    let source_data: Vec<[f32; 3]> = (0..size)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            [x, y, (x + y) / 2.0]
        })
        .collect();

    let distorted_data: Vec<[f32; 3]> = source_data
        .iter()
        .map(|&[r, g, b]| {
            [
                (r * 0.95).min(1.0),
                (g * 1.02).min(1.0),
                (b * 0.98).min(1.0),
            ]
        })
        .collect();

    let nz_width = std::num::NonZeroUsize::new(width).unwrap();
    let nz_height = std::num::NonZeroUsize::new(height).unwrap();
    let source = Rgb::new(
        source_data,
        nz_width,
        nz_height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let distorted = Rgb::new(
        distorted_data,
        nz_width,
        nz_height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    (source, distorted)
}

fn benchmark_config(
    name: &str,
    config: Ssimulacra2Config,
    source: &Rgb,
    distorted: &Rgb,
    iterations: usize,
) -> f64 {
    // Warmup
    for _ in 0..3 {
        let _ = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config);
    }

    let start = Instant::now();
    let mut score = 0.0;
    for _ in 0..iterations {
        score = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config)
            .unwrap();
    }
    let elapsed = start.elapsed();
    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!("  {:<30} {:.2}ms  score={:.6}", name, ms_per_iter, score);
    ms_per_iter
}

fn main() {
    println!("SSIMULACRA2 SIMD Profiler");
    println!("=========================\n");

    let sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)];
    let iterations = 20;

    for (width, height) in sizes {
        println!(
            "Image size: {}x{} ({} iterations)",
            width, height, iterations
        );
        println!("{:-<60}", "");

        let (source, distorted) = create_test_images(width, height);

        let scalar_ms = benchmark_config(
            "Scalar",
            Ssimulacra2Config::scalar(),
            &source,
            &distorted,
            iterations,
        );
        let simd_ms = benchmark_config(
            "SIMD (archmage)",
            Ssimulacra2Config::simd(),
            &source,
            &distorted,
            iterations,
        );

        println!();
        println!("  Speedup vs Scalar:");
        println!("    SIMD: {:.2}x", scalar_ms / simd_ms);

        println!();
    }

    // Component-level profiling for blur (the biggest time consumer)
    println!("\n=== Blur Component Breakdown (1024x1024) ===\n");
    blur_profile(1024, 1024);
}

fn blur_profile(width: usize, height: usize) {
    let iterations = 50;

    // Create test planar data
    let planar: [Vec<f32>; 3] = [
        vec![0.5f32; width * height],
        vec![0.5f32; width * height],
        vec![0.5f32; width * height],
    ];

    println!(
        "Blur (per 3-channel blur operation, {} iterations):",
        iterations
    );

    // Scalar blur
    let mut blur_scalar = Blur::with_simd_impl(width, height, SimdImpl::Scalar);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = blur_scalar.blur(&planar);
    }
    let scalar_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("  Scalar:          {:.3}ms", scalar_ms);

    // SIMD blur
    let mut blur_simd = Blur::with_simd_impl(width, height, SimdImpl::Simd);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = blur_simd.blur(&planar);
    }
    let simd_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "  SIMD (archmage): {:.3}ms ({:.2}x vs scalar)",
        simd_ms,
        scalar_ms / simd_ms
    );
}
