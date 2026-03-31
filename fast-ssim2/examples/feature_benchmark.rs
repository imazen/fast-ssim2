//! Comprehensive benchmark comparing all runtime implementation configurations
//!
//! Run with:
//! ```bash
//! cargo run --release --example feature_benchmark
//! ```

use fast_ssim2::{Ssimulacra2Config, compute_frame_ssimulacra2_with_config};
use std::time::Instant;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn create_test_image_512x512() -> (Rgb, Rgb) {
    let width = 512;
    let height = 512;
    let size = width * height;

    // Create a gradient pattern (common test pattern)
    let source_data: Vec<[f32; 3]> = (0..size)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            [x, y, (x + y) / 2.0]
        })
        .collect();

    // Create distorted version with slight variations
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
    source: &Rgb,
    distorted: &Rgb,
    config: Ssimulacra2Config,
    iterations: usize,
) -> (f64, f64, f64, f64) {
    // Warmup
    for _ in 0..3 {
        let _ = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config);
    }

    // Actual benchmark
    let mut times = Vec::with_capacity(iterations);
    let mut score = 0.0;
    for _ in 0..iterations {
        let start = Instant::now();
        score = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config)
            .unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0); // Convert to ms
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let p95 = times[(times.len() as f32 * 0.95) as usize];

    (mean, median, p95, score)
}

fn main() {
    println!("SSIMULACRA2 Runtime Configuration Benchmark - 512x512 images");
    println!("=============================================================\n");

    println!("Creating test images...");
    let (source, distorted) = create_test_image_512x512();
    let iterations = 50;

    println!(
        "Running benchmarks with {} iterations each...\n",
        iterations
    );
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>12}",
        "Configuration", "Mean (ms)", "Median", "P95", "Score"
    );
    println!("{:-<70}", "");

    // Scalar baseline
    let config = Ssimulacra2Config::scalar();
    let (mean, median, p95, score) = benchmark_config(&source, &distorted, config, iterations);
    println!(
        "{:<25} {:>10.3} {:>10.3} {:>10.3} {:>12.6}",
        "Scalar", mean, median, p95, score
    );

    // SIMD (archmage)
    let config = Ssimulacra2Config::simd();
    let (mean, median, p95, score) = benchmark_config(&source, &distorted, config, iterations);
    println!(
        "{:<25} {:>10.3} {:>10.3} {:>10.3} {:>12.6}",
        "SIMD (archmage)", mean, median, p95, score
    );

    println!();
}
