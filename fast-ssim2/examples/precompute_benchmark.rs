//! Benchmark comparing Ssim2Reference precomputation vs full computation.
//!
//! Run with: cargo run --release --example precompute_benchmark

use fast_ssim2::{Ssimulacra2Reference, compute_ssimulacra2};
use std::time::Instant;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn main() {
    let sizes = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)];
    let iterations = 20;

    println!("SSIMULACRA2 Precompute Benchmark\n");
    println!(
        "{:>12} {:>6} {:>14} {:>14} {:>14} {:>10} {:>10}",
        "Size", "Iters", "Full", "compare", "compare_with", "vs_full", "vs_compare"
    );
    println!("{:-<92}", "");

    for (width, height) in sizes {
        // Create reference and distorted test images
        let reference_data: Vec<[f32; 3]> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f32 / width as f32;
                let y = (i / width) as f32 / height as f32;
                [x, y, 0.5]
            })
            .collect();

        let distorted_data: Vec<[f32; 3]> = reference_data
            .iter()
            .map(|&[r, g, b]| [r * 0.9, g * 0.95, b * 1.05])
            .collect();

        // Benchmark full computation (both source and distorted processed each time)
        let start = Instant::now();
        for _ in 0..iterations {
            let nz_width = std::num::NonZeroUsize::new(width).unwrap();
            let nz_height = std::num::NonZeroUsize::new(height).unwrap();
            let source = Rgb::new(
                reference_data.clone(),
                nz_width,
                nz_height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap();
            let distorted = Rgb::new(
                distorted_data.clone(),
                nz_width,
                nz_height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap();
            let _ = compute_ssimulacra2(source, distorted).unwrap();
        }
        let full_time = start.elapsed() / iterations as u32;

        // Benchmark precomputed (source processed once, distorted many times)
        let nz_width = std::num::NonZeroUsize::new(width).unwrap();
        let nz_height = std::num::NonZeroUsize::new(height).unwrap();
        let reference = Rgb::new(
            reference_data.clone(),
            nz_width,
            nz_height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        // Precompute reference once (not counted in benchmark)
        let precomputed = Ssimulacra2Reference::new(reference).unwrap();

        // Benchmark compare() — allocates working buffers per call
        let start = Instant::now();
        for _ in 0..iterations {
            let distorted = Rgb::new(
                distorted_data.clone(),
                nz_width,
                nz_height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap();
            let _ = precomputed.compare(distorted).unwrap();
        }
        let precompute_time = start.elapsed() / iterations as u32;

        // Benchmark compare_with() — zero-alloc after first call
        let mut ctx = precomputed.compare_context();
        let start = Instant::now();
        for _ in 0..iterations {
            let distorted = Rgb::new(
                distorted_data.clone(),
                nz_width,
                nz_height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap();
            let _ = precomputed.compare_with(&mut ctx, distorted).unwrap();
        }
        let compare_with_time = start.elapsed() / iterations as u32;

        let speedup_full = full_time.as_secs_f64() / compare_with_time.as_secs_f64();
        let speedup_compare = precompute_time.as_secs_f64() / compare_with_time.as_secs_f64();

        println!(
            "{:>5}x{:<5} {:>6} {:>11.2?} {:>11.2?} {:>11.2?} {:>9.2}x {:>9.2}x",
            width,
            height,
            iterations,
            full_time,
            precompute_time,
            compare_with_time,
            speedup_full,
            speedup_compare
        );
    }

    println!("\nNote: Precomputed benchmark excludes one-time reference preprocessing.");
    println!("For simulated annealing with 1000+ iterations, speedup approaches 2x.");
}
