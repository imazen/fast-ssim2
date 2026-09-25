//! Benchmark: Ssimulacra2Reference (cached ref) vs one-shot compute.
//!
//! Run with: cargo run --release --features imgref --example precompute_benchmark

use fast_ssim2::{Ssimulacra2Reference, compute_ssimulacra2};
use std::time::Instant;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn main() {
    let sizes = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)];
    let iterations = 20;

    println!("SSIMULACRA2 Precompute Benchmark\n");
    println!(
        "{:>12} {:>6} {:>14} {:>14} {:>10}",
        "Size", "Iters", "One-shot", "Cached-compare", "Speedup"
    );
    println!("{:-<64}", "");

    for (width, height) in sizes {
        let reference_data: Vec<[f32; 3]> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f32 / width as f32;
                let y = (i / width) as f32 / height as f32;
                [x, y, 0.5]
            })
            .collect();
        let distorted_data: Vec<[f32; 3]> = reference_data
            .iter()
            .map(|&[r, g, b]| [(r * 1.05).min(1.0), g, b])
            .collect();

        let nz_width = std::num::NonZeroUsize::new(width).unwrap();
        let nz_height = std::num::NonZeroUsize::new(height).unwrap();
        let mk = |d: &Vec<[f32; 3]>| {
            Rgb::new(
                d.clone(),
                nz_width,
                nz_height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap()
        };

        // One-shot
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = compute_ssimulacra2(mk(&reference_data), mk(&distorted_data)).unwrap();
        }
        let full_time = start.elapsed() / iterations as u32;

        let reference = mk(&reference_data);
        let precomputed = Ssimulacra2Reference::new(reference).unwrap();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = precomputed.compare(mk(&distorted_data)).unwrap();
        }
        let compare_time = start.elapsed() / iterations as u32;

        println!(
            "{:>5}x{:<5} {:>6} {:>11.2?} {:>11.2?} {:>9.2}x",
            width,
            height,
            iterations,
            full_time,
            compare_time,
            full_time.as_secs_f64() / compare_time.as_secs_f64(),
        );
    }

    println!("\n`compare` skips the reference-side pipeline (~36% of one-shot work).");
}
