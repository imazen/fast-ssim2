//! Compare peak heap memory between the full-image and strip-walker
//! SSIMULACRA2 paths.
//!
//! Run under heaptrack to measure:
//!
//! ```
//! cargo build --release --example strip_memory
//! heaptrack -o /tmp/ss2_full   ../target/release/examples/strip_memory full   7680 5120
//! heaptrack -o /tmp/ss2_strip  ../target/release/examples/strip_memory strip  7680 5120 256
//! heaptrack -o /tmp/ss2_wstrip ../target/release/examples/strip_memory wstrip 7680 5120 256
//! heaptrack_print /tmp/ss2_full.zst   | head -40
//! heaptrack_print /tmp/ss2_strip.zst  | head -40
//! heaptrack_print /tmp/ss2_wstrip.zst | head -40
//! ```
//!
//! Without heaptrack, the program just prints the score so the
//! measurement loop is end-to-end correct.

use std::time::Instant;

use fast_ssim2::{
    LinearRgbImage, Ssimulacra2Reference, compute_ssimulacra2, compute_ssimulacra2_strip,
};

fn make_pair(width: usize, height: usize) -> (LinearRgbImage, LinearRgbImage) {
    let mut src = Vec::with_capacity(width * height);
    let mut dst = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (((x * 7 + y * 13) & 0xff) as f32) / 255.0;
            let g = (((x * 11 + y * 3 + 50) & 0xff) as f32) / 255.0;
            let b = (((x * 5 + y * 17 + 100) & 0xff) as f32) / 255.0;
            src.push([r, g, b]);
            // Slight per-pixel shift for distorted.
            let rd = ((r + 0.02).min(1.0)).max(0.0);
            let gd = ((g - 0.01).min(1.0)).max(0.0);
            let bd = ((b + 0.005).min(1.0)).max(0.0);
            dst.push([rd, gd, bd]);
        }
    }
    (
        LinearRgbImage::new(src, width, height),
        LinearRgbImage::new(dst, width, height),
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "full".to_string());
    let width: usize = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7680);
    let height: usize = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5120);
    let strip_h: u32 = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    eprintln!("fast-ssim2 strip_memory: mode={mode} size={width}x{height} strip_h={strip_h}");
    let t0 = Instant::now();
    let (src, dst) = make_pair(width, height);
    eprintln!("  fixture built in {:.2}s", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    let score = match mode.as_str() {
        "full" => compute_ssimulacra2(src, dst).expect("full"),
        "strip" => compute_ssimulacra2_strip(src, dst, strip_h).expect("strip"),
        "wstrip" => {
            let r = Ssimulacra2Reference::new(src).expect("ref");
            r.compare_strip(dst, strip_h).expect("compare_strip")
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    };
    eprintln!("  score = {score:.6} in {:.2}s", t1.elapsed().as_secs_f64());
    println!("{score:.6}");
}
