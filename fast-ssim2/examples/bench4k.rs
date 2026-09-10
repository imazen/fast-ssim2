//! One number per run: 4K `compute_ssimulacra2`, median of N iterations.
//!
//! The `benches` target sweeps sizes and takes ~11 minutes on a slow box, which
//! makes a fleet-wide comparison impractical. This does the one size that
//! matters for throughput planning and prints a single line, so the same
//! command can be pasted at every machine and the results diffed.
//!
//!   cargo run --release --example bench4k                    # single-threaded
//!   cargo run --release --features rayon --example bench4k   # multi-threaded
//!
//!   BENCH4K_ITERS=10   how many timed iterations (default 7)
//!   BENCH4K_SIZE=WxH   default 3840x2160

use std::time::Instant;

use fast_ssim2::{LinearRgbImage, compute_ssimulacra2, srgb_u8_to_linear};

fn main() {
    let (w, h) = std::env::var("BENCH4K_SIZE")
        .ok()
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((3840usize, 2160usize));
    let iters: usize = std::env::var("BENCH4K_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);

    // Structured content, so every pyramid level has something to score, and
    // deterministic so every machine measures the same work.
    let mut s: u64 = 0x0bad_c0de_dead_beef;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) & 0xFF) as u8
    };
    let src: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            let x = (i % w) as f32 / w as f32;
            let y = (i / w) as f32 / h as f32;
            let ring = (((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt() * 40.0).sin() * 0.5 + 0.5;
            let noise = f32::from(next()) / 255.0;
            [
                srgb_u8_to_linear(((x * 0.5 + ring * 0.5) * 255.0) as u8),
                srgb_u8_to_linear(((y * 0.6 + 0.2) * 255.0) as u8),
                srgb_u8_to_linear(((ring * 0.7 + noise * 0.3) * 255.0) as u8),
            ]
        })
        .collect();
    let dist: Vec<[f32; 3]> = src
        .iter()
        .map(|p| [p[0] * 0.97 + 0.01, p[1], p[2] * 1.02])
        .collect();

    let a = LinearRgbImage::new(src, w, h);
    let b = LinearRgbImage::new(dist, w, h);

    // Warm up: first call touches pages and grows the blur's buffers.
    let warm = compute_ssimulacra2(a.clone(), b.clone()).expect("score");

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let s = compute_ssimulacra2(a.clone(), b.clone()).expect("score");
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(s);
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let threads = if cfg!(feature = "rayon") {
        std::thread::available_parallelism().map_or(0, |n| n.get())
    } else {
        1
    };
    println!(
        "{w}x{h}\tthreads={threads}\tmedian={median:.1}ms\tmin={:.1}\tmax={:.1}\tscore={warm:.6}",
        times[0],
        times[times.len() - 1]
    );
}
