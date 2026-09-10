//! Regression guard for the horizontal blur's 4 KiB-aliasing cliff.
//!
//! The horizontal pass runs the IIR over eight rows at once, one row per SIMD
//! lane, so each column access is eight loads at stride `width * 4` bytes. At a
//! power-of-two width those eight addresses are congruent modulo 4096; if the
//! destination plane is congruent with the source as well, the loads, the
//! stores and each other collide in one cache set. Before
//! `SimdGaussian::temp_offset` existed, that cost **7.6×** in the pass alone on
//! Zen 4 (5.34 vs 0.70 ns/px at width 1024) and **35%** end-to-end on an M4 Pro
//! at width 4096 (119.85 → 77.82 ms). See
//! `benchmarks/blur_stride_2026-09-09.md`.
//!
//! Each power-of-two width is paired with a neighbour eight pixels away. The
//! two should time within a few percent of each other *per pixel*; a
//! power-of-two case pulling several times its neighbour means the de-aliasing
//! regressed.
//!
//! Run: `cargo bench -p fast-ssim2 --bench blur_stride`

use fast_ssim2::{LinearRgbImage, compute_ssimulacra2, srgb_u8_to_linear};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

fn make_pair(w: usize, h: usize) -> (LinearRgbImage, LinearRgbImage) {
    let mut s: u64 = 0x5dee_ce66d;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) & 0xFF) as u8
    };
    let a: Vec<[f32; 3]> = (0..w * h)
        .map(|_| {
            [
                srgb_u8_to_linear(next()),
                srgb_u8_to_linear(next()),
                srgb_u8_to_linear(next()),
            ]
        })
        .collect();
    // A mild, structured distortion — the score does not matter here, only that
    // the full pyramid runs.
    let b: Vec<[f32; 3]> = a
        .iter()
        .map(|p| [p[0] * 0.98 + 0.01, p[1], p[2] * 1.01])
        .collect();
    (LinearRgbImage::new(a, w, h), LinearRgbImage::new(b, w, h))
}

fn bench(c: &mut Criterion) {
    // Pairs: (power-of-two width, neighbour). Same height, so a per-pixel
    // comparison is a direct one.
    for (w, h) in [
        (1024usize, 512usize),
        (1032, 512),
        (4096, 256),
        (4104, 256),
    ] {
        let (a, b) = make_pair(w, h);
        let mut group = c.benchmark_group(format!("stride_{w}x{h}"));
        group.bench_function("compute_ssimulacra2", |bencher| {
            bencher.iter(|| {
                let s = compute_ssimulacra2(a.clone(), b.clone()).unwrap();
                std::hint::black_box(s);
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
