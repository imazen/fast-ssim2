//! Does the horizontal blur's 4K-aliasing cliff reach `compute_ssimulacra2`?
//!
//! The blur reads one plane and writes another, both plain `Vec<f32>` of
//! exactly `width * height`. For planes >= a couple of MB the allocator hands
//! back page-aligned mappings, so the two are 4K-congruent *deterministically*
//! — and the shipped horizontal pass gathers 8 rows at `width * 4` bytes, which
//! at a power-of-two width makes all 8 lanes congruent too.
//!
//! Widths straddling 1024 / 2048 tell us whether that is a microbenchmark
//! artefact or something users hit.

use fast_ssim2::{LinearRgbImage, compute_ssimulacra2, srgb_u8_to_linear};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

fn make_pair(w: usize, h: usize) -> (LinearRgbImage, LinearRgbImage) {
    let mut s: u64 = 0x5deece66d;
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
    let b: Vec<[f32; 3]> = a
        .iter()
        .map(|p| [p[0] * 0.98 + 0.01, p[1], p[2] * 1.01])
        .collect();
    (
        LinearRgbImage::new(a, w, h),
        LinearRgbImage::new(b, w, h),
    )
}

fn bench(c: &mut Criterion) {
    for (w, h) in [
        (1000usize, 1000usize),
        (1024, 1024),
        (1032, 1032),
        (2040, 1024),
        (2048, 1024),
        (2056, 1024),
        (4088, 512),
        (4096, 512),
        (4104, 512),
    ] {
        let (a, b) = make_pair(w, h);
        let mut group = c.benchmark_group(format!("ssimulacra2_{w}x{h}"));
        group.bench_function("full", |bencher| {
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
