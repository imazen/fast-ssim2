use fast_ssim2::compute_ssimulacra2;
use num_traits::clamp;
use rand::RngExt;
use std::hint::black_box;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

fn make_rgb_pair(width: usize, height: usize) -> (Rgb, Rgb) {
    let mut rng = rand::rng();
    let source_data: Vec<[f32; 3]> = (0..width * height)
        .map(|_| {
            [
                rng.random_range(0.0f32..=1.0),
                rng.random_range(0.0f32..=1.0),
                rng.random_range(0.0f32..=1.0),
            ]
        })
        .collect();

    let distorted_data: Vec<[f32; 3]> = source_data
        .iter()
        .map(|&[r, g, b]| {
            [
                clamp(r + rng.random_range(-0.05f32..=0.05), 0.0, 1.0),
                clamp(g + rng.random_range(-0.05f32..=0.05), 0.0, 1.0),
                clamp(b + rng.random_range(-0.05f32..=0.05), 0.0, 1.0),
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

fn bench_ssimulacra2(c: &mut Criterion) {
    for (w, h) in [(320, 240), (1920, 1080), (3840, 2160)] {
        let (source, distorted) = make_rgb_pair(w, h);
        c.bench_function(format!("ssimulacra2_{w}x{h}"), |b| {
            let s = source.clone();
            let d = distorted.clone();
            b.iter_batched(
                move || (s.clone(), d.clone()),
                |(s, d)| compute_ssimulacra2(black_box(s), black_box(d)).unwrap(),
                BatchSize::LargeInput,
            )
        });
    }
}

criterion_group!(benches, bench_ssimulacra2);
criterion_main!(benches);
