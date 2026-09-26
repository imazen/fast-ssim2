use fast_ssim2::compute_ssimulacra2;
use num_traits::clamp;
use rand::RngExt;
use std::hint::black_box;

/// Owned encoded-sRGB `f32` pixels (k/255 grid stays LUT-exact).
fn srgb_f32_owned(data: Vec<[f32; 3]>, w: usize, h: usize) -> zenpixels::PixelBuffer {
    zenpixels::PixelBuffer::from_vec(
        bytemuck::cast_slice::<f32, u8>(data.as_flattened()).to_vec(),
        w as u32,
        h as u32,
        zenpixels::PixelDescriptor::RGBF32.with_transfer(zenpixels::TransferFunction::Srgb),
    )
    .unwrap()
}
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

fn make_rgb_pair(width: usize, height: usize) -> (zenpixels::PixelBuffer, zenpixels::PixelBuffer) {
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

    let source = srgb_f32_owned(source_data, width, height);

    let distorted = srgb_f32_owned(distorted_data, width, height);

    (source, distorted)
}

fn bench_ssimulacra2(c: &mut Criterion) {
    for (w, h) in [(320, 240), (1920, 1080), (3840, 2160)] {
        let (source, distorted) = make_rgb_pair(w, h);
        c.bench_function(format!("ssimulacra2_{w}x{h}"), |b| {
            b.iter(
                || compute_ssimulacra2(
                    black_box(&source.as_slice()),
                    black_box(&distorted.as_slice()),
                )
                .unwrap(),
            )
        });
    }
}

criterion_group!(benches, bench_ssimulacra2);
criterion_main!(benches);
