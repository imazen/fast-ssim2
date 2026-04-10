#![allow(deprecated)]

use fast_ssim2::{Blur, compute_frame_ssimulacra2, compute_ssimulacra2};
use num_traits::clamp;
use rand::RngExt;
use std::hint::black_box;
use yuvxyb::{ChromaSubsampling, FrameBuilder};
use yuvxyb::{
    ColorPrimaries, Frame, MatrixCoefficients, Rgb, TransferCharacteristic, Yuv, YuvConfig,
};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

fn make_yuv_sized(
    width: usize,
    height: usize,
    subsampling: ChromaSubsampling,
    full_range: bool,
    mc: MatrixCoefficients,
    tc: TransferCharacteristic,
    cp: ColorPrimaries,
) -> Yuv<u8> {
    let nz_width = std::num::NonZeroUsize::new(width).unwrap();
    let nz_height = std::num::NonZeroUsize::new(height).unwrap();
    let bit_depth = std::num::NonZeroU8::new(8).unwrap();
    let mut data: Frame<u8> = FrameBuilder::new(nz_width, nz_height, subsampling, bit_depth)
        .build::<u8>()
        .unwrap();
    let mut rng = rand::rng();
    for val in data.y_plane.pixels_mut() {
        *val = rng.random_range(if full_range { 0..=255 } else { 16..=235 });
    }
    if let Some(u_plane) = data.u_plane.as_mut() {
        for val in u_plane.pixels_mut() {
            *val = rng.random_range(if full_range { 0..=255 } else { 16..=240 });
        }
    }
    if let Some(v_plane) = data.v_plane.as_mut() {
        for val in v_plane.pixels_mut() {
            *val = rng.random_range(if full_range { 0..=255 } else { 16..=240 });
        }
    }
    let (ss_x, ss_y) = match subsampling {
        ChromaSubsampling::Yuv444 => (0u8, 0u8),
        ChromaSubsampling::Yuv422 => (1, 0),
        ChromaSubsampling::Yuv420 => (1, 1),
        ChromaSubsampling::Monochrome => (0, 0),
    };
    Yuv::new(
        data,
        YuvConfig {
            bit_depth: 8,
            subsampling_x: ss_x,
            subsampling_y: ss_y,
            full_range,
            matrix_coefficients: mc,
            transfer_characteristics: tc,
            color_primaries: cp,
        },
    )
    .unwrap()
}

fn distort_yuv(input: &Yuv<u8>) -> Yuv<u8> {
    let mut rng = rand::rng();
    let mut data = input.data().clone();
    for pix in data.y_plane.pixels_mut() {
        *pix = clamp(i16::from(*pix) + rng.random_range(-16..=16), 0, 255) as u8;
    }
    if let Some(u_plane) = data.u_plane.as_mut() {
        for pix in u_plane.pixels_mut() {
            *pix = clamp(i16::from(*pix) + rng.random_range(-16..=16), 0, 255) as u8;
        }
    }
    if let Some(v_plane) = data.v_plane.as_mut() {
        for pix in v_plane.pixels_mut() {
            *pix = clamp(i16::from(*pix) + rng.random_range(-16..=16), 0, 255) as u8;
        }
    }
    Yuv::new(data, input.config()).unwrap()
}

fn make_test_pair(width: usize, height: usize) -> (Yuv<u8>, Yuv<u8>) {
    let input = make_yuv_sized(
        width,
        height,
        ChromaSubsampling::Yuv444,
        true,
        MatrixCoefficients::BT709,
        TransferCharacteristic::BT1886,
        ColorPrimaries::BT709,
    );
    let distorted = distort_yuv(&input);
    (input, distorted)
}

fn bench_ssimulacra2(c: &mut Criterion) {
    // 320x240 (legacy)
    let (input, distorted) = make_test_pair(320, 240);
    c.bench_function("ssimulacra2_320x240", |b| {
        b.iter(|| compute_frame_ssimulacra2(black_box(&input), black_box(&distorted)).unwrap())
    });

    // 1920x1080 (FHD)
    let (input, distorted) = make_test_pair(1920, 1080);
    c.bench_function("ssimulacra2_1920x1080", |b| {
        b.iter(|| compute_frame_ssimulacra2(black_box(&input), black_box(&distorted)).unwrap())
    });

    // 3840x2160 (4K)
    let (input, distorted) = make_test_pair(3840, 2160);
    c.bench_function("ssimulacra2_3840x2160", |b| {
        b.iter(|| compute_frame_ssimulacra2(black_box(&input), black_box(&distorted)).unwrap())
    });
}

fn read_image(path: &str) -> ([Vec<f32>; 3], usize, usize) {
    let img = image::open(path).unwrap();

    let img = match img {
        image::DynamicImage::ImageRgb8(img) => img,
        x => x.to_rgb8(),
    };

    let (width, height) = img.dimensions();

    let mut img_vec = [Vec::new(), Vec::new(), Vec::new()];
    for pixel in img.pixels() {
        img_vec[0].push(pixel[0] as f32);
        img_vec[1].push(pixel[1] as f32);
        img_vec[2].push(pixel[2] as f32);
    }

    (img_vec, width as usize, height as usize)
}

fn bench_blur(c: &mut Criterion) {
    c.bench_function("blur", |b| {
        let (image, width, height) = read_image("test_data/tank_source.png");
        let mut blur = Blur::new(width, height);
        b.iter(|| blur.blur(black_box(&image)))
    });
}

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

fn bench_ssimulacra2_rgb(c: &mut Criterion) {
    // Use iter_batched to exclude clone cost from measurement
    let (source, distorted) = make_rgb_pair(320, 240);
    c.bench_function("ssimulacra2_rgb_320x240", |b| {
        let s = source.clone();
        let d = distorted.clone();
        b.iter_batched(
            move || (s.clone(), d.clone()),
            |(s, d)| compute_ssimulacra2(black_box(s), black_box(d)).unwrap(),
            BatchSize::LargeInput,
        )
    });

    let (source, distorted) = make_rgb_pair(1920, 1080);
    c.bench_function("ssimulacra2_rgb_1920x1080", |b| {
        let s = source.clone();
        let d = distorted.clone();
        b.iter_batched(
            move || (s.clone(), d.clone()),
            |(s, d)| compute_ssimulacra2(black_box(s), black_box(d)).unwrap(),
            BatchSize::LargeInput,
        )
    });

    let (source, distorted) = make_rgb_pair(3840, 2160);
    c.bench_function("ssimulacra2_rgb_3840x2160", |b| {
        let s = source.clone();
        let d = distorted.clone();
        b.iter_batched(
            move || (s.clone(), d.clone()),
            |(s, d)| compute_ssimulacra2(black_box(s), black_box(d)).unwrap(),
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(
    benches,
    bench_ssimulacra2,
    bench_ssimulacra2_rgb,
    bench_blur
);
criterion_main!(benches);
