use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};
use fast_ssim2::{
    Blur, ColorPrimaries, Frame, MatrixCoefficients, Plane, Rgb, TransferCharacteristic, Yuv,
    YuvConfig, compute_frame_ssimulacra2, compute_ssimulacra2,
};
use num_traits::clamp;
use rand::RngExt;
use std::hint::black_box;

fn make_yuv_sized(
    width: usize,
    height: usize,
    ss: (u8, u8),
    full_range: bool,
    mc: MatrixCoefficients,
    tc: TransferCharacteristic,
    cp: ColorPrimaries,
) -> Yuv<u8> {
    let uv_dims = (width >> ss.0, height >> ss.1);
    let mut data: Frame<u8> = Frame {
        planes: [
            Plane::new(width, height, 0, 0, 0, 0),
            Plane::new(
                uv_dims.0,
                uv_dims.1,
                usize::from(ss.0),
                usize::from(ss.1),
                0,
                0,
            ),
            Plane::new(
                uv_dims.0,
                uv_dims.1,
                usize::from(ss.0),
                usize::from(ss.1),
                0,
                0,
            ),
        ],
    };
    let mut rng = rand::rng();
    for (i, plane) in data.planes.iter_mut().enumerate() {
        for val in plane.data_origin_mut().iter_mut() {
            *val = rng.random_range(if full_range {
                0..=255
            } else if i == 0 {
                16..=235
            } else {
                16..=240
            });
        }
    }
    Yuv::new(
        data,
        YuvConfig {
            bit_depth: 8,
            subsampling_x: ss.0,
            subsampling_y: ss.1,
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
    let mut planes = [
        input.data()[0].clone(),
        input.data()[1].clone(),
        input.data()[2].clone(),
    ];
    for plane in &mut planes {
        for pix in plane.data_origin_mut() {
            *pix = clamp(i16::from(*pix) + rng.random_range(-16..=16), 0, 255) as u8;
        }
    }
    let data: Frame<u8> = Frame { planes };
    Yuv::new(data, input.config()).unwrap()
}

fn make_test_pair(width: usize, height: usize) -> (Yuv<u8>, Yuv<u8>) {
    let input = make_yuv_sized(
        width,
        height,
        (0, 0),
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

    let source = Rgb::new(
        source_data,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let distorted = Rgb::new(
        distorted_data,
        width,
        height,
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
