//! Cross-path API contracts: original dimensions, options, and pixel semantics.
use fast_ssim2::{
    PixelFormat, PixelView, Ssimulacra2Config, Ssimulacra2Error, Ssimulacra2Reference,
    compute_ssimulacra2_with_config, formats,
};

fn rgb(bytes: &[u8], w: u32, h: u32) -> PixelView<'_> {
    PixelView::new(bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap()
}
fn floats<'a>(data: &'a [f32], w: u32, h: u32, fmt: &'a PixelFormat) -> PixelView<'a> {
    PixelView::new(
        bytemuck::cast_slice(data),
        w,
        h,
        w as usize * fmt.layout.bytes_per_pixel(),
        fmt,
    )
    .unwrap()
}

#[test]
fn original_dimensions_must_match_on_every_path() {
    for (w, h, dw, dh) in [(1, 1, 2, 2), (7, 5, 5, 7), (8, 8, 16, 16), (16, 16, 8, 8)] {
        let a = vec![128; w * h * 3];
        let b = vec![128; dw * dh * 3];
        let af = vec![0.5; w * h * 3];
        let bf = vec![0.5; dw * dh * 3];
        for linear in [false, true] {
            let a = if linear {
                floats(&af, w as u32, h as u32, &formats::RGBF32_LINEAR)
            } else {
                rgb(&a, w as u32, h as u32)
            };
            let b = if linear {
                floats(&bf, dw as u32, dh as u32, &formats::RGBF32_LINEAR)
            } else {
                rgb(&b, dw as u32, dh as u32)
            };
            let cache = Ssimulacra2Reference::new(&a).unwrap();
            for cfg in [Ssimulacra2Config::default(), Ssimulacra2Config::strips(8)] {
                assert_eq!(
                    compute_ssimulacra2_with_config(&a, &b, &cfg),
                    Err(Ssimulacra2Error::NonMatchingImageDimensions)
                );
                assert_eq!(
                    cache.compare_with_config(&b, &cfg),
                    Err(Ssimulacra2Error::NonMatchingImageDimensions)
                );
            }
        }
    }
}

#[test]
fn cancelled_reference_construction_and_unsupported_options_error() {
    let data = vec![128; 8 * 8 * 3];
    let source = rgb(&data, 8, 8);
    let stop = almost_enough::Stopper::cancelled();
    assert!(matches!(
        Ssimulacra2Reference::new_with_config(
            &source,
            &Ssimulacra2Config::default().with_stop(&stop)
        ),
        Err(Ssimulacra2Error::Cancelled(_))
    ));
    assert!(matches!(
        Ssimulacra2Reference::new_with_config(&source, &Ssimulacra2Config::strips(8)),
        Err(Ssimulacra2Error::InvalidConfiguration(_))
    ));
    assert!(matches!(
        compute_ssimulacra2_with_config(&source, &source, &Ssimulacra2Config::strips(0)),
        Err(Ssimulacra2Error::InvalidConfiguration(_))
    ));
}

#[test]
fn alpha_only_on_distorted_side_matches_explicit_opaque_reference() {
    let w = 16;
    for linear in [false, true] {
        let rgb8: Vec<u8> = (0..w * w * 3).map(|i| (i * 17 % 251) as u8).collect();
        let rgba8: Vec<u8> = rgb8
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        let dist8: Vec<u8> = rgba8
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, p)| [p[0], p[1], p[2], (i * 31 % 255) as u8])
            .collect();
        let rf: Vec<f32> = rgb8.iter().map(|&v| v as f32 / 255.0).collect();
        let raf: Vec<f32> = rgba8.iter().map(|&v| v as f32 / 255.0).collect();
        let df: Vec<f32> = dist8.iter().map(|&v| v as f32 / 255.0).collect();
        let a = if linear {
            floats(&rf, w as u32, w as u32, &formats::RGBF32_LINEAR)
        } else {
            rgb(&rgb8, w as u32, w as u32)
        };
        let aa = if linear {
            floats(&raf, w as u32, w as u32, &formats::RGBAF32_LINEAR)
        } else {
            PixelView::new(&rgba8, w as u32, w as u32, w * 4, &formats::RGBA8_SRGB).unwrap()
        };
        let d = if linear {
            floats(&df, w as u32, w as u32, &formats::RGBAF32_LINEAR)
        } else {
            PixelView::new(&dist8, w as u32, w as u32, w * 4, &formats::RGBA8_SRGB).unwrap()
        };
        for backend in [Ssimulacra2Config::scalar(), Ssimulacra2Config::simd()] {
            let cache = Ssimulacra2Reference::new_with_config(&a, &backend).unwrap();
            let expected = compute_ssimulacra2_with_config(&aa, &d, &backend).unwrap();
            assert_eq!(
                compute_ssimulacra2_with_config(&a, &d, &backend).unwrap(),
                expected
            );
            assert_eq!(cache.compare_with_config(&d, &backend).unwrap(), expected);
            let cfg = backend.with_strip(fast_ssim2::StripConfig::new(8));
            assert_eq!(
                cache.compare_with_config(&d, &cfg).unwrap(),
                compute_ssimulacra2_with_config(&a, &d, &cfg).unwrap()
            );
        }
    }
}

#[cfg(feature = "hdr-pu")]
#[test]
#[allow(deprecated)] // The legacy PU21 contract remains covered during migration.
fn hdr_composites_in_nits_and_rejects_strips() {
    use fast_ssim2::{compute_ssimulacra2_pu, compute_ssimulacra2_pu_with_config};
    let data: Vec<f32> = (0..64)
        .flat_map(|i| [20.0 + i as f32, 100.0, 300.0, 0.25])
        .collect();
    let other: Vec<f32> = (0..64)
        .flat_map(|i| [40.0, 80.0 + i as f32, 200.0, 0.75])
        .collect();
    let a = floats(&data, 8, 8, &formats::RGBAF32_LINEAR);
    let b = floats(&other, 8, 8, &formats::RGBAF32_LINEAR);
    let mut expected = f64::INFINITY;
    for bg in [20.0, 200.0] {
        let composite = |p: &[f32]| {
            p.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    [
                        p[3] * p[0] + (1.0 - p[3]) * bg,
                        p[3] * p[1] + (1.0 - p[3]) * bg,
                        p[3] * p[2] + (1.0 - p[3]) * bg,
                    ]
                })
                .collect::<Vec<_>>()
        };
        let ac = composite(&data);
        let bc = composite(&other);
        expected = expected.min(
            compute_ssimulacra2_pu(
                &floats(&ac, 8, 8, &formats::RGBF32_LINEAR),
                &floats(&bc, 8, 8, &formats::RGBF32_LINEAR),
            )
            .unwrap(),
        );
    }
    assert_eq!(compute_ssimulacra2_pu(&a, &b).unwrap(), expected);
    assert!(matches!(
        compute_ssimulacra2_pu_with_config(&a, &b, &Ssimulacra2Config::strips(8)),
        Err(Ssimulacra2Error::InvalidConfiguration(_))
    ));
}

#[test]
fn integer_grayscale_matches_expanded_rgb() {
    fn mk<'a>(data: &'a [u8], fmt: &'a PixelFormat) -> PixelView<'a> {
        PixelView::new(data, 8, 8, 8 * fmt.layout.bytes_per_pixel(), fmt).unwrap()
    }

    let gray: Vec<u8> = (0..64).map(|i| (i * 37 % 255) as u8).collect();
    let other: Vec<u8> = gray.iter().map(|&v| v.saturating_add(7)).collect();

    // 8-bit
    {
        let expand8 = |data: &[u8], channels: usize| -> Vec<u8> {
            data.iter()
                .flat_map(|&v| std::iter::repeat_n(v, channels))
                .collect()
        };
        let a = expand8(&gray, 1);
        let b = expand8(&other, 1);
        let ar = expand8(&gray, 3);
        let br = expand8(&other, 3);
        let (gd, rd) = (&formats::GRAY8_SRGB, &formats::RGB8_SRGB);
        for cfg in [Ssimulacra2Config::default(), Ssimulacra2Config::strips(8)] {
            let expected =
                compute_ssimulacra2_with_config(&mk(&ar, rd), &mk(&br, rd), &cfg).unwrap();
            assert_eq!(
                compute_ssimulacra2_with_config(&mk(&a, gd), &mk(&b, gd), &cfg).unwrap(),
                expected
            );
            assert_eq!(
                Ssimulacra2Reference::new(&mk(&a, gd))
                    .unwrap()
                    .compare_with_config(&mk(&b, gd), &cfg)
                    .unwrap(),
                expected
            );
        }
    }

    // 16-bit (backed by Vec<u16> for guaranteed 2-byte alignment)
    {
        let expand16 = |data: &[u8], channels: usize| -> Vec<u16> {
            data.iter()
                .flat_map(|&v| std::iter::repeat_n(v as u16 * 257, channels))
                .collect()
        };
        let a = expand16(&gray, 1);
        let b = expand16(&other, 1);
        let ar = expand16(&gray, 3);
        let br = expand16(&other, 3);
        let (gd, rd) = (&formats::GRAY16_SRGB, &formats::RGB16_SRGB);
        let a_b = bytemuck::cast_slice(&a);
        let b_b = bytemuck::cast_slice(&b);
        let ar_b = bytemuck::cast_slice(&ar);
        let br_b = bytemuck::cast_slice(&br);
        for cfg in [Ssimulacra2Config::default(), Ssimulacra2Config::strips(8)] {
            let expected =
                compute_ssimulacra2_with_config(&mk(ar_b, rd), &mk(br_b, rd), &cfg).unwrap();
            assert_eq!(
                compute_ssimulacra2_with_config(&mk(a_b, gd), &mk(b_b, gd), &cfg).unwrap(),
                expected
            );
            assert_eq!(
                Ssimulacra2Reference::new(&mk(a_b, gd))
                    .unwrap()
                    .compare_with_config(&mk(b_b, gd), &cfg)
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn extreme_strip_height_is_clamped_without_overflow() {
    let data = vec![128; 8 * 8 * 3];
    let image = rgb(&data, 8, 8);
    let cfg = Ssimulacra2Config::default()
        .with_strip(fast_ssim2::StripConfig::new(usize::MAX).with_halo_rows(usize::MAX));
    assert_eq!(
        compute_ssimulacra2_with_config(&image, &image, &cfg).unwrap(),
        100.0
    );
    assert_eq!(
        Ssimulacra2Reference::new(&image)
            .unwrap()
            .compare_with_config(&image, &cfg)
            .unwrap(),
        100.0
    );
}

#[cfg(not(feature = "rayon"))]
#[test]
fn parallel_strips_require_rayon() {
    let data = vec![128; 8 * 8 * 3];
    let image = rgb(&data, 8, 8);
    let cfg = Ssimulacra2Config::default()
        .with_strip(fast_ssim2::StripConfig::default().with_parallel_strips(true));
    assert!(matches!(
        compute_ssimulacra2_with_config(&image, &image, &cfg),
        Err(Ssimulacra2Error::InvalidConfiguration(_))
    ));
}
