//! Cross-path API contracts: original dimensions, options, and pixel semantics.
use fast_ssim2::{
    PixelDescriptor, PixelSlice, Ssimulacra2Config, Ssimulacra2Error, Ssimulacra2Reference,
    compute_ssimulacra2_with_config,
};

fn rgb(bytes: &[u8], w: u32, h: u32) -> PixelSlice<'_> {
    PixelSlice::new(bytes, w, h, w as usize * 3, PixelDescriptor::RGB8_SRGB).unwrap()
}
fn floats(data: &[f32], w: u32, h: u32, desc: PixelDescriptor) -> PixelSlice<'_> {
    PixelSlice::new(
        bytemuck::cast_slice(data),
        w,
        h,
        w as usize * desc.format.bytes_per_pixel(),
        desc,
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
                floats(&af, w as u32, h as u32, PixelDescriptor::RGBF32_LINEAR)
            } else {
                rgb(&a, w as u32, h as u32)
            };
            let b = if linear {
                floats(&bf, dw as u32, dh as u32, PixelDescriptor::RGBF32_LINEAR)
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
            floats(&rf, w as u32, w as u32, PixelDescriptor::RGBF32_LINEAR)
        } else {
            rgb(&rgb8, w as u32, w as u32)
        };
        let aa = if linear {
            floats(&raf, w as u32, w as u32, PixelDescriptor::RGBAF32_LINEAR)
        } else {
            PixelSlice::new(
                &rgba8,
                w as u32,
                w as u32,
                w * 4,
                PixelDescriptor::RGBA8_SRGB,
            )
            .unwrap()
        };
        let d = if linear {
            floats(&df, w as u32, w as u32, PixelDescriptor::RGBAF32_LINEAR)
        } else {
            PixelSlice::new(
                &dist8,
                w as u32,
                w as u32,
                w * 4,
                PixelDescriptor::RGBA8_SRGB,
            )
            .unwrap()
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
fn hdr_composites_in_nits_and_rejects_strips() {
    use fast_ssim2::{compute_ssimulacra2_pu, compute_ssimulacra2_pu_with_config};
    let data: Vec<f32> = (0..64)
        .flat_map(|i| [20.0 + i as f32, 100.0, 300.0, 0.25])
        .collect();
    let other: Vec<f32> = (0..64)
        .flat_map(|i| [40.0, 80.0 + i as f32, 200.0, 0.75])
        .collect();
    let desc = PixelDescriptor::RGBAF32_LINEAR;
    let a = floats(&data, 8, 8, desc);
    let b = floats(&other, 8, 8, desc);
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
                &floats(&ac, 8, 8, PixelDescriptor::RGBF32_LINEAR),
                &floats(&bc, 8, 8, PixelDescriptor::RGBF32_LINEAR),
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
