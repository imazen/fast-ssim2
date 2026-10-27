//! Native reference HDR conversion and public-path consistency.
#[path = "common/native_hdr.rs"]
mod native_hdr;
use fast_ssim2::{
    PixelDescriptor, PixelSlice, Ssimulacra2Config, Ssimulacra2Error, Ssimulacra2Reference,
    StripConfig, TransferFunction, compute_ssimulacra2, compute_ssimulacra2_maps_with_config,
    compute_ssimulacra2_with_config,
};
use native_hdr::Case;

#[test]
fn native_hdr_scores_maps_and_cpp_reference() {
    for line in include_str!("native_hdr_scores.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        let parse = |f: &[&str]| Case {
            transfer: f[0].parse().unwrap(),
            primaries: f[1].parse().unwrap(),
            target: f[2].parse().unwrap(),
            alpha: f[3].parse().unwrap(),
            width: f[4].parse().unwrap(),
            height: f[5].parse().unwrap(),
        };
        let (case, other, cpp_expected) = if f[0] == "pair" {
            (
                parse(&f[1..7]),
                parse(&f[7..13]),
                u64::from_str_radix(f[13], 16).unwrap(),
            )
        } else {
            (
                parse(&f[..6]),
                parse(&f[..6]),
                u64::from_str_radix(f[6], 16).unwrap(),
            )
        };
        let a = case.samples(false);
        let b = other.samples(true);
        let (a, b) = (case.image(&a, false), other.image(&b, true));
        let expected = compute_ssimulacra2_with_config(&a, &b, &Ssimulacra2Config::scalar())
            .unwrap()
            .to_bits();
        // Captures use x86-64 glibc. Other libm implementations can differ in
        // the final f64 pow, even with identical frontend and aggregate bits.
        // Every platform still requires exact scalar/SIMD/cached/map equality.
        if cfg!(all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )) {
            assert_eq!(expected, cpp_expected, "C++: {line}");
        }
        for cfg in [Ssimulacra2Config::scalar(), Ssimulacra2Config::simd()] {
            let score = compute_ssimulacra2_with_config(&a, &b, &cfg).unwrap();
            assert_eq!(score.to_bits(), expected, "{line}, {cfg:?}");
            let cache = Ssimulacra2Reference::new_with_config(&a, &cfg).unwrap();
            assert_eq!(
                cache.compare_with_config(&b, &cfg).unwrap().to_bits(),
                expected,
                "cached: {line}"
            );
            let maps = compute_ssimulacra2_maps_with_config(&a, &b, &cfg).unwrap();
            let cached = cache.compare_with_maps_with_config(&b, &cfg).unwrap();
            assert_eq!(maps.score.to_bits(), expected);
            assert_eq!(cached.score.to_bits(), expected);
            assert_eq!(maps.alpha_background, cached.alpha_background);
            assert_eq!(maps.scales.len(), cached.scales.len());
            for (a, b) in maps.scales.iter().zip(&cached.scales) {
                assert_eq!((a.width, a.height), (b.width, b.height));
                for (ap, bp) in a
                    .ssim
                    .iter()
                    .chain(&a.artifact)
                    .chain(&a.detail_lost)
                    .zip(b.ssim.iter().chain(&b.artifact).chain(&b.detail_lost))
                {
                    assert_eq!(
                        ap.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        bp.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn native_default_targets_and_strided_integer_layouts() {
    for transfer in [TransferFunction::Pq, TransferFunction::Hlg] {
        for (width, height) in [(1, 2), (17, 13)] {
            for alpha in [false, true] {
                for bgr in [false, true] {
                    let stride = width * 4 + 8;
                    let mut rgba = vec![0xffu8; stride * height];
                    let mut floats = Vec::new();
                    for y in 0..height {
                        for x in 0..width {
                            let rgb = [
                                ((x * 37 + y * 11) % 251) as u8,
                                ((x * 17 + y * 91) % 253) as u8,
                                ((x * 83 + y * 7) % 255) as u8,
                            ];
                            let a = if alpha {
                                ((x * 61 + y * 13) % 255) as u8
                            } else {
                                255
                            };
                            let (r, b) = if bgr {
                                (rgb[2], rgb[0])
                            } else {
                                (rgb[0], rgb[2])
                            };
                            rgba[y * stride + x * 4..y * stride + x * 4 + 4]
                                .copy_from_slice(&[r, rgb[1], b, a]);
                            floats.extend(rgb.map(|v| v as f32 * (1.0 / 255.0)));
                            floats.push(a as f32 * (1.0 / 255.0));
                        }
                    }
                    let desc = if bgr {
                        PixelDescriptor::BGRA8_SRGB
                    } else {
                        PixelDescriptor::RGBA8_SRGB
                    }
                    .with_transfer(transfer);
                    let a =
                        PixelSlice::new(&rgba, width as u32, height as u32, stride, desc).unwrap();
                    let b = PixelSlice::new(
                        bytemuck::cast_slice(&floats),
                        width as u32,
                        height as u32,
                        width * 16,
                        PixelDescriptor::RGBAF32_LINEAR.with_transfer(transfer),
                    )
                    .unwrap();
                    assert_eq!(compute_ssimulacra2(&a, &b).unwrap(), 100.0, "{desc:?}");
                    let cfg = Ssimulacra2Config::strips(8);
                    assert!(matches!(
                        compute_ssimulacra2_with_config(&a, &b, &cfg),
                        Err(Ssimulacra2Error::InvalidConfiguration(_))
                    ));
                    let cache = Ssimulacra2Reference::new(&a).unwrap();
                    assert!(matches!(
                        cache.compare_with_config(&b, &cfg),
                        Err(Ssimulacra2Error::InvalidConfiguration(_))
                    ));
                }
            }
        }
    }
}

#[test]
#[allow(deprecated)] // Native/PU21 score separation is an intentional compatibility check.
fn native_hdr_does_not_use_pu21_or_clamp_highlights() {
    let case = Case {
        transfer: 8,
        primaries: 9,
        target: 203.0,
        alpha: 0,
        width: 17,
        height: 13,
    };
    let a = case.samples(false);
    let mut b = a.clone();
    for px in b.as_chunks_mut::<4>().0.iter_mut().step_by(5) {
        for v in &mut px[..3] {
            *v = v.min(1.0);
        }
    }
    let (a, b) = (case.image(&a, false), case.image(&b, true));
    assert!(compute_ssimulacra2(&a, &b).unwrap() < 100.0);
    #[cfg(feature = "hdr-pu")]
    assert_ne!(
        compute_ssimulacra2(&a, &b).unwrap(),
        fast_ssim2::compute_ssimulacra2_pu(&a, &b).unwrap()
    );
    let cache = Ssimulacra2Reference::new(&a).unwrap();
    assert_eq!(
        cache.compare(&b).unwrap(),
        compute_ssimulacra2(&a, &b).unwrap()
    );
    assert!(matches!(
        cache.compare_with_config(
            &b,
            &Ssimulacra2Config::default().with_strip(StripConfig::new(8))
        ),
        Err(Ssimulacra2Error::InvalidConfiguration(_))
    ));
}

#[test]
fn native_hdr_validates_anchors_and_honors_cancellation() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct StopAt(AtomicUsize);
    impl enough::Stop for StopAt {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.0.fetch_sub(1, Ordering::Relaxed) == 0 {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }
    let case = Case {
        transfer: 18,
        primaries: 9,
        target: 1000.0,
        alpha: 3,
        width: 17,
        height: 129,
    };
    let pixels = case.samples(false);
    let image = case.image(&pixels, false);
    let cache = Ssimulacra2Reference::new(&image).unwrap();
    for polls in [0, 3, 5, 9] {
        let stop = StopAt(AtomicUsize::new(polls));
        let cfg = Ssimulacra2Config::default().with_stop(&stop);
        assert!(matches!(
            compute_ssimulacra2_with_config(&image, &image, &cfg),
            Err(Ssimulacra2Error::Cancelled(_))
        ));
        let stop = StopAt(AtomicUsize::new(polls));
        assert!(matches!(
            cache.compare_with_config(&image, &Ssimulacra2Config::default().with_stop(&stop)),
            Err(Ssimulacra2Error::Cancelled(_))
        ));
        let stop = StopAt(AtomicUsize::new(polls));
        assert!(matches!(
            Ssimulacra2Reference::new_with_config(
                &image,
                &Ssimulacra2Config::default().with_stop(&stop)
            ),
            Err(Ssimulacra2Error::Cancelled(_))
        ));
    }
    for target in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let invalid = case
            .image(&pixels, false)
            .with_diffuse_white(zenpixels::DiffuseWhite::new(target));
        assert!(matches!(
            compute_ssimulacra2(&invalid, &invalid),
            Err(Ssimulacra2Error::UnsupportedInput(_))
        ));
    }
}

#[test]
#[allow(deprecated)] // Validate alpha rejection through every retained entry point.
fn alpha_metadata_without_a_stored_channel_is_rejected() {
    use zenpixels::{AlphaMode, PixelFormat};
    for format in [
        PixelFormat::Rgb8,
        PixelFormat::Rgb16,
        PixelFormat::RgbF32,
        PixelFormat::Gray8,
        PixelFormat::Gray16,
        PixelFormat::GrayF32,
    ] {
        for transfer in [
            TransferFunction::Srgb,
            TransferFunction::Pq,
            TransferFunction::Hlg,
            TransferFunction::Linear,
        ] {
            for alpha in [AlphaMode::Straight, AlphaMode::Premultiplied] {
                let mut desc = PixelDescriptor::RGB8_SRGB
                    .with_transfer(transfer)
                    .with_alpha(Some(alpha));
                desc.format = format;
                let bpp = format.bytes_per_pixel();
                let data = vec![0u32; 64 * bpp / 4];
                let image =
                    PixelSlice::new(bytemuck::cast_slice(&data), 8, 8, 8 * bpp, desc).unwrap();
                assert!(
                    matches!(
                        compute_ssimulacra2(&image, &image),
                        Err(Ssimulacra2Error::UnsupportedInput(_))
                    ),
                    "{desc:?}"
                );
                assert!(matches!(
                    Ssimulacra2Reference::new(&image),
                    Err(Ssimulacra2Error::UnsupportedInput(_))
                ));
                #[cfg(feature = "hdr-pu")]
                if transfer != TransferFunction::Srgb {
                    assert!(matches!(
                        fast_ssim2::compute_ssimulacra2_pu(&image, &image),
                        Err(Ssimulacra2Error::UnsupportedInput(_))
                    ));
                }
            }
        }
    }
}
