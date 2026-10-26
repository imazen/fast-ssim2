//! Interleaved single-thread comparisons on identical RGB8 input buffers.
//! Pixel generation and decoding are outside the timed region.
//! Small borrowed image-wrapper construction is included for each version.
#![forbid(unsafe_code)]
use std::{sync::Arc, time::Duration};

mod native_hdr;

#[path = "../../../fast-ssim2/src/reference_data.rs"]
mod reference_data;
#[path = "../../../fast-ssim2/dev/testcases.rs"]
mod testcases;

fn parity() {
    let cases = testcases::generate_test_cases();
    let mut sums = [0.0f64; 3];
    let mut maxima = [0.0f64; 3];
    let mut n = 0;
    for expected in reference_data::REFERENCE_CASES {
        let case = cases.iter().find(|c| c.name == expected.name).unwrap();
        assert_eq!(case.source_hash, expected.source_hash);
        assert_eq!(case.distorted_hash, expected.distorted_hash);
        assert_eq!((case.width, case.height), (expected.width, expected.height));
        if case.source_hash == case.distorted_hash {
            continue;
        }
        n += 1;
        for version in 0..3 {
            let actual = score(
                case.source_data.as_chunks::<3>().0,
                case.distorted_data.as_chunks::<3>().0,
                case.width,
                case.height,
                version,
            );
            let delta = actual - expected.expected_score;
            sums[version] += delta.abs();
            maxima[version] = maxima[version].max(delta.abs());
            println!(
                "parity\t{}\t{}\t{:.12}\t{:.12}\t{:.12}",
                case.name,
                ["0.7.1", "0.8.2", "current"][version],
                expected.expected_score,
                actual,
                delta
            );
        }
    }
    for version in 0..3 {
        println!(
            "summary\t{}\tnonidentity={n}\tmean_abs={:.12}\tmax_abs={:.12}",
            ["0.7.1", "0.8.2", "current"][version],
            sums[version] / n as f64,
            maxima[version]
        );
    }
}

fn main() {
    if std::env::var_os("AUDIT_NATIVE_HDR").is_some() {
        native_hdr::run();
        return;
    }
    if std::env::var_os("AUDIT_PARITY").is_some() {
        parity();
        return;
    }
    let rounds: usize = std::env::var("AUDIT_ROUNDS")
        .unwrap_or_else(|_| "7".into())
        .parse()
        .unwrap();
    assert!(rounds > 0);
    let result = zenbench::run_gated(zenbench::GateConfig::disabled(), |suite| {
        if let Ok(group) = std::env::var("AUDIT_GROUP") {
            suite.set_group_filter(group);
        }
        for (w, h) in [(320, 240), (1920, 1080), (3840, 2160)] {
            let mut state = 0x0bad_c0de_dead_beefu64;
            let src: Vec<[u8; 3]> = (0..w * h)
                .map(|i| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let noise = ((state >> 33) & 255) as u8;
                    [
                        (i % w * 255 / (w - 1)) as u8,
                        (i / w * 255 / (h - 1)) as u8,
                        noise,
                    ]
                })
                .collect();
            let dst: Vec<[u8; 3]> = src
                .iter()
                .map(|p| [p[0].saturating_add(3), p[1], p[2].saturating_sub(2)])
                .collect();
            let src = Arc::new(src);
            let dst = Arc::new(dst);
            for (version, label) in ["0.7.1", "0.8.2", "current"].into_iter().enumerate() {
                println!(
                    "score\t{w}\t{h}\t{label}\t{:.12}",
                    score(&src, &dst, w, h, version)
                );
            }
            let (one_src, one_dst) = (Arc::clone(&src), Arc::clone(&dst));
            suite.compare(format!("rgb8_{w}x{h}"), move |g| {
                let (src, dst) = (one_src, one_dst);
                g.baseline("0.8.2");
                g.config()
                    .max_rounds(rounds)
                    .min_rounds(rounds)
                    .warmup_time(Duration::from_millis(200));
                g.config().auto_rounds = false;
                g.config().max_time = Duration::from_secs(90);
                g.config().max_wall_time = Duration::from_secs(120);
                for (version, label) in ["0.7.1", "0.8.2", "current"].into_iter().enumerate() {
                    let (src, dst) = (Arc::clone(&src), Arc::clone(&dst));
                    g.bench(label, move |b| b.iter(|| score(&src, &dst, w, h, version)));
                }
            });
            if std::env::var_os("AUDIT_CACHE").is_some() && w == 1920 {
                suite.compare("cached_1920x1080", move |g| {
                    g.baseline("0.8.2_compare_with");
                    g.config().max_rounds(rounds).min_rounds(rounds);
                    g.config().auto_rounds = false;
                    g.config().max_time = Duration::from_secs(90);
                    for reuse in [false, true] {
                        let reference =
                            v082::Ssimulacra2Reference::new(imgref::Img::new(src.as_slice(), w, h))
                                .unwrap();
                        let mut ctx = reference.compare_context();
                        let dst = Arc::clone(&dst);
                        let label = if reuse {
                            "0.8.2_compare_with"
                        } else {
                            "0.8.2_compare"
                        };
                        g.bench(label, move |b| {
                            b.iter(|| {
                                let d = imgref::Img::new(dst.as_slice(), w, h);
                                if reuse {
                                    reference.compare_with(&mut ctx, d).unwrap()
                                } else {
                                    reference.compare(d).unwrap()
                                }
                            })
                        });
                    }
                    let a = current::PixelSlice::new(
                        bytemuck::cast_slice(src.as_slice()),
                        w as u32,
                        h as u32,
                        w * 3,
                        current::PixelDescriptor::RGB8_SRGB,
                    )
                    .unwrap();
                    let cfg = current::Ssimulacra2Config::simd();
                    let reference =
                        current::Ssimulacra2Reference::new_with_config(&a, &cfg).unwrap();
                    g.bench("current_compare", move |b| {
                        b.iter(|| {
                            let d = current::PixelSlice::new(
                                bytemuck::cast_slice(dst.as_slice()),
                                w as u32,
                                h as u32,
                                w * 3,
                                current::PixelDescriptor::RGB8_SRGB,
                            )
                            .unwrap();
                            reference.compare_with_config(&d, &cfg).unwrap()
                        })
                    });
                });
            }
        }
    });
    zenbench::postprocess_result(&result);
}

fn score(src: &[[u8; 3]], dst: &[[u8; 3]], w: usize, h: usize, version: usize) -> f64 {
    match version {
        0 => v071::compute_ssimulacra2(imgref::Img::new(src, w, h), imgref::Img::new(dst, w, h))
            .unwrap(),
        1 => v082::compute_ssimulacra2(imgref::Img::new(src, w, h), imgref::Img::new(dst, w, h))
            .unwrap(),
        2 => {
            let a = current::PixelSlice::new(
                bytemuck::cast_slice(src),
                w as u32,
                h as u32,
                w * 3,
                current::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            let b = current::PixelSlice::new(
                bytemuck::cast_slice(dst),
                w as u32,
                h as u32,
                w * 3,
                current::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            current::compute_ssimulacra2_with_config(&a, &b, &current::Ssimulacra2Config::simd())
                .unwrap()
        }
        _ => unreachable!(),
    }
}
