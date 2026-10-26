//! In-memory score parity against the independent C++ native-HDR oracle.
use fast_ssim2::{
    Ssimulacra2Config, Ssimulacra2Reference, compute_ssimulacra2_maps_with_config,
    compute_ssimulacra2_with_config,
};
use std::{
    fmt::Write,
    io::Write as IoWrite,
    path::PathBuf,
    process::{Command, Stdio},
};
#[path = "../tests/common/native_hdr.rs"]
mod native_hdr;
use native_hdr::Case;

fn oracle_score(oracle: &PathBuf, case: &Case, a: &[f32], b: &[f32], raw: bool) -> f64 {
    oracle_score_pair(oracle, case, case, a, b, raw)
}

fn oracle_score_pair(
    oracle: &PathBuf,
    case: &Case,
    other: &Case,
    a: &[f32],
    b: &[f32],
    raw: bool,
) -> f64 {
    let backgrounds: &[f32] = if case.alpha & 1 == 0 && other.alpha & 2 == 0 {
        &[0.5]
    } else {
        &[0.1, 0.9]
    };
    let mut best = f64::INFINITY;
    for &bg in backgrounds {
        let mut request = String::from(if raw { "score-raw\n" } else { "score\n" });
        for (data, distorted, case) in [(a, false, case), (b, true, other)] {
            let alpha = if distorted {
                case.alpha & 2 != 0
            } else {
                case.alpha & 1 != 0
            };
            writeln!(
                request,
                "{} {} {} {bg} {} {} {}",
                case.transfer,
                case.primaries,
                case.target,
                u8::from(alpha),
                case.width,
                case.height
            )
            .unwrap();
            for px in data.as_chunks::<4>().0 {
                writeln!(
                    request,
                    "{:08x} {:08x} {:08x} {:08x}",
                    px[0].to_bits(),
                    px[1].to_bits(),
                    px[2].to_bits(),
                    px[3].to_bits()
                )
                .unwrap();
            }
        }
        let mut child = Command::new(oracle)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("start C++ oracle");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(request.as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "C++ oracle failed");
        let text = String::from_utf8(out.stdout).unwrap();
        if raw {
            println!("background {bg}\n{text}");
        }
        let bits = u64::from_str_radix(text.split_whitespace().next().unwrap(), 16).unwrap();
        let score = f64::from_bits(bits);
        assert!(score.is_finite(), "C++ nonfinite score: {text}");
        best = best.min(score);
    }
    best
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let oracle = PathBuf::from(
        args.next()
            .expect("usage: native_hdr_parity ORACLE [NEW_CAPTURE_PATH]"),
    );
    let capture = args.next().map(PathBuf::from);
    if capture == Some(PathBuf::from("--raw-case")) {
        let case = Case {
            transfer: 16,
            primaries: 11,
            target: 255.0,
            alpha: 1,
            width: 65,
            height: 63,
        };
        oracle_score(
            &oracle,
            &case,
            &case.samples(false),
            &case.samples(true),
            true,
        );
        return;
    }
    #[cfg(feature = "unstable-internals")]
    if capture == Some(PathBuf::from("--transform")) {
        let bits = args.next().unwrap().into_string().unwrap();
        let sum = f64::from_bits(u64::from_str_radix(&bits, 16).unwrap());
        let score = fast_ssim2::pipeline::score::transform(sum);
        println!("{:016x} {score:.17}", score.to_bits());
        return;
    }
    if let Some(path) = &capture {
        assert!(!path.exists(), "capture already exists: {}", path.display());
    }
    use sha2::{Digest, Sha256};
    let identity = Sha256::digest(std::fs::read(&oracle).expect("read oracle binary"))
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect::<String>();
    let mut rows = format!(
        "# Cloudinary 2.1 81feacf6a5b0bbd7ec11b4a881a4615033ab0b10; generator tests/common/native_hdr.rs\n# oracle_sha256 {identity}\n# tf primaries target alpha width height score_f64_bits; pair rows hold both descriptors\n"
    );
    let mut count = 0;
    for (tf, targets) in [
        (8, [255.0, 203.0]),
        (16, [10000.0, 255.0]),
        (18, [1000.0, 300.0]),
    ] {
        for primaries in [1, 9, 11] {
            for target in targets {
                for alpha in 0..4 {
                    for (width, height) in [(8, 8), (17, 13), (65, 63), (513, 257)] {
                        if width == 513 && target != targets[0] {
                            continue;
                        }
                        let case = Case {
                            transfer: tf,
                            primaries,
                            target,
                            alpha,
                            width,
                            height,
                        };
                        let a = case.samples(false);
                        let b = case.samples(true);
                        let expected = oracle_score(&oracle, &case, &a, &b, false);
                        for cfg in [Ssimulacra2Config::scalar(), Ssimulacra2Config::simd()] {
                            let (src, dst) = (case.image(&a, false), case.image(&b, true));
                            let score = compute_ssimulacra2_with_config(&src, &dst, &cfg).unwrap();
                            assert_eq!(
                                score.to_bits(),
                                expected.to_bits(),
                                "tf={tf}, p={primaries}, target={target}, alpha={alpha}, {width}x{height}, cfg={cfg:?}: Rust {score}, C++ {expected}"
                            );
                            let cache = Ssimulacra2Reference::new_with_config(&src, &cfg).unwrap();
                            assert_eq!(
                                cache.compare_with_config(&dst, &cfg).unwrap().to_bits(),
                                score.to_bits()
                            );
                            let maps =
                                compute_ssimulacra2_maps_with_config(&src, &dst, &cfg).unwrap();
                            assert_eq!(maps.score.to_bits(), score.to_bits());
                            let cached = cache.compare_with_maps_with_config(&dst, &cfg).unwrap();
                            assert_eq!(maps.scales.len(), cached.scales.len());
                            assert_eq!(cached.score.to_bits(), score.to_bits());
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
                            assert_eq!(maps.alpha_background, cached.alpha_background);
                        }
                        writeln!(
                            rows,
                            "{tf} {primaries} {target} {alpha} {width} {height} {:016x}",
                            expected.to_bits()
                        )
                        .unwrap();
                        count += 1;
                    }
                }
            }
        }
        println!(
            "native HDR tf={tf}: verified {count} cumulative cases against C++ (both kernels, cached, maps)"
        );
    }
    for ((at, ap, an), (bt, bp, bn)) in [
        ((16, 9, 10000.0), (18, 11, 1000.0)),
        ((18, 1, 1000.0), (8, 9, 203.0)),
        ((8, 11, 255.0), (16, 1, 255.0)),
    ] {
        for alpha in 0..4 {
            let a_case = Case {
                transfer: at,
                primaries: ap,
                target: an,
                alpha,
                width: 17,
                height: 13,
            };
            let b_case = Case {
                transfer: bt,
                primaries: bp,
                target: bn,
                alpha,
                width: 17,
                height: 13,
            };
            let a = a_case.samples(false);
            let b = b_case.samples(true);
            let (src, dst) = (a_case.image(&a, false), b_case.image(&b, true));
            let expected = oracle_score_pair(&oracle, &a_case, &b_case, &a, &b, false);
            for cfg in [Ssimulacra2Config::scalar(), Ssimulacra2Config::simd()] {
                let score = compute_ssimulacra2_with_config(&src, &dst, &cfg).unwrap();
                assert_eq!(
                    score.to_bits(),
                    expected.to_bits(),
                    "mixed {at}/{bt}, alpha={alpha}"
                );
                let cache = Ssimulacra2Reference::new_with_config(&src, &cfg).unwrap();
                assert_eq!(
                    cache.compare_with_config(&dst, &cfg).unwrap().to_bits(),
                    expected.to_bits()
                );
                assert_eq!(
                    cache
                        .compare_with_maps_with_config(&dst, &cfg)
                        .unwrap()
                        .score
                        .to_bits(),
                    expected.to_bits()
                );
                assert_eq!(
                    compute_ssimulacra2_maps_with_config(&src, &dst, &cfg)
                        .unwrap()
                        .score
                        .to_bits(),
                    expected.to_bits()
                );
            }
            writeln!(
                rows,
                "pair {at} {ap} {an} {alpha} 17 13 {bt} {bp} {bn} {alpha} 17 13 {:016x}",
                expected.to_bits()
            )
            .unwrap();
            count += 1;
        }
    }
    if let Some(path) = capture {
        assert!(rows.len() <= 30000);
        std::fs::write(path, rows).unwrap();
    }
    println!("All {count} native HDR cases are bit-exact against C++.");
}
