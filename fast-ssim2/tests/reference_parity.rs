//! Tests that verify ssimulacra2 scores against C++ reference values.
//!
//! These tests use pre-captured C++ ssimulacra2 scores for synthetic test images,
//! allowing parity verification without requiring the C++ binary at runtime.
//!
//! To regenerate reference data:
//!   SSIMULACRA2_BIN=/path/to/ssimulacra2 cargo run --example capture_cpp_reference
//!
//! Run tests with: cargo test --test reference_parity

use fast_ssim2::compute_ssimulacra2;
use fast_ssim2::reference_data::{REFERENCE_CASES, ReferenceCase};
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

// ============================================================================
// Image generation
// ============================================================================
//
// The generators live in `dev/testcases.rs`, shared verbatim with
// `examples/capture_cpp_reference.rs` (which produced the table this file
// checks) and `examples/parity_report.rs`. All three used to carry their own
// copy, kept in step by a comment reading "must match capture_cpp_reference.rs
// exactly" — one silent edit away from a parity gate that compared different
// images than the ones the C++ binary scored. The SHA256 assertions below are
// the backstop; sharing the source removes the need for them to fire.

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/dev/testcases.rs"));

/// Look up a compiled-in reference case's images by name.
///
/// Panics rather than skipping if the table and the generator disagree about
/// which cases exist: a parity gate that silently tests fewer cases than it
/// claims is worse than one that fails.
fn generate_test_image(case: &ReferenceCase) -> (Vec<u8>, Vec<u8>) {
    let cases = generate_test_cases();
    let found = cases
        .into_iter()
        .find(|c| c.name == case.name)
        .unwrap_or_else(|| {
            panic!(
                "reference case {:?} has no generator in dev/testcases.rs — the \
                 compiled-in table and the generator have diverged",
                case.name
            )
        });
    (found.source_data, found.distorted_data)
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_reference_parity() {
    // The case table is compiled into src/reference_data.rs (66 cases; the
    // file header records the capture date). An empty table would make this
    // parity gate silently test
    // nothing — fail loudly instead of skipping. Regenerate the table with:
    // SSIMULACRA2_BIN=/path/to/ssimulacra2 cargo run --example capture_cpp_reference
    assert!(
        !REFERENCE_CASES.is_empty(),
        "no reference cases compiled in — the C++ parity gate would be vacuous"
    );

    let mut failures = Vec::new();
    let mut max_error = 0.0f64;

    for (i, case) in REFERENCE_CASES.iter().enumerate() {
        let (source_data, distorted_data) = generate_test_image(case);

        // Verify hashes match (detects changes in image generation)
        let source_hash = sha256_hex(&source_data);
        let distorted_hash = sha256_hex(&distorted_data);

        if source_hash != case.source_hash {
            eprintln!(
                "\nERROR: Source image hash mismatch for {}!\nExpected: {}\nGot:      {}\nThis indicates the image generation algorithm changed.",
                case.name, case.source_hash, source_hash
            );
            panic!("Image generation changed for {}", case.name);
        }

        if distorted_hash != case.distorted_hash {
            eprintln!(
                "\nERROR: Distorted image hash mismatch for {}!\nExpected: {}\nGot:      {}\nThis indicates the image generation algorithm changed.",
                case.name, case.distorted_hash, distorted_hash
            );
            panic!("Image generation changed for {}", case.name);
        }

        // Convert to RGB format
        let source_rgb: Vec<[f32; 3]> = source_data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect();

        let distorted_rgb: Vec<[f32; 3]> = distorted_data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect();

        let nz_width = std::num::NonZeroUsize::new(case.width).unwrap();
        let nz_height = std::num::NonZeroUsize::new(case.height).unwrap();
        let source = Rgb::new(
            source_rgb,
            nz_width,
            nz_height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let distorted = Rgb::new(
            distorted_rgb,
            nz_width,
            nz_height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let score = compute_ssimulacra2(source, distorted).unwrap();
        let error = (score - case.expected_score).abs();
        max_error = max_error.max(error);

        // Per-pattern tolerance. Measured 2026-08-31 against
        // /opt/homebrew/bin/ssimulacra2 (jpeg-xl 0.12.0) on aarch64; see
        // benchmarks/cpp_parity_2026-08-31.md and src/cpp_parity_diag.rs.
        let tolerance = if case.name.contains("uniform_shift") {
            // NOT "different FP rounding in SIMD paths" — that was the earlier
            // reading of this line and it is wrong twice over. The two
            // fast-ssim2 backends agree here to ~1e-8; the disagreement is
            // with the C++ binary, and it is one-directional: on 18 of these
            // 20 cases fast-ssim2 scores *higher*, mean +0.408.
            //
            // The cause is that a flat field makes SSIM' numerically
            // degenerate. `sigma11 - mu1^2`, `sigma22 - mu2^2` and
            // `sigma12 - mu1*mu2` are all analytically zero away from the
            // border, so `num_s` and `denom_s` both collapse onto
            // `kC2 = 9e-4` plus whatever the f32 Gaussian left behind — a
            // residual of ~5e-7, i.e. ~1/1600 of the term it perturbs. On
            // `uniform_shift_1` the true per-pixel error is 1.11e-5 and the
            // rounding contributes 3.31e-5, three times more; over half the
            // pixels have |d - d_exact| > d_exact, and at 256x256 the centre
            // pixel's `d` clamps to exactly 0. `d = max(..., 0)` then
            // rectifies the residual so it cannot cancel, and the
            // implementation with the noisier blur reports the lower score.
            //
            // Measured amplification is ~1e6: swapping in jpegli's own
            // 4-unrolled horizontal Gaussian moves these scores by up to 1.2
            // and cuts the mean error to 0.258. The C++ values are themselves
            // not monotonic in the shift (shift 1 -> 97.75 but shift 5 ->
            // 98.81 at 32x32), which is what a noise-dominated measurement
            // looks like. This family therefore cannot discriminate
            // implementations at all, and the tolerance is wide because the
            // *reference* is uninformative here, not because our SIMD is
            // sloppy. Real photographic content, where the metric is well
            // conditioned, agrees with the same binary to a mean of 0.024
            // (examples/photo_parity.rs).
            10.0
        } else if case.name.contains("boxblur8x8")
            || case.name.contains("sharpen")
            || case.name.contains("yuv_roundtrip")
        {
            0.2 // Allow some variance for distortion patterns
        } else if case.name.contains("_vs_")
            || case.name.starts_with("perfect_match")
            || case.name.starts_with("gradient_h_")
            || case.name.starts_with("gradient_v_")
            || case.name.starts_with("checkerboard_")
            || case.name.starts_with("noise_seed_")
            || case.name.starts_with("edge_")
        {
            0.01 // Synthetic patterns and non-identical comparisons should be close
        } else {
            0.05 // Fallback for any other patterns
        };

        if error > tolerance {
            failures.push((i, case.name, case.expected_score, score, error));
        }
    }

    if !failures.is_empty() {
        eprintln!(
            "\n{} / {} tests FAILED:",
            failures.len(),
            REFERENCE_CASES.len()
        );
        eprintln!(
            "{:<5} {:<50} {:>15} {:>15} {:>10}",
            "Index", "Name", "Expected", "Actual", "Error"
        );
        eprintln!("{:-<100}", "");
        for (i, name, expected, actual, error) in &failures {
            eprintln!(
                "{:<5} {:<50} {:>15.6} {:>15.6} {:>10.6}",
                i, name, expected, actual, error
            );
        }
        eprintln!("\nMax error: {:.6}", max_error);
        panic!("{} tests failed", failures.len());
    }

    // Show error distribution
    #[derive(Debug, Clone)]
    struct ErrorCase {
        name: &'static str,
        expected: f64,
        actual: f64,
        error: f64,
    }

    let all_errors: Vec<ErrorCase> = REFERENCE_CASES
        .iter()
        .map(|case| {
            let (source_data, distorted_data) = generate_test_image(case);
            let source_rgb: Vec<[f32; 3]> = source_data
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ]
                })
                .collect();
            let distorted_rgb: Vec<[f32; 3]> = distorted_data
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ]
                })
                .collect();
            let nz_width = std::num::NonZeroUsize::new(case.width).unwrap();
            let nz_height = std::num::NonZeroUsize::new(case.height).unwrap();
            let source = Rgb::new(
                source_rgb,
                nz_width,
                nz_height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap();
            let distorted = Rgb::new(
                distorted_rgb,
                nz_width,
                nz_height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap();
            let score = compute_ssimulacra2(source, distorted).unwrap();
            ErrorCase {
                name: case.name,
                expected: case.expected_score,
                actual: score,
                error: (score - case.expected_score).abs(),
            }
        })
        .collect();

    // Sort by error descending for reporting
    let mut sorted_errors = all_errors.clone();
    sorted_errors.sort_by(|a, b| b.error.partial_cmp(&a.error).unwrap());

    println!("\n{:=^100}", " REFERENCE PARITY TEST RESULTS ");
    println!(
        "All {} reference tests passed! Max error: {:.6}",
        REFERENCE_CASES.len(),
        max_error
    );

    // Error percentiles
    let mut error_values: Vec<f64> = all_errors.iter().map(|e| e.error).collect();
    error_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\nError percentiles: p50={:.4}, p90={:.4}, p95={:.4}, p99={:.4}",
        error_values[error_values.len() / 2],
        error_values[(error_values.len() * 90) / 100],
        error_values[(error_values.len() * 95) / 100],
        error_values[(error_values.len() * 99) / 100]
    );
    println!(
        "Errors >0.1: {}, >0.5: {}, >1.0: {}",
        error_values.iter().filter(|&&e| e > 0.1).count(),
        error_values.iter().filter(|&&e| e > 0.5).count(),
        error_values.iter().filter(|&&e| e > 1.0).count()
    );

    // Top 10 errors
    println!("\n{:-^100}", " Top 10 Largest Errors ");
    println!(
        "{:<50} {:>15} {:>15} {:>10}",
        "Test Case", "Expected", "Actual", "Error"
    );
    println!("{:-<100}", "");
    for case in sorted_errors.iter().take(10) {
        println!(
            "{:<50} {:>15.6} {:>15.6} {:>10.6}",
            case.name, case.expected, case.actual, case.error
        );
    }

    // Error breakdown by pattern type
    println!("\n{:-^100}", " Error Breakdown by Pattern Type ");

    let mut pattern_errors: std::collections::HashMap<&str, Vec<f64>> =
        std::collections::HashMap::new();
    for case in &all_errors {
        let pattern = if case.name.contains("uniform_shift") {
            "uniform_shift"
        } else if case.name.contains("boxblur8x8")
            || case.name.contains("sharpen")
            || case.name.contains("yuv_roundtrip")
        {
            "distortions"
        } else if case.name.contains("_vs_") {
            "synthetic_vs"
        } else if case.name.starts_with("perfect_match") {
            "perfect_match"
        } else if case.name.starts_with("gradient") {
            "gradients"
        } else if case.name.starts_with("checkerboard") {
            "checkerboard"
        } else if case.name.starts_with("noise_seed") {
            "noise"
        } else if case.name.starts_with("edge") {
            "edges"
        } else {
            "other"
        };
        pattern_errors.entry(pattern).or_default().push(case.error);
    }

    println!(
        "{:<20} {:>10} {:>15} {:>15} {:>15}",
        "Pattern", "Count", "Max Error", "Mean Error", "P95 Error"
    );
    println!("{:-<80}", "");
    let mut pattern_names: Vec<_> = pattern_errors.keys().copied().collect();
    pattern_names.sort();
    for pattern in pattern_names {
        if let Some(errors) = pattern_errors.get_mut(pattern) {
            errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let max = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = errors.iter().sum::<f64>() / errors.len() as f64;
            let p95 = errors[(errors.len() * 95) / 100];
            println!(
                "{:<20} {:>10} {:>15.6} {:>15.6} {:>15.6}",
                pattern,
                errors.len(),
                max,
                mean,
                p95
            );
        }
    }
    println!("{:=^100}", "");
}
