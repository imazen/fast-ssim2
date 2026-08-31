//! SIMD tier consistency tests for fast-ssim2.
//!
//! Runs SSIMULACRA2 under every archmage SIMD tier permutation available on the
//! host, and under both [`SimdImpl`] backends, and requires them all to agree.
//!
//! ## Why the tolerance is what it is
//!
//! SSIMULACRA2 is unusually sensitive to rounding. Its SSIM' term divides by
//! `kC2 = 9e-4`, while the f32 recursive Gaussian leaves a residual of ~5e-7 in
//! `sigma11 - mu1^2` on smooth content — where that difference is analytically
//! zero. `d = max(1 - num_m * num_s / denom_s, 0)` then *rectifies* the residual,
//! so it cannot cancel: it accumulates as a one-directional error. Measured
//! amplification on a flat field is ~1e6, i.e. a 1e-7 difference anywhere
//! upstream can move the final score by 0.1.
//!
//! That makes this test a genuinely sharp instrument, and it caught a real bug:
//! until 0.8.3 `SimdImpl::Scalar` ran `yuvxyb`'s f64 cube root while
//! `SimdImpl::Simd` ran two f32 Halley steps, so the two backends computed
//! *different metrics*. On 576 real photographic pairs they disagreed by up to
//! 0.879 on the 0..100 scale — above the 0.5 tolerance this file used to carry,
//! though the synthetic images here never provoked it. With both backends on
//! the same arithmetic the same corpus measures a maximum disagreement of
//! 2.4e-7. See `benchmarks/cpp_parity_2026-08-31.md`.
//!
//! x86 (AVX2/AVX-512) and wasm128 are **NOT MEASURED** here — this host is
//! aarch64, and the tier sweep emulates missing tokens rather than missing
//! hardware. Run `cargo test --test simd_consistency -- --nocapture` on those
//! targets, read the printed spreads, and tighten if they hold.

#![forbid(unsafe_code)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use fast_ssim2::{
    LinearRgbImage, SimdImpl, Ssimulacra2Config, ToLinearRgb, compute_ssimulacra2_with_config,
};

/// Maximum permitted end-to-end spread between two (tier, backend)
/// combinations **whose blur output differs**.
///
/// aarch64 / Apple M4 Pro, 2026-08-31, 50 combinations per image:
///
/// | combinations                          | measured max spread |
/// |---------------------------------------|---------------------|
/// | blur bit-identical (`KERNEL_TOLERANCE`)| 2.2e-8              |
/// | blur differs, textured content         | 8.3e-2              |
/// | blur differs, smooth content           | 4.97e-1             |
///
/// The second class is one specific, understood thing. The recursive Gaussian's
/// `MUL_PREV` step is a fused multiply-add, matching the C++ reference, and
/// `magetypes` lowers `mul_add` to a real FMA only on NEON/AVX2/AVX-512 — on
/// wasm128 (no FMA in the wasm SIMD MVP) and on its scalar polyfill it emits
/// `a * b + c`. SSIMULACRA2 then amplifies that ~1-ulp-per-step difference by
/// ~1e6 on smooth content, which is where 4.97e-1 comes from.
///
/// Unfusing our side was measured and rejected: it makes every target agree
/// bit-for-bit, at the cost of agreeing with the C++ binary 2.8x less well
/// (mean |delta| over 576 real photographic pairs 0.024 -> 0.067, max
/// 0.52 -> 1.04, and the unfused form acquires a systematic -0.058 bias). The
/// fix belongs in `magetypes`, whose own `f32x1::mul_add` already uses `fmaf`;
/// when it lands, delete this constant and use `KERNEL_TOLERANCE` throughout.
///
/// This bound is deliberately loose because it gates an *amplified* effect.
/// The cause is gated 200x more tightly, at the point where it is born, by
/// [`blur_tier_divergence_is_bounded`] (1e-5 per blurred sample, 2.5e-6
/// measured). Everything that is not the blur is gated at bit-identity by
/// [`dispatched_kernels_are_bit_identical_across_tiers`].
const TIER_TOLERANCE: f64 = 6e-1;

/// What any two combinations with a bit-identical blur are held to. This is
/// the bound that matters: it covers every FMA-capable target (so every x86-64
/// and every aarch64 machine), and both `SimdImpl` backends. It is 5000x
/// tighter than the single 0.5 tolerance this file carried until 0.8.3, which
/// was loose enough to hide `SimdImpl::Scalar` and `SimdImpl::Simd` computing
/// different metrics — they disagreed by up to 0.879 on real photographs.
const KERNEL_TOLERANCE: f64 = 1e-4;

/// Generate a deterministic test image of varied linear RGB pixels.
fn generate_test_image(width: usize, height: usize) -> LinearRgbImage {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 7 + y * 13) % 256) as f32 / 255.0;
            let g = ((x * 11 + y * 3 + 50) % 256) as f32 / 255.0;
            let b = ((x * 5 + y * 17 + 100) % 256) as f32 / 255.0;
            data.push([r, g, b]);
        }
    }
    LinearRgbImage::new(data, width, height)
}

/// Generate a slightly different image (simulated distortion).
fn generate_distorted_image(width: usize, height: usize) -> LinearRgbImage {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (((x * 7 + y * 13) % 256) as f32 / 255.0 + 0.02).min(1.0);
            let g = (((x * 11 + y * 3 + 50) % 256) as f32 / 255.0 - 0.01).max(0.0);
            let b = (((x * 5 + y * 17 + 100) % 256) as f32 / 255.0 + 0.005).min(1.0);
            data.push([r, g, b]);
        }
    }
    LinearRgbImage::new(data, width, height)
}

/// A near-flat pair with a small constant offset — the regime where the SSIM'
/// denominator collapses onto `kC2` and rounding is amplified hardest. A tier
/// test that only uses busy images cannot see the divergence class that
/// actually bites.
fn generate_smooth_pair(width: usize, height: usize) -> (LinearRgbImage, LinearRgbImage) {
    let mut a = Vec::with_capacity(width * height);
    let mut b = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            // Very gentle ramp: 0.2 .. 0.22 across the whole image.
            let v = 0.2 + 0.02 * (x + y) as f32 / (width + height) as f32;
            a.push([v, v, v]);
            b.push([v + 0.002, v + 0.002, v + 0.002]);
        }
    }
    (
        LinearRgbImage::new(a, width, height),
        LinearRgbImage::new(b, width, height),
    )
}

/// Fingerprint of the dispatched blur under the currently-enabled tokens.
///
/// Used to partition tier permutations into blur-equivalence classes without
/// parsing token labels: permutations whose blur agrees bit-for-bit must
/// produce scores that agree to [`KERNEL_TOLERANCE`], and only permutations
/// whose blur differs — i.e. the non-FMA `magetypes` arm — get the wider
/// [`TIER_TOLERANCE`].
fn blur_fingerprint() -> u64 {
    use fast_ssim2::Blur;
    const W: usize = 40;
    const H: usize = 40;
    let plane: Vec<f32> = (0..W * H)
        .map(|i| 0.3 + 0.4 * ((i * 7) % 251) as f32 / 251.0)
        .collect();
    let out = Blur::new(W, H).blur(&[plane.clone(), plane.clone(), plane]);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in out.iter().flat_map(|p| p.iter()) {
        h ^= u64::from(v.to_bits());
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

struct Combo {
    label: String,
    score: f64,
    blur: u64,
}

/// Score `(source, distorted)` under every tier permutation and both backends.
fn score_everywhere(source: &LinearRgbImage, distorted: &LinearRgbImage) -> Vec<Combo> {
    let mut out = Vec::new();
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let blur = blur_fingerprint();
        for (name, cfg) in [
            ("simd", Ssimulacra2Config::new(SimdImpl::Simd)),
            ("scalar", Ssimulacra2Config::new(SimdImpl::Scalar)),
        ] {
            let s = source.try_to_linear_rgb().unwrap();
            let d = distorted.try_to_linear_rgb().unwrap();
            let score = compute_ssimulacra2_with_config(s, d, cfg).expect("score");
            out.push(Combo {
                label: format!("{}/{name}", perm.label),
                score,
                blur,
            });
        }
    });
    out
}

fn assert_agreement(case: &str, combos: &[Combo]) -> (f64, f64) {
    assert!(
        combos.len() >= 2,
        "{case}: only {} combination(s) ran — the tier sweep is vacuous",
        combos.len()
    );

    let mut worst_same = (0.0f64, String::new());
    let mut worst_any = (0.0f64, String::new());
    let base = &combos[0];
    for c in &combos[1..] {
        let diff = (c.score - base.score).abs();
        if diff > worst_any.0 {
            worst_any = (diff, c.label.clone());
        }
        if c.blur == base.blur && diff > worst_same.0 {
            worst_same = (diff, c.label.clone());
        }
    }
    let classes = {
        let mut v: Vec<u64> = combos.iter().map(|c| c.blur).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    println!(
        "{case:<20} combos={:<3} blur-classes={classes}  same-blur spread {:.3e}  \
         cross-blur spread {:.3e}",
        combos.len(),
        worst_same.0,
        worst_any.0
    );

    assert!(
        worst_same.0 < KERNEL_TOLERANCE,
        "{case}: two tiers with a bit-identical blur still disagree by {:e} \
         (bound {KERNEL_TOLERANCE:e}); worst is '{}' vs '{}'. That cannot be the \
         known magetypes FMA gap — something in the XYB, SSIM' or edge-diff \
         kernels is tier-dependent.",
        worst_same.0,
        worst_same.1,
        base.label
    );
    assert!(
        worst_any.0 < TIER_TOLERANCE,
        "{case}: tiers disagree by {:e} (bound {TIER_TOLERANCE:e}); worst is \
         '{}' vs '{}'. Expected cause is magetypes not fusing `mul_add` on \
         wasm128 / scalar; a larger value means the gap has grown.",
        worst_any.0,
        worst_any.1,
        base.label
    );
    (worst_same.0, worst_any.0)
}

#[test]
fn ssimulacra2_all_tiers_within_tolerance() {
    println!("\nmeasured tier-to-tier spread (all archmage tiers x both backends):");
    let (mut same, mut any) = (0.0f64, 0.0f64);

    for size in [32usize, 64, 128] {
        let source = generate_test_image(size, size);
        let distorted = generate_distorted_image(size, size);
        let (s, a) = assert_agreement(
            &format!("textured_{size}x{size}"),
            &score_everywhere(&source, &distorted),
        );
        same = same.max(s);
        any = any.max(a);

        let (p, q) = generate_smooth_pair(size, size);
        let (s, a) = assert_agreement(&format!("smooth_{size}x{size}"), &score_everywhere(&p, &q));
        same = same.max(s);
        any = any.max(a);
    }

    println!(
        "overall: same-blur {same:.3e} (bound {KERNEL_TOLERANCE:e}), \
         cross-blur {any:.3e} (bound {TIER_TOLERANCE:e})"
    );
}

/// The sharp gate: every `#[magetypes]`-dispatched kernel except the blur must
/// produce **bit-identical** output on every tier the host can emulate.
///
/// This is what the old blanket 0.5 end-to-end tolerance was hiding. Until
/// 0.8.3 the opsin matrix and the SSIM' `num_m` term were FMA chains, and
/// `magetypes` does not fuse `mul_add` on its scalar polyfill or on wasm128,
/// so those two kernels returned different values on any target without
/// AVX2/NEON — 1.79e-7 in the XYB output, which SSIMULACRA2 amplifies to 0.085
/// on the 0..100 scale.
#[test]
fn dispatched_kernels_are_bit_identical_across_tiers() {
    use fast_ssim2::__bench_kernels as k;

    const N: usize = 1024;
    let rgb: Vec<[f32; 3]> = (0..N)
        .map(|i| {
            let f = i as f32 / N as f32;
            [0.10 + 0.80 * f, 0.20 + 0.70 * (1.0 - f), 0.30 + 0.60 * f]
        })
        .collect();
    let plane = |off: f32| -> [Vec<f32>; 3] {
        let mut p = [vec![0f32; N], vec![0f32; N], vec![0f32; N]];
        for (i, f) in (0..N).map(|i| i as f32 / N as f32).enumerate() {
            p[0][i] = 0.42 + 0.01 * f + off;
            p[1][i] = 0.30 + 0.20 * f + off;
            p[2][i] = 0.55 + 0.05 * (1.0 - f) + off;
        }
        p
    };
    let a = plane(0.0);
    let b = plane(0.001);
    // Second moments shaped like the blur's output: close to a*a, not equal.
    let scaled = |p: &[Vec<f32>; 3], q: &[Vec<f32>; 3]| -> [Vec<f32>; 3] {
        let mut o = p.clone();
        for c in 0..3 {
            for i in 0..N {
                o[c][i] = p[c][i] * q[c][i] * 1.000_001;
            }
        }
        o
    };
    let (aa, bb, ab) = (scaled(&a, &a), scaled(&b, &b), scaled(&a, &b));

    /// XYB pixels, SSIM' plane averages, edge-diff plane averages, product planes.
    type KernelOutputs = (Vec<[f32; 3]>, [f64; 6], [f64; 12], [Vec<f32>; 3]);
    let mut baseline: Option<KernelOutputs> = None;
    let mut checked = 0usize;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut xyb = rgb.clone();
        k::linear_rgb_to_xyb_simd(&mut xyb);
        let ssim = k::ssim_map_simd(6, 0, N, 1, &a, &b, &aa, &bb, &ab);
        let edge = k::edge_diff_map_simd(6, 0, N, 1, &a, &aa, &b, &bb);
        let mut mulout = [vec![0f32; N], vec![0f32; N], vec![0f32; N]];
        k::image_multiply_simd(&a, &b, &mut mulout);

        match &baseline {
            None => baseline = Some((xyb, ssim, edge, mulout)),
            Some((x0, s0, e0, m0)) => {
                checked += 1;
                for (i, (p, q)) in xyb.iter().zip(x0.iter()).enumerate() {
                    for (c, (a, b)) in p.iter().zip(q.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "linear_rgb_to_xyb_simd differs under '{}' at pixel {i} channel {c}: \
                             {a} vs {b}",
                            perm.label,
                        );
                    }
                }
                assert_eq!(ssim, *s0, "ssim_map_simd differs under '{}'", perm.label);
                assert_eq!(
                    edge, *e0,
                    "edge_diff_map_simd differs under '{}'",
                    perm.label
                );
                assert_eq!(
                    mulout, *m0,
                    "image_multiply_simd differs under '{}'",
                    perm.label
                );
            }
        }
    });

    assert!(
        checked > 0,
        "no alternative tier was exercised — is the archmage `testable_dispatch` \
         feature enabled? The gate would be vacuous."
    );
    println!("{checked} alternative tier permutations, all bit-identical");
}

/// The blur is the one kernel that is *not* bit-identical across tiers, and
/// this pins how far apart it is allowed to get. See [`TIER_TOLERANCE`].
#[test]
fn blur_tier_divergence_is_bounded() {
    use fast_ssim2::Blur;

    /// Max permitted per-sample difference in a blurred plane between the
    /// FMA and non-FMA arms. aarch64 2026-08-31 measured value is printed by
    /// the test; the bound is set from it with room for a different lane width.
    const BLUR_TOLERANCE: f32 = 1e-5;

    const W: usize = 96;
    const H: usize = 96;
    let plane: Vec<f32> = (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            0.3 + 0.4 * ((x * 7 + y * 13) % 251) as f32 / 251.0
        })
        .collect();
    let img = [plane.clone(), plane.clone(), plane];

    let mut baseline: Option<[Vec<f32>; 3]> = None;
    let mut worst = 0.0f32;
    let mut worst_label = String::new();
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let out = Blur::new(W, H).blur(&img);
        match &baseline {
            None => baseline = Some(out),
            Some(b0) => {
                let d = out
                    .iter()
                    .zip(b0.iter())
                    .flat_map(|(p, q)| p.iter().zip(q.iter()).map(|(a, b)| (a - b).abs()))
                    .fold(0.0f32, f32::max);
                if d > worst {
                    worst = d;
                    worst_label = perm.label.to_string();
                }
            }
        }
    });
    println!(
        "blur max per-sample tier divergence = {worst:.3e} (worst: {})",
        if worst_label.is_empty() {
            "none".into()
        } else {
            worst_label.clone()
        }
    );
    assert!(
        worst < BLUR_TOLERANCE,
        "blur diverges by {worst:e} across tiers under '{worst_label}' \
         (bound {BLUR_TOLERANCE:e}); the known cause is magetypes not fusing \
         `mul_add` on wasm128 / scalar, so a larger value means something new"
    );
}

#[test]
fn ssimulacra2_roundtrip_stability() {
    let source = generate_test_image(32, 32);
    let distorted = generate_distorted_image(32, 32);

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let s1 = source.try_to_linear_rgb().unwrap();
        let d1 = distorted.try_to_linear_rgb().unwrap();
        let score1 =
            compute_ssimulacra2_with_config(s1, d1, Ssimulacra2Config::simd()).expect("score1");

        let s2 = source.try_to_linear_rgb().unwrap();
        let d2 = distorted.try_to_linear_rgb().unwrap();
        let score2 =
            compute_ssimulacra2_with_config(s2, d2, Ssimulacra2Config::simd()).expect("score2");

        assert_eq!(
            score1.to_bits(),
            score2.to_bits(),
            "ssimulacra2 not deterministic under '{}': {score1} vs {score2}",
            perm.label,
        );
    });
}
