//! Test-only diagnostics that pin down *where* a score difference is born.
//!
//! `tests/reference_parity.rs` and `tests/simd_consistency.rs` can only see the
//! final scalar score, which is a weighted sum of 108 sub-scores squashed
//! through a cubic and a power curve — a useless signal for localising a
//! divergence. These helpers re-run the pipeline with individually swappable
//! stages and report the per-scale, per-channel, per-norm sub-scores.
//!
//! Run with:
//!   cargo test --release --lib cpp_parity_diag -- --nocapture

#![cfg(test)]

use yuvxyb::{ColorPrimaries, LinearRgb, Rgb, TransferCharacteristic, Xyb};

use crate::{
    Blur, MsssimScale, SimdImpl, downscale_by_2, edge_diff_map, image_multiply, make_positive_xyb,
    ssim_map, weights, xyb_to_planar_into,
};

/// Which cube-root approximation feeds the opsin nonlinearity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CbrtKind {
    /// What `SimdImpl::Simd` ships: bit-hack seed + 2 Halley steps in f32.
    OursSimd,
    /// What `SimdImpl::Scalar` ships: `yuvxyb`, bit-hack seed + 2 Newton steps in f64.
    OursScalar,
    /// Verbatim transliteration of jpegli `CubeRootAndAdd` (`lib/base/fast_math-inl.h`),
    /// documented there as "cbrt(x) + add with 6 ulp max error". This is what
    /// the C++ SSIMULACRA2 binary actually evaluates.
    CppFastMath,
}

/// jpegli `CubeRootAndAdd`, scalar transliteration. Newton-Raphson on the
/// *inverse* cube root in f32, three plain steps plus a corrected one.
pub fn cpp_cube_root_and_add(x: f32, add: f32) -> f32 {
    const K_EXP_BIAS: i32 = 0x5480_0000; // cast(1.) + cast(1.) / 3
    const K_EXP_MUL: i32 = 0x002A_AAAA; // shifted 1/3
    const K1_3: f32 = 1.0f32 / 3.0;
    const K4_3: f32 = 4.0f32 / 3.0;

    let xa = x; // "assume inputs never negative"
    let xa_3 = K1_3 * xa;

    let m1 = xa.to_bits() as i32;
    // Special-case 0 exactly as the C++ does (IfThenZeroElse on m1 == 0).
    let m2 = if m1 == 0 {
        0
    } else {
        K_EXP_BIAS - ((m1 >> 23) * K_EXP_MUL)
    };
    let mut r = f32::from_bits(m2 as u32);

    for _ in 0..3 {
        let r2 = r * r;
        // NegMulAdd(a, b, c) = c - a * b
        r = K4_3.mul_add(r, -(xa_3 * (r2 * r2)));
    }
    let mut r2 = r * r;
    r = K1_3.mul_add(r.mul_add(1.0, -(xa * (r2 * r2))), r);
    r2 = r * r;
    r2.mul_add(x, add)
}

/// Our SIMD path's cube root, scalar transliteration (`xyb_simd.rs`).
fn ours_simd_cbrt(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let hx = (ui & 0x7FFF_FFFF) / 3 + B1;
    let mut t = f32::from_bits((ui & 0x8000_0000) | hx);
    for _ in 0..2 {
        let r = t * t * t;
        t *= x.mul_add(2.0, r) / (x + r.mul_add(2.0, 0.0));
    }
    t
}

/// Same as [`ours_simd_cbrt`] but with a configurable iteration count, to find
/// how many f32 Halley steps are needed to reach the f64 path's accuracy.
fn halley_cbrt_f32(x: f32, iters: usize) -> f32 {
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let hx = (ui & 0x7FFF_FFFF) / 3 + B1;
    let mut t = f32::from_bits((ui & 0x8000_0000) | hx);
    for _ in 0..iters {
        let r = t * t * t;
        t *= x.mul_add(2.0, r) / (x + r.mul_add(2.0, 0.0));
    }
    t
}

/// Our scalar path's cube root (`xyb_simd::cbrtf_fast`), which `yuvxyb` mirrors.
fn ours_scalar_cbrt(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7FFF_FFFF;
    hx = hx / 3 + B1;
    ui &= 0x8000_0000;
    ui |= hx;
    let mut t: f64 = f64::from(f32::from_bits(ui));
    let xf64 = f64::from(x);
    let mut r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

/// Linear RGB -> positive XYB with a selectable cube root, otherwise following
/// `xyb_simd::convert_pixel_scalar` + `make_positive_xyb` exactly.
fn to_positive_xyb(linear: &LinearRgb, kind: CbrtKind) -> Xyb {
    use crate::xyb_simd::{K_B0, K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22};
    let m = [
        K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
    ];
    let cbrt = |v: f32| match kind {
        CbrtKind::OursSimd => ours_simd_cbrt(v),
        CbrtKind::OursScalar => ours_scalar_cbrt(v),
        CbrtKind::CppFastMath => cpp_cube_root_and_add(v, 0.0),
    };
    // The C++ folds `-cbrtf(bias)` (an exact libm cbrtf) into the add operand;
    // our paths add `-cbrtf_fast(bias)`. Use each path's own convention.
    let neg_bias_cbrt = match kind {
        CbrtKind::CppFastMath => -(K_B0 as f64).cbrt() as f32,
        _ => -ours_scalar_cbrt(K_B0),
    };

    let mut data: Vec<[f32; 3]> = linear.data().to_vec();
    for pix in data.iter_mut() {
        let (r, g, b) = (pix[0], pix[1], pix[2]);
        let mut mixed = [
            m[0].mul_add(r, m[1].mul_add(g, m[2].mul_add(b, K_B0))),
            m[3].mul_add(r, m[4].mul_add(g, m[5].mul_add(b, K_B0))),
            m[6].mul_add(r, m[7].mul_add(g, m[8].mul_add(b, K_B0))),
        ];
        for v in mixed.iter_mut() {
            *v = v.max(0.0);
            *v = cbrt(*v) + neg_bias_cbrt;
        }
        pix[0] = 0.5 * (mixed[0] - mixed[1]);
        pix[1] = 0.5 * (mixed[0] + mixed[1]);
        pix[2] = mixed[2];
    }
    let mut xyb =
        Xyb::new(data, linear.width(), linear.height()).expect("XYB construction should not fail");
    make_positive_xyb(&mut xyb);
    xyb
}

// ---------------------------------------------------------------------------
// Blur: ours (sequential recurrence) vs jpegli's (4-step closed form)
// ---------------------------------------------------------------------------

mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

/// Which horizontal Gaussian pass to use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HorizKind {
    /// What fast-ssim2 ships: one recurrence step per output sample.
    Sequential,
    /// What jpegli's `FastGaussian1D` evaluates on any non-scalar Highway
    /// target (so: every machine anyone runs the C++ binary on). Four outputs
    /// per iteration from f32-rounded 2nd/3rd/4th-power coefficients.
    CppUnrolled4,
    /// The sequential recurrence carried in f64 — the numerical ground truth
    /// both of the above are approximating.
    F64Reference,
}

fn horizontal_row(input: &[f32], output: &mut [f32], width: usize, kind: HorizKind) {
    let big_n = consts::RADIUS as isize;
    let at = |i: isize| -> f32 {
        if i >= 0 && (i as usize) < width {
            input[i as usize]
        } else {
            0.0
        }
    };

    if kind == HorizKind::F64Reference {
        let (mut p1, mut p3, mut p5) = (0f64, 0f64, 0f64);
        let (mut q1, mut q3, mut q5) = (0f64, 0f64, 0f64);
        let mut n = -big_n + 1;
        while n < width as isize {
            let sum = f64::from(at(n - big_n - 1)) + f64::from(at(n + big_n - 1));
            let o1 = sum * f64::from(consts::MUL_IN_1)
                + f64::from(consts::MUL_PREV2_1) * q1
                + f64::from(consts::MUL_PREV_1) * p1;
            let o3 = sum * f64::from(consts::MUL_IN_3)
                + f64::from(consts::MUL_PREV2_3) * q3
                + f64::from(consts::MUL_PREV_3) * p3;
            let o5 = sum * f64::from(consts::MUL_IN_5)
                + f64::from(consts::MUL_PREV2_5) * q5
                + f64::from(consts::MUL_PREV_5) * p5;
            (q1, q3, q5) = (p1, p3, p5);
            (p1, p3, p5) = (o1, o3, o5);
            if n >= 0 {
                output[n as usize] = (o1 + o3 + o5) as f32;
            }
            n += 1;
        }
        return;
    }

    // Both f32 variants share the scalar prologue/epilogue; they differ only
    // in the interior, exactly as jpegli's FastGaussian1D does.
    let (mut p1, mut p3, mut p5) = (0f32, 0f32, 0f32);
    let (mut q1, mut q3, mut q5) = (0f32, 0f32, 0f32);
    let step = |sum: f32, p: &mut (f32, f32, f32), q: &mut (f32, f32, f32)| -> f32 {
        let mut o1 = sum * consts::MUL_IN_1;
        let mut o3 = sum * consts::MUL_IN_3;
        let mut o5 = sum * consts::MUL_IN_5;
        o1 = consts::MUL_PREV2_1.mul_add(q.0, o1);
        o3 = consts::MUL_PREV2_3.mul_add(q.1, o3);
        o5 = consts::MUL_PREV2_5.mul_add(q.2, o5);
        *q = *p;
        o1 = consts::MUL_PREV_1.mul_add(p.0, o1);
        o3 = consts::MUL_PREV_3.mul_add(p.1, o3);
        o5 = consts::MUL_PREV_5.mul_add(p.2, o5);
        *p = (o1, o3, o5);
        o1 + o3 + o5
    };

    let mut n = -big_n + 1;
    if kind == HorizKind::Sequential {
        while n < width as isize {
            let sum = at(n - big_n - 1) + at(n + big_n - 1);
            let mut pv = (p1, p3, p5);
            let mut qv = (q1, q3, q5);
            let o = step(sum, &mut pv, &mut qv);
            (p1, p3, p5) = pv;
            (q1, q3, q5) = qv;
            if n >= 0 {
                output[n as usize] = o;
            }
            n += 1;
        }
        return;
    }

    // jpegli: scalar until `first_aligned = RoundUpTo(N + 1, 4)`.
    const LANES: isize = 4;
    let first_aligned = ((big_n + 1) + LANES - 1) / LANES * LANES;
    while n < first_aligned.min(width as isize) {
        let sum = at(n - big_n - 1) + at(n + big_n - 1);
        let mut pv = (p1, p3, p5);
        let mut qv = (q1, q3, q5);
        let o = step(sum, &mut pv, &mut qv);
        (p1, p3, p5) = pv;
        (q1, q3, q5) = qv;
        if n >= 0 {
            output[n as usize] = o;
        }
        n += 1;
    }

    // Unrolled interior: four outputs at once, from the powered coefficients.
    // out_j = sum_{i<=j} mul_in[j-i]*sum_i + mul_prev[j]*prev + mul_prev2[j]*prev2
    while n < width as isize - big_n + 1 - (LANES - 1) {
        let s: [f32; 4] = std::array::from_fn(|k| {
            let m = n + k as isize;
            at(m - big_n - 1) + at(m + big_n - 1)
        });
        let mut o = [[0f32; 4]; 3]; // [k-index (1,3,5)][lane]
        for (band, base) in [(0usize, 0usize), (1, 4), (2, 8)] {
            let (prev, prev2) = match band {
                0 => (p1, q1),
                1 => (p3, q3),
                _ => (p5, q5),
            };
            for (lane, slot) in o[band].iter_mut().enumerate() {
                // Broadcast-and-shift: lane `lane` accumulates
                // mul_in[lane - i] * s[i] for i <= lane.
                let mut acc = 0f32;
                for (i, si) in s.iter().enumerate().take(lane + 1) {
                    acc = consts::CPP_MUL_IN[base + (lane - i)].mul_add(*si, acc);
                }
                acc = consts::CPP_MUL_PREV2[base + lane].mul_add(prev2, acc);
                acc = consts::CPP_MUL_PREV[base + lane].mul_add(prev, acc);
                *slot = acc;
            }
        }
        // prev2 = lane 2, prev = lane 3 (Broadcast<LANES-2> / <LANES-1>).
        q1 = o[0][2];
        q3 = o[1][2];
        q5 = o[2][2];
        p1 = o[0][3];
        p3 = o[1][3];
        p5 = o[2][3];
        for lane in 0..4usize {
            output[(n + lane as isize) as usize] = o[0][lane] + o[1][lane] + o[2][lane];
        }
        n += LANES;
    }

    while n < width as isize {
        let sum = at(n - big_n - 1) + at(n + big_n - 1);
        let mut pv = (p1, p3, p5);
        let mut qv = (q1, q3, q5);
        let o = step(sum, &mut pv, &mut qv);
        (p1, p3, p5) = pv;
        (q1, q3, q5) = qv;
        output[n as usize] = o;
        n += 1;
    }
}

/// Vertical pass, identical in fast-ssim2 and jpegli (two FMAs per sample,
/// one column per SIMD lane, so vectorisation cannot change the result).
fn vertical_pass(input: &[f32], output: &mut [f32], width: usize, height: usize) {
    let big_n = consts::RADIUS as isize;
    for x in 0..width {
        let (mut p1, mut p3, mut p5) = (0f32, 0f32, 0f32);
        let (mut q1, mut q3, mut q5) = (0f32, 0f32, 0f32);
        let mut n = -big_n + 1;
        while n < height as isize {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;
            let t = if top >= 0 {
                input[top as usize * width + x]
            } else {
                0.0
            };
            let b = if bottom < height as isize {
                input[bottom as usize * width + x]
            } else {
                0.0
            };
            let sum = t + b;
            let o1 = sum.mul_add(
                consts::VERT_MUL_IN_1,
                -p1.mul_add(consts::VERT_MUL_PREV_1, q1),
            );
            let o3 = sum.mul_add(
                consts::VERT_MUL_IN_3,
                -p3.mul_add(consts::VERT_MUL_PREV_3, q3),
            );
            let o5 = sum.mul_add(
                consts::VERT_MUL_IN_5,
                -p5.mul_add(consts::VERT_MUL_PREV_5, q5),
            );
            (q1, q3, q5) = (p1, p3, p5);
            (p1, p3, p5) = (o1, o3, o5);
            if n >= 0 {
                output[n as usize * width + x] = o1 + o3 + o5;
            }
            n += 1;
        }
    }
}

fn blur_plane(input: &[f32], width: usize, height: usize, kind: HorizKind) -> Vec<f32> {
    let mut tmp = vec![0f32; width * height];
    let mut out = vec![0f32; width * height];
    for y in 0..height {
        horizontal_row(
            &input[y * width..][..width],
            &mut tmp[y * width..][..width],
            width,
            kind,
        );
    }
    vertical_pass(&tmp, &mut out, width, height);
    out
}

/// Knobs for one diagnostic run.
#[derive(Clone, Copy, Debug)]
pub struct DiagConfig {
    pub cbrt: CbrtKind,
    /// Backend for multiply / ssim-map / edge-map.
    pub kernels: SimdImpl,
    /// `None` uses the shipped [`Blur`]; `Some` uses the self-contained
    /// transliterations above, so the C++ horizontal pass can be swapped in.
    pub horiz: Option<HorizKind>,
}

/// Re-implementation of `compute_frame_flavored` that also hands back the
/// per-scale sub-scores. Kept deliberately close to the original so a
/// divergence here means the original changed.
pub fn run(img1: LinearRgb, img2: LinearRgb, cfg: DiagConfig) -> (f64, Vec<MsssimScale>) {
    let mut img1 = img1;
    let mut img2 = img2;
    let mut width = img1.width().get();
    let mut height = img1.height().get();
    let impl_type = cfg.kernels;
    let scales_n = weights::count_scales(width, height);

    let alloc_plane = || vec![0.0f32; width * height];
    let alloc_3planes = || [alloc_plane(), alloc_plane(), alloc_plane()];
    let mut mul = alloc_3planes();
    let mut sigma1_sq = alloc_3planes();
    let mut sigma2_sq = alloc_3planes();
    let mut sigma12 = alloc_3planes();
    let mut mu1 = alloc_3planes();
    let mut mu2 = alloc_3planes();
    let mut img1_planar = alloc_3planes();
    let mut img2_planar = alloc_3planes();

    let mut blur = Blur::with_simd_impl(width, height, impl_type);
    let mut scales = Vec::new();

    for scale in 0..crate::NUM_SCALES {
        if width < 8 || height < 8 {
            break;
        }
        if scale > 0 {
            img1 = downscale_by_2(&img1);
            img2 = downscale_by_2(&img2);
            width = img1.width().get();
            height = img1.height().get();
        }
        let size = width * height;
        for buf in [
            &mut mul,
            &mut sigma1_sq,
            &mut sigma2_sq,
            &mut sigma12,
            &mut mu1,
            &mut mu2,
            &mut img1_planar,
            &mut img2_planar,
        ] {
            for c in buf.iter_mut() {
                c.truncate(size);
            }
        }
        blur.shrink_to(width, height);

        let a = to_positive_xyb(&img1, cfg.cbrt);
        let b = to_positive_xyb(&img2, cfg.cbrt);
        xyb_to_planar_into(&a, &mut img1_planar);
        xyb_to_planar_into(&b, &mut img2_planar);

        let do_blur =
            |src: &[Vec<f32>; 3], dst: &mut [Vec<f32>; 3], blur: &mut Blur| match cfg.horiz {
                None => blur.blur_into(src, dst),
                Some(kind) => {
                    for c in 0..3 {
                        dst[c].copy_from_slice(&blur_plane(&src[c], width, height, kind));
                    }
                }
            };

        image_multiply(&img1_planar, &img1_planar, &mut mul, impl_type);
        do_blur(&mul, &mut sigma1_sq, &mut blur);
        image_multiply(&img2_planar, &img2_planar, &mut mul, impl_type);
        do_blur(&mul, &mut sigma2_sq, &mut blur);
        image_multiply(&img1_planar, &img2_planar, &mut mul, impl_type);
        do_blur(&mul, &mut sigma12, &mut blur);
        do_blur(&img1_planar, &mut mu1, &mut blur);
        do_blur(&img2_planar, &mut mu2, &mut blur);

        scales.push(MsssimScale {
            avg_ssim: ssim_map(
                scales_n, scale, width, height, &mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12,
                impl_type,
            ),
            avg_edgediff: edge_diff_map(
                scales_n,
                scale,
                width,
                height,
                &img1_planar,
                &mu1,
                &img2_planar,
                &mu2,
                impl_type,
            ),
        });
    }

    let msssim = crate::Msssim {
        scales: scales.clone(),
    };
    (msssim.score(), scales)
}

pub fn uniform_pair(size: usize, shift: u8) -> (LinearRgb, LinearRgb) {
    let mk = |v: u8| {
        let px = vec![[v as f32 / 255.0; 3]; size * size];
        let rgb = Rgb::new(
            px,
            std::num::NonZeroUsize::new(size).unwrap(),
            std::num::NonZeroUsize::new(size).unwrap(),
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        LinearRgb::try_from(rgb).unwrap()
    };
    (mk(128), mk(128 + shift))
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[test]
fn diag_cbrt_accuracy() {
    // Absolute error of each cube-root approximation against an f64 reference,
    // sampled over the opsin-mixed range that 8-bit sRGB actually produces.
    println!("\ncbrt error vs f64 reference over x in [0.001, 1.0]:");
    let mut worst = [(0.0f64, 0.0f32); 3];
    for i in 1..=100_000u32 {
        let x = i as f32 / 100_000.0;
        let exact = (x as f64).cbrt();
        for (k, v) in [
            ours_simd_cbrt(x),
            ours_scalar_cbrt(x),
            cpp_cube_root_and_add(x, 0.0),
        ]
        .into_iter()
        .enumerate()
        {
            let e = (v as f64 - exact).abs();
            if e > worst[k].0 {
                worst[k] = (e, x);
            }
        }
    }
    for (name, (e, at)) in ["ours_simd", "ours_scalar", "cpp_fastmath"]
        .iter()
        .zip(worst.iter())
    {
        println!("  {name:<14} max abs err {e:.3e}  at x = {at}");
    }

    // The opsin-mixed value can never be below the absorbance bias, so the
    // domain that actually matters is [kB0, ~1.004]. Sweep every f32 in it.
    println!("\nsame, over the domain the opsin stage actually produces:");
    let lo = crate::xyb_simd::K_B0;
    let hi = 1.004f32;
    let mut w = [(0.0f64, 0.0f32); 5];
    let mut x = lo;
    let mut n = 0u64;
    while x <= hi {
        let exact = (x as f64).cbrt();
        let cands = [
            halley_cbrt_f32(x, 2),
            halley_cbrt_f32(x, 3),
            halley_cbrt_f32(x, 4),
            ours_scalar_cbrt(x),
            cpp_cube_root_and_add(x, 0.0),
        ];
        for (k, v) in cands.into_iter().enumerate() {
            let e = (v as f64 - exact).abs();
            if e > w[k].0 {
                w[k] = (e, x);
            }
        }
        x = f32::from_bits(x.to_bits() + 1);
        n += 1;
    }
    println!("  ({n} f32 values sampled exhaustively)");
    for (name, (e, at)) in [
        "halley f32 x2 (ours simd)",
        "halley f32 x3",
        "halley f32 x4",
        "newton f64 x2 (ours scalar)",
        "jpegli CubeRootAndAdd",
    ]
    .iter()
    .zip(w.iter())
    {
        println!(
            "  {name:<30} max abs err {e:.3e} = {:.2} ulp  at x = {at}",
            e / f64::from(f32::EPSILON * (*at).cbrt())
        );
    }
}

#[test]
fn diag_uniform_shift_by_stage() {
    // The uniform_shift reference cases are the only place fast-ssim2 is far
    // from the C++ binary. Print the score under every (cbrt, kernels)
    // combination so the responsible stage is visible.
    println!("\nuniform_shift scores by stage:");
    println!(
        "{:>6} {:>6} {:>16} {:>16} {:>16} {:>16}",
        "size", "shift", "simd+ourscbrt", "scalar+ourscbrt", "scalar+cppcbrt", "simd+cppcbrt"
    );
    for size in [32usize, 64, 128, 256] {
        for shift in [1u8, 5, 10, 20, 50] {
            let (a, b) = uniform_pair(size, shift);
            let s1 = run(
                a.clone(),
                b.clone(),
                DiagConfig {
                    cbrt: CbrtKind::OursSimd,
                    kernels: SimdImpl::Simd,
                    horiz: None,
                },
            )
            .0;
            let s2 = run(
                a.clone(),
                b.clone(),
                DiagConfig {
                    cbrt: CbrtKind::OursScalar,
                    kernels: SimdImpl::Scalar,
                    horiz: None,
                },
            )
            .0;
            let s3 = run(
                a.clone(),
                b.clone(),
                DiagConfig {
                    cbrt: CbrtKind::CppFastMath,
                    kernels: SimdImpl::Scalar,
                    horiz: None,
                },
            )
            .0;
            let s4 = run(
                a,
                b,
                DiagConfig {
                    cbrt: CbrtKind::CppFastMath,
                    kernels: SimdImpl::Simd,
                    horiz: None,
                },
            )
            .0;
            println!("{size:>6} {shift:>6} {s1:>16.8} {s2:>16.8} {s3:>16.8} {s4:>16.8}");
        }
    }
}

/// The SSIM' term the metric actually evaluates, pulled apart for one plane.
///
/// On a flat field `sigma11 - mu1^2`, `sigma22 - mu2^2` and `sigma12 - mu1*mu2`
/// are analytically zero in the interior, so `num_s` and `denom_s` both
/// collapse onto `kC2 = 9e-4` plus rounding. Print the actual magnitudes so
/// "the flat case is conditioned on rounding" stops being an assertion.
#[test]
fn diag_flat_field_conditioning() {
    use crate::xyb_to_planar;

    for (size, shift) in [(32usize, 1u8), (32, 50), (256, 1)] {
        let (l1, l2) = uniform_pair(size, shift);
        let a = to_positive_xyb(&l1, CbrtKind::OursSimd);
        let b = to_positive_xyb(&l2, CbrtKind::OursSimd);
        let p1 = xyb_to_planar(&a);
        let p2 = xyb_to_planar(&b);

        let mut blur = Blur::with_simd_impl(size, size, SimdImpl::Simd);
        let mut mul = [
            vec![0.0f32; size * size],
            vec![0.0; size * size],
            vec![0.0; size * size],
        ];
        let mut s11 = mul.clone();
        let mut s22 = mul.clone();
        let mut s12 = mul.clone();
        let mut mu1 = mul.clone();
        let mut mu2 = mul.clone();
        image_multiply(&p1, &p1, &mut mul, SimdImpl::Simd);
        blur.blur_into(&mul, &mut s11);
        image_multiply(&p2, &p2, &mut mul, SimdImpl::Simd);
        blur.blur_into(&mul, &mut s22);
        image_multiply(&p1, &p2, &mut mul, SimdImpl::Simd);
        blur.blur_into(&mul, &mut s12);
        blur.blur_into(&p1, &mut mu1);
        blur.blur_into(&p2, &mut mu2);

        // Channel 1 (Y) carries almost all the weight for grey shifts.
        let c = 1usize;
        let centre = (size / 2) * size + size / 2;
        let (m1, m2) = (mu1[c][centre], mu2[c][centre]);
        let var1 = s11[c][centre] - m1 * m1;
        let var2 = s22[c][centre] - m2 * m2;
        let cov = s12[c][centre] - m1 * m2;
        let signal = (p1[c][centre] - p2[c][centre]) as f64;
        println!("\nuniform_shift_{shift}_{size}x{size}, Y plane, image centre:");
        println!("  XYB value             a = {:.9}", p1[c][centre]);
        println!(
            "  XYB shift signal    a-b = {signal:.3e}   (d_exact = (a-b)^2 = {:.3e})",
            signal * signal
        );
        println!("  sigma11 - mu1^2         = {var1:.3e}   (analytically 0)");
        println!("  sigma22 - mu2^2         = {var2:.3e}   (analytically 0)");
        println!("  sigma12 - mu1*mu2       = {cov:.3e}   (analytically 0)");
        println!("  kC2                     = {:.3e}", 0.0009f32);
        let num_m = 1.0f32 - (m1 - m2) * (m1 - m2);
        let num_s = 2.0f32 * cov + 0.0009;
        let den_s = var1 + var2 + 0.0009;
        let d = (1.0f32 - num_m * num_s / den_s).max(0.0);
        println!(
            "  d = max(1 - num_m*num_s/denom_s, 0) = {d:.3e}, of which rounding = {:+.3e}",
            d as f64 - signal * signal
        );
        // How far off is `d` from the analytic answer, as a ratio?
        println!(
            "  rounding / signal       = {:.1}x",
            (d as f64 - signal * signal).abs() / (signal * signal)
        );

        // Scan the whole plane: how many pixels are dominated by rounding?
        let mut worse = 0usize;
        for i in 0..size * size {
            let (m1, m2) = (mu1[c][i], mu2[c][i]);
            let num_m = 1.0f32 - (m1 - m2) * (m1 - m2);
            let num_s = 2.0f32 * (s12[c][i] - m1 * m2) + 0.0009;
            let den_s = (s11[c][i] - m1 * m1) + (s22[c][i] - m2 * m2) + 0.0009;
            let d = (1.0f32 - num_m * num_s / den_s).max(0.0) as f64;
            if (d - signal * signal).abs() > signal * signal {
                worse += 1;
            }
        }
        println!(
            "  pixels where |d - d_exact| > d_exact: {worse}/{} ({:.0}%)",
            size * size,
            100.0 * worse as f64 / (size * size) as f64
        );
    }
}

/// Which horizontal Gaussian leaves the larger residual on a constant field.
///
/// A normalised blur of a constant `c` must return `c` in the interior. What
/// it actually returns differs by rounding, and *that* residual is what the
/// SSIM' term divides by `kC2` on a flat field. Measured against the same
/// recurrence carried in f64.
#[test]
fn diag_blur_flat_field_residual() {
    println!("\nblur(constant) residual vs the same recurrence in f64:");
    println!(
        "{:>6} {:>10} {:>16} {:>16}",
        "width", "const", "sequential (ours)", "cpp 4-unrolled"
    );
    for width in [32usize, 64, 128, 256] {
        for c in [0.457_412_3f32, 0.55, 0.9] {
            let input = vec![c; width * width];
            let r64 = blur_plane(&input, width, width, HorizKind::F64Reference);
            let ours = blur_plane(&input, width, width, HorizKind::Sequential);
            let cpp = blur_plane(&input, width, width, HorizKind::CppUnrolled4);
            // Interior only: the borders are zero-padded by design, so their
            // deviation from `c` is the filter, not rounding.
            let n = consts::RADIUS * 2;
            let mut e_ours = 0f64;
            let mut e_cpp = 0f64;
            for y in n..width - n {
                for x in n..width - n {
                    let i = y * width + x;
                    e_ours = e_ours.max((f64::from(ours[i]) - f64::from(r64[i])).abs());
                    e_cpp = e_cpp.max((f64::from(cpp[i]) - f64::from(r64[i])).abs());
                }
            }
            println!("{width:>6} {c:>10.6} {e_ours:>16.3e} {e_cpp:>16.3e}");
        }
    }
}

/// The decisive test. If the `uniform_shift` gap is fast-ssim2 being *more*
/// numerically accurate than the C++ binary rather than wrong, then swapping
/// jpegli's own 4-unrolled horizontal pass into our pipeline must move our
/// scores onto the C++ binary's.
///
/// C++ binary values (jpeg-xl 0.12.0, /opt/homebrew/bin/ssimulacra2, aarch64)
/// are inlined so the comparison is visible without re-running the binary.
#[test]
fn diag_uniform_shift_with_cpp_blur() {
    // (size, shift, C++ binary score)
    const CPP: &[(usize, u8, f64)] = &[
        (32, 1, 97.749_214_30),
        (32, 5, 98.808_273_82),
        (32, 10, 96.531_140_90),
        (32, 20, 93.798_232_20),
        (32, 50, 85.014_095_88),
        (64, 1, 97.986_721_26),
        (64, 5, 91.523_576_93),
        (64, 10, 80.242_893_93),
        (64, 20, 56.058_555_84),
        (64, 50, -8.759_422_56),
        (128, 1, 96.789_207_38),
        (128, 5, 88.540_708_76),
        (128, 10, 77.842_962_99),
        (128, 20, 54.278_839_17),
        (128, 50, -9.823_833_15),
        (256, 1, 96.150_270_98),
        (256, 5, 89.732_094_36),
        (256, 10, 80.291_491_36),
        (256, 20, 59.778_929_18),
        (256, 50, -0.390_857_84),
    ];
    println!("\nuniform_shift: shipped blur vs jpegli's own horizontal pass");
    println!(
        "{:>6} {:>6} {:>14} {:>14} {:>13} {:>14} {:>13}",
        "size", "shift", "C++ binary", "ours (seq)", "d(seq)", "ours (cpp blur)", "d(cpp)"
    );
    let (mut sum_seq, mut sum_cpp) = (0.0f64, 0.0f64);
    for &(size, shift, cpp_score) in CPP {
        let (a, b) = uniform_pair(size, shift);
        let seq = run(
            a.clone(),
            b.clone(),
            DiagConfig {
                cbrt: CbrtKind::CppFastMath,
                kernels: SimdImpl::Simd,
                horiz: Some(HorizKind::Sequential),
            },
        )
        .0;
        let cpp = run(
            a,
            b,
            DiagConfig {
                cbrt: CbrtKind::CppFastMath,
                kernels: SimdImpl::Simd,
                horiz: Some(HorizKind::CppUnrolled4),
            },
        )
        .0;
        sum_seq += (seq - cpp_score).abs();
        sum_cpp += (cpp - cpp_score).abs();
        println!(
            "{size:>6} {shift:>6} {cpp_score:>14.8} {seq:>14.8} {:>+13.8} {cpp:>14.8} {:>+13.8}",
            seq - cpp_score,
            cpp - cpp_score
        );
    }
    println!(
        "\nmean |delta| vs C++ binary: sequential blur {:.6}, jpegli blur {:.6}",
        sum_seq / CPP.len() as f64,
        sum_cpp / CPP.len() as f64
    );
}

/// Which dispatched kernel is responsible when a tier permutation changes the
/// score. Compares each `#[magetypes]` kernel's raw output between the tokens
/// the host actually has and the scalar-polyfill arm.
#[test]
fn diag_kernel_tier_divergence() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let n = 1024usize;
    let mk = |off: f32| -> Vec<[f32; 3]> {
        (0..n)
            .map(|i| {
                let f = i as f32 / n as f32;
                [0.1 + 0.8 * f + off, 0.2 + 0.7 * (1.0 - f), 0.3 + 0.6 * f]
            })
            .collect()
    };

    let mut xyb_results: Vec<(String, Vec<[f32; 3]>)> = Vec::new();
    let mut ssim_results: Vec<(String, [f64; 6])> = Vec::new();
    let mut edge_results: Vec<(String, [f64; 12])> = Vec::new();

    let plane = |off: f32| -> [Vec<f32>; 3] {
        let mut p = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
        for (i, f) in (0..n).map(|i| i as f32 / n as f32).enumerate() {
            p[0][i] = 0.42 + 0.01 * f + off;
            p[1][i] = 0.30 + 0.20 * f + off;
            p[2][i] = 0.55 + 0.05 * (1.0 - f) + off;
        }
        p
    };
    let a = plane(0.0);
    let b = plane(0.001);
    // Blur-like second moments: close to a*a but not exactly, as the real
    // pipeline produces.
    let mut aa = a.clone();
    let mut bb = b.clone();
    let mut ab = a.clone();
    for c in 0..3 {
        for i in 0..n {
            aa[c][i] = a[c][i] * a[c][i] * 1.000_001;
            bb[c][i] = b[c][i] * b[c][i] * 1.000_001;
            ab[c][i] = a[c][i] * b[c][i] * 1.000_001;
        }
    }

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = mk(0.0);
        crate::xyb_simd::linear_rgb_to_xyb_simd(&mut data);
        xyb_results.push((perm.label.to_string(), data));
        ssim_results.push((
            perm.label.to_string(),
            crate::simd_ops::ssim_map_simd(6, 0, n, 1, &a, &b, &aa, &bb, &ab),
        ));
        edge_results.push((
            perm.label.to_string(),
            crate::simd_ops::edge_diff_map_simd(6, 0, n, 1, &a, &aa, &b, &bb),
        ));
    });

    let worst_xyb = xyb_results[1..]
        .iter()
        .map(|(l, v)| {
            let m = v
                .iter()
                .zip(xyb_results[0].1.iter())
                .flat_map(|(p, q)| p.iter().zip(q.iter()).map(|(a, b)| (a - b).abs()))
                .fold(0f32, f32::max);
            (m, l.clone())
        })
        .fold(
            (0f32, String::new()),
            |acc, x| if x.0 > acc.0 { x } else { acc },
        );
    let worst_ssim = ssim_results[1..]
        .iter()
        .map(|(l, v)| {
            let m = v
                .iter()
                .zip(ssim_results[0].1.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0f64, f64::max);
            (m, l.clone())
        })
        .fold(
            (0f64, String::new()),
            |acc, x| if x.0 > acc.0 { x } else { acc },
        );
    let worst_edge = edge_results[1..]
        .iter()
        .map(|(l, v)| {
            let m = v
                .iter()
                .zip(edge_results[0].1.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0f64, f64::max);
            (m, l.clone())
        })
        .fold(
            (0f64, String::new()),
            |acc, x| if x.0 > acc.0 { x } else { acc },
        );

    println!(
        "\nper-kernel max deviation across {} token permutations:",
        xyb_results.len()
    );
    println!(
        "  linear_rgb_to_xyb_simd  {:.3e}   worst: {}",
        worst_xyb.0, worst_xyb.1
    );
    println!(
        "  ssim_map_simd           {:.3e}   worst: {}",
        worst_ssim.0, worst_ssim.1
    );
    println!(
        "  edge_diff_map_simd      {:.3e}   worst: {}",
        worst_edge.0, worst_edge.1
    );
}

#[test]
fn diag_per_scale_simd_vs_scalar() {
    // Where in the pyramid do the two backends part company?
    for (size, shift) in [(32usize, 50u8), (256, 50), (256, 1)] {
        let (a, b) = uniform_pair(size, shift);
        let (score_s, scales_s) = run(
            a.clone(),
            b.clone(),
            DiagConfig {
                cbrt: CbrtKind::OursSimd,
                kernels: SimdImpl::Simd,
                horiz: None,
            },
        );
        let (score_c, scales_c) = run(
            a,
            b,
            DiagConfig {
                cbrt: CbrtKind::OursSimd,
                kernels: SimdImpl::Scalar,
                horiz: None,
            },
        );
        println!(
            "\nuniform_shift_{shift}_{size}x{size}: simd {score_s:.8} scalar {score_c:.8} \
             delta {:+.8}",
            score_s - score_c
        );
        println!(
            "{:>5} {:>3} {:>14} {:>14} {:>14} {:>14}",
            "scale", "c", "ssim_l1 d", "ssim_l4 d", "ring_l1 d", "blur_l1 d"
        );
        for (i, (s, c)) in scales_s.iter().zip(scales_c.iter()).enumerate() {
            for ch in 0..3 {
                let d0 = s.avg_ssim[ch * 2] - c.avg_ssim[ch * 2];
                let d1 = s.avg_ssim[ch * 2 + 1] - c.avg_ssim[ch * 2 + 1];
                let d2 = s.avg_edgediff[ch * 4] - c.avg_edgediff[ch * 4];
                let d3 = s.avg_edgediff[ch * 4 + 2] - c.avg_edgediff[ch * 4 + 2];
                if d0 != 0.0 || d1 != 0.0 || d2 != 0.0 || d3 != 0.0 {
                    println!("{i:>5} {ch:>3} {d0:>+14.3e} {d1:>+14.3e} {d2:>+14.3e} {d3:>+14.3e}");
                }
            }
        }
    }
}
