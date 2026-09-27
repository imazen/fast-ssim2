//! Is jpegli's cube root faster than the one fast-ssim2 ships — and how do both
//! compare to `magetypes`' own `f32x8::cbrt_midp`?
//!
//! Three XYB kernels, identical except for the cube root:
//!
//! | kernel | seed | refinement | divisions | round-trips |
//! |---|---|---|--:|--:|
//! | `shipped` | scalar per-lane bit hack (integer `/ 3`) | 2 Halley steps | 2 | 1 (to_array/from_array) |
//! | `jpegli` | **vectorised** `kExpBias - (bits >> 23) * kExpMul` | 3 Newton + 1 corrected step on the *reciprocal* cbrt | **0** | **0** |
//! | `magetypes` | scalar per-lane bit hack (inside `cbrt_midp`) | 2 Halley steps | 2 | 1 |
//!
//! The interesting asymmetry: jpegli iterates on `x^(-1/3)`, so every step is
//! mul/mul/FMA and the final `r*r*x` recovers the cube root — no divides at all,
//! and the seed needs no integer division either, so it vectorises.
//!
//! Run: `cargo bench -p fast-ssim2 --bench cbrt_perf`
//! Requires the study tree's `[patch.crates-io]` on magetypes (>= 0.9.29 for
//! `cbrt_midp`).

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::i32x8 as GenericI32x8;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

// Opsin constants, same values as `src/xyb_simd.rs`.
const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;
const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;
const K_M20: f32 = 0.243_422_69;
const K_M21: f32 = 0.204_767_45;
const K_M22: f32 = 1.0 - K_M20 - K_M21;
const K_B0: f32 = 0.003_793_073_4;
const M: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];

// ---------------------------------------------------------------------------
// kernel 1: what fast-ssim2 ships (scalar bit-hack seed + 2 f32 Halley steps)
// ---------------------------------------------------------------------------

#[inline(always)]
fn cbrtf_initial_f32(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let hx = (ui & 0x7FFF_FFFF) / 3 + B1;
    f32::from_bits((ui & 0x8000_0000) | hx)
}

#[inline]
fn cbrtf_fast(x: f32) -> f32 {
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

#[magetypes(v3, neon, wasm128, scalar)]
fn xyb_shipped_inner(token: Token, input: &mut [[f32; 3]]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let absorbance_bias = -cbrtf_fast(K_B0);
    let m: [f32x8; 9] = core::array::from_fn(|i| f32x8::splat(token, M[i]));
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let two = f32x8::splat(token, 2.0);
    let absorb_bias = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);

    for chunk in input.chunks_exact_mut(LANES) {
        let mut r_arr = [0.0f32; LANES];
        let mut g_arr = [0.0f32; LANES];
        let mut b_arr = [0.0f32; LANES];
        for (i, p) in chunk.iter().enumerate() {
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        let r = f32x8::from_array(token, r_arr);
        let g = f32x8::from_array(token, g_arr);
        let b = f32x8::from_array(token, b_arr);

        let mixed0 = m[0] * r + (m[1] * g + (m[2] * b + bias));
        let mixed1 = m[3] * r + (m[4] * g + (m[5] * b + bias));
        let mixed2 = m[6] * r + (m[7] * g + (m[8] * b + bias));
        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..LANES {
            est0[i] = cbrtf_initial_f32(est0[i]);
            est1[i] = cbrtf_initial_f32(est1[i]);
            est2[i] = cbrtf_initial_f32(est2[i]);
        }
        let mut t0 = f32x8::from_array(token, est0);
        let mut t1 = f32x8::from_array(token, est1);
        let mut t2 = f32x8::from_array(token, est2);
        for _ in 0..2 {
            let r0 = t0 * t0 * t0;
            let r1 = t1 * t1 * t1;
            let r2 = t2 * t2 * t2;
            t0 *= mixed0.mul_add(two, r0) / (mixed0 + r0.mul_add(two, zero));
            t1 *= mixed1.mul_add(two, r1) / (mixed1 + r1.mul_add(two, zero));
            t2 *= mixed2.mul_add(two, r2) / (mixed2 + r2.mul_add(two, zero));
        }
        let mixed0 = t0 + absorb_bias;
        let mixed1 = t1 + absorb_bias;
        let mixed2 = t2 + absorb_bias;

        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let xa = x.to_array();
        let ya = y.to_array();
        let ba = mixed2.to_array();
        for (i, p) in chunk.iter_mut().enumerate() {
            *p = [xa[i], ya[i], ba[i]];
        }
    }
}

// ---------------------------------------------------------------------------
// kernel 2: jpegli `CubeRootAndAdd`, fully vectorised (no divides, no seed
// round-trip). `lib/base/fast_math-inl.h`.
// ---------------------------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
fn xyb_jpegli_inner(token: Token, input: &mut [[f32; 3]]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;
    const LANES: usize = 8;

    let absorbance_bias = -(f64::from(K_B0).cbrt() as f32);
    let m: [f32x8; 9] = core::array::from_fn(|i| f32x8::splat(token, M[i]));
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let half = f32x8::splat(token, 0.5);
    let add = f32x8::splat(token, absorbance_bias);
    let k1_3 = f32x8::splat(token, 1.0 / 3.0);
    let k4_3 = f32x8::splat(token, 4.0 / 3.0);
    let exp_bias = i32x8::splat(token, 0x5480_0000);
    let exp_mul = i32x8::splat(token, 0x002A_AAAA);
    let izero = i32x8::zero(token);

    // cbrt(x) + add, exactly as CubeRootAndAdd computes it.
    let cbrt_add = |x: f32x8| -> f32x8 {
        let xa_3 = k1_3 * x;
        let m1 = x.bitcast_to_i32();
        let m2 = exp_bias - m1.shr_arithmetic_const::<23>() * exp_mul;
        // IfThenZeroElse(m1 == 0, ...)
        let m2 = i32x8::blend(m1.simd_eq(izero), izero, m2);
        let mut r = m2.bitcast_to_f32();
        for _ in 0..3 {
            let r2 = r * r;
            // NegMulAdd(xa_3, r2*r2, k4_3 * r)
            r = (zero - xa_3).mul_add(r2 * r2, k4_3 * r);
        }
        let r2 = r * r;
        r = k1_3.mul_add((zero - x).mul_add(r2 * r2, r), r);
        let r2 = r * r;
        r2.mul_add(x, add)
    };

    for chunk in input.chunks_exact_mut(LANES) {
        let mut r_arr = [0.0f32; LANES];
        let mut g_arr = [0.0f32; LANES];
        let mut b_arr = [0.0f32; LANES];
        for (i, p) in chunk.iter().enumerate() {
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        let r = f32x8::from_array(token, r_arr);
        let g = f32x8::from_array(token, g_arr);
        let b = f32x8::from_array(token, b_arr);

        // jpegli fuses this chain; keep it identical to the C++.
        let mixed0 = m[0].mul_add(r, m[1].mul_add(g, m[2].mul_add(b, bias)));
        let mixed1 = m[3].mul_add(r, m[4].mul_add(g, m[5].mul_add(b, bias)));
        let mixed2 = m[6].mul_add(r, m[7].mul_add(g, m[8].mul_add(b, bias)));
        let mixed0 = cbrt_add(mixed0.max(zero));
        let mixed1 = cbrt_add(mixed1.max(zero));
        let mixed2 = cbrt_add(mixed2.max(zero));

        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let xa = x.to_array();
        let ya = y.to_array();
        let ba = mixed2.to_array();
        for (i, p) in chunk.iter_mut().enumerate() {
            *p = [xa[i], ya[i], ba[i]];
        }
    }
}

// ---------------------------------------------------------------------------
// kernel 3: magetypes' own `f32x8::cbrt_midp` (same algorithm shape as ours:
// scalar bit-hack seed + 2 Halley steps, plus sign/zero handling)
// ---------------------------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
fn xyb_magetypes_inner(token: Token, input: &mut [[f32; 3]]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let absorbance_bias = -cbrtf_fast(K_B0);
    let m: [f32x8; 9] = core::array::from_fn(|i| f32x8::splat(token, M[i]));
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let half = f32x8::splat(token, 0.5);
    let absorb_bias = f32x8::splat(token, absorbance_bias);

    for chunk in input.chunks_exact_mut(LANES) {
        let mut r_arr = [0.0f32; LANES];
        let mut g_arr = [0.0f32; LANES];
        let mut b_arr = [0.0f32; LANES];
        for (i, p) in chunk.iter().enumerate() {
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        let r = f32x8::from_array(token, r_arr);
        let g = f32x8::from_array(token, g_arr);
        let b = f32x8::from_array(token, b_arr);

        let mixed0 = m[0] * r + (m[1] * g + (m[2] * b + bias));
        let mixed1 = m[3] * r + (m[4] * g + (m[5] * b + bias));
        let mixed2 = m[6] * r + (m[7] * g + (m[8] * b + bias));
        let mixed0 = mixed0.max(zero).cbrt_midp() + absorb_bias;
        let mixed1 = mixed1.max(zero).cbrt_midp() + absorb_bias;
        let mixed2 = mixed2.max(zero).cbrt_midp() + absorb_bias;

        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let xa = x.to_array();
        let ya = y.to_array();
        let ba = mixed2.to_array();
        for (i, p) in chunk.iter_mut().enumerate() {
            *p = [xa[i], ya[i], ba[i]];
        }
    }
}

fn xyb_shipped(input: &mut [[f32; 3]]) {
    incant!(xyb_shipped_inner(input), [v3, neon, wasm128, scalar])
}
fn xyb_jpegli(input: &mut [[f32; 3]]) {
    incant!(xyb_jpegli_inner(input), [v3, neon, wasm128, scalar])
}
fn xyb_magetypes(input: &mut [[f32; 3]]) {
    incant!(xyb_magetypes_inner(input), [v3, neon, wasm128, scalar])
}

fn make_input(pixels: usize) -> Vec<[f32; 3]> {
    // Deterministic LCG over [0, 1] — the linear-RGB range the real pipeline
    // hands the opsin stage.
    let mut s: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u32 << 24) as f32)
    };
    (0..pixels).map(|_| [next(), next(), next()]).collect()
}

fn agreement() {
    // Report how far the three kernels sit from each other, so a speed claim is
    // never mistaken for an equivalence claim.
    let base = make_input(1 << 16);
    let mut a = base.clone();
    let mut b = base.clone();
    let mut c = base.clone();
    xyb_shipped(&mut a);
    xyb_jpegli(&mut b);
    xyb_magetypes(&mut c);
    let maxd = |u: &[[f32; 3]], v: &[[f32; 3]]| {
        u.iter()
            .zip(v)
            .flat_map(|(p, q)| (0..3).map(move |i| (f64::from(p[i]) - f64::from(q[i])).abs()))
            .fold(0.0f64, f64::max)
    };
    println!(
        "XYB agreement over {} pixels: |shipped-jpegli| {:.4e}   |shipped-magetypes| {:.4e}   |jpegli-magetypes| {:.4e}",
        base.len(),
        maxd(&a, &b),
        maxd(&a, &c),
        maxd(&b, &c),
    );
}

fn bench(c: &mut Criterion) {
    agreement();
    for pixels in [64 * 64usize, 256 * 256, 1024 * 1024] {
        let input = make_input(pixels);
        let mut group = c.benchmark_group(format!("xyb_{pixels}px"));
        for (name, f) in [
            // The kernels are in-place and destructive, so every arm re-copies
            // the input inside the timed region. `copy_only` measures that
            // memcpy floor so it can be subtracted from the other three.
            ("copy_only", (|_: &mut [[f32; 3]]| {}) as fn(&mut [[f32; 3]])),
            ("shipped", xyb_shipped as fn(&mut [[f32; 3]])),
            ("jpegli", xyb_jpegli as fn(&mut [[f32; 3]])),
            ("magetypes_cbrt_midp", xyb_magetypes as fn(&mut [[f32; 3]])),
        ] {
            let mut buf = input.clone();
            group.bench_function(name, |bencher| {
                bencher.iter(|| {
                    buf.copy_from_slice(&input);
                    f(&mut buf);
                    std::hint::black_box(&buf);
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
