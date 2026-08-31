//! SIMD-optimized RGB to XYB conversion.
//!
//! Uses archmage/magetypes for cross-platform SIMD. The cbrt initial estimate stays
//! scalar (integer bit manipulation), but Halley refinement iterations run in SIMD.
//! Matrix multiply, clamp, and XYB transform are also fully vectorized.

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;

// XYB color space constants from jpegli
pub(crate) const K_M02: f32 = 0.078f32;
pub(crate) const K_M00: f32 = 0.30f32;
pub(crate) const K_M01: f32 = 1.0f32 - K_M02 - K_M00;
pub(crate) const K_M12: f32 = 0.078f32;
pub(crate) const K_M10: f32 = 0.23f32;
pub(crate) const K_M11: f32 = 1.0f32 - K_M12 - K_M10;
pub(crate) const K_M20: f32 = 0.243_422_69_f32;
pub(crate) const K_M21: f32 = 0.204_767_45_f32;
pub(crate) const K_M22: f32 = 1.0f32 - K_M20 - K_M21;
pub(crate) const K_B0: f32 = 0.003_793_073_4_f32;

const OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];

const OPSIN_ABSORBANCE_BIAS: f32 = K_B0;

/// Scalar cube root initial estimate via integer bit manipulation.
/// Returns an approximation to cbrt(x) suitable for refinement by Halley iterations.
#[inline(always)]
fn cbrtf_initial_f32(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let hx = (ui & 0x7FFF_FFFF) / 3 + B1;
    f32::from_bits((ui & 0x8000_0000) | hx)
}

/// Two f32 Halley steps from [`cbrtf_initial_f32`] — the cube root the
/// vectorised body evaluates, written out so the scalar arm and the
/// short-plane remainder can compute the *same* value instead of a
/// differently-rounded one.
///
/// Accuracy over the domain the opsin stage produces (`[kB0, 1.004]`,
/// exhaustively swept over all 67.6M f32 values): max 1.92e-7 = 1.75 ulp.
/// A third and fourth Halley step do not improve on that (1.58 ulp) — the
/// iteration is rounding-limited, not convergence-limited — so raising the
/// accuracy would mean carrying the iteration in f64, which is what the
/// old scalar path did and what made the two backends disagree.
///
/// For reference, jpegli's own `CubeRootAndAdd` (what the C++ SSIMULACRA2
/// binary evaluates) measures 3.34 ulp on the same sweep, and documents
/// itself as "6 ulp max error".
#[inline(always)]
fn cbrtf_halley_f32(x: f32) -> f32 {
    let mut t = cbrtf_initial_f32(x);
    // Written to match the vectorised body operation-for-operation:
    // `t *= fma(x, 2, r) / (x + fma(r, 2, 0))`.
    for _ in 0..2 {
        let r = t * t * t;
        t *= x.mul_add(2.0, r) / (x + r.mul_add(2.0, 0.0));
    }
    t
}

/// Fast scalar cube root using bit manipulation + Newton-Raphson in f64.
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

/// Scalar remainder / full-scalar XYB conversion for a single pixel.
#[inline]
fn convert_pixel_scalar(pix: &mut [f32; 3], absorbance_bias: f32) {
    let r = pix[0];
    let g = pix[1];
    let b = pix[2];

    // Unfused, and in the same association order as the vectorised body.
    // See `linear_rgb_to_xyb_inner` for why this must not be an FMA chain.
    let m = &OPSIN_ABSORBANCE_MATRIX;
    let mut mixed0 = m[0] * r + (m[1] * g + (m[2] * b + OPSIN_ABSORBANCE_BIAS));
    let mut mixed1 = m[3] * r + (m[4] * g + (m[5] * b + OPSIN_ABSORBANCE_BIAS));
    let mut mixed2 = m[6] * r + (m[7] * g + (m[8] * b + OPSIN_ABSORBANCE_BIAS));

    mixed0 = mixed0.max(0.0);
    mixed1 = mixed1.max(0.0);
    mixed2 = mixed2.max(0.0);

    // Must be the SAME cube root the vectorised body uses. This function is
    // both the scalar arm of the dispatch and the `len % 8` remainder of the
    // vector arm; when it used the f64 `cbrtf_fast` instead, a plane's last
    // seven pixels were converted with different math than the rest of it,
    // and `SimdImpl::Scalar` computed a different metric than `SimdImpl::Simd`
    // rather than the same metric more slowly.
    mixed0 = cbrtf_halley_f32(mixed0) + absorbance_bias;
    mixed1 = cbrtf_halley_f32(mixed1) + absorbance_bias;
    mixed2 = cbrtf_halley_f32(mixed2) + absorbance_bias;

    pix[0] = 0.5 * (mixed0 - mixed1);
    pix[1] = 0.5 * (mixed0 + mixed1);
    pix[2] = mixed2;
}

/// Generic XYB conversion — processes 8 pixels at a time on all platforms.
#[magetypes(v3, neon, wasm128, scalar)]
fn linear_rgb_to_xyb_inner(token: Token, input: &mut [[f32; 3]]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let absorbance_bias = -cbrtf_fast(OPSIN_ABSORBANCE_BIAS);

    let m00 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[0]);
    let m01 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[1]);
    let m02 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[2]);
    let m10 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[3]);
    let m11 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[4]);
    let m12 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[5]);
    let m20 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[6]);
    let m21 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[7]);
    let m22 = f32x8::splat(token, OPSIN_ABSORBANCE_MATRIX[8]);
    let bias = f32x8::splat(token, OPSIN_ABSORBANCE_BIAS);
    let zero = f32x8::zero(token);
    let two = f32x8::splat(token, 2.0);
    let absorb_bias = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);

    let chunks = input.len() / LANES;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * LANES;

        // AoS -> SoA transpose
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        for i in 0..LANES {
            let p = input[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let r = f32x8::from_array(token, r_arr);
        let g = f32x8::from_array(token, g_arr);
        let b = f32x8::from_array(token, b_arr);

        // Opsin matrix, deliberately NOT an FMA chain.
        //
        // `magetypes` implements `mul_add` for its 8-lane *scalar polyfill* as
        // `a * b + c` — two roundings — while the NEON, AVX2 and AVX-512 arms
        // emit a real fused multiply-add. Every fused expression in a
        // `#[magetypes]` body therefore computes a different value on a target
        // without SIMD (i686 below SSE4.2, wasm without simd128) than on one
        // with it. Measured here: 1.79e-7 in the XYB output, which SSIMULACRA2
        // amplifies to 0.085 on the 0..100 scale (`benchmarks/cpp_parity_2026-08-31.md`).
        //
        // Written unfused, every arm agrees bit-for-bit. The cost against the
        // C++ reference (which does use `MulAdd` here) is at most 1 ulp per
        // term — an order of magnitude below the 1.75 ulp the cube root that
        // consumes these values already carries.
        let mixed0 = m00 * r + (m01 * g + (m02 * b + bias));
        let mixed1 = m10 * r + (m11 * g + (m12 * b + bias));
        let mixed2 = m20 * r + (m21 * g + (m22 * b + bias));

        // Clamp to zero
        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        // Scalar initial estimates (integer bit manipulation — can't vectorize)
        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..LANES {
            est0[i] = cbrtf_initial_f32(est0[i]);
            est1[i] = cbrtf_initial_f32(est1[i]);
            est2[i] = cbrtf_initial_f32(est2[i]);
        }

        // Halley's method iterations in SIMD (3 channels interleaved for ILP)
        let mut t0 = f32x8::from_array(token, est0);
        let mut t1 = f32x8::from_array(token, est1);
        let mut t2 = f32x8::from_array(token, est2);

        // Iteration 1
        let mut r0 = t0 * t0 * t0;
        let mut r1 = t1 * t1 * t1;
        let mut r2 = t2 * t2 * t2;
        t0 *= mixed0.mul_add(two, r0) / (mixed0 + r0.mul_add(two, zero));
        t1 *= mixed1.mul_add(two, r1) / (mixed1 + r1.mul_add(two, zero));
        t2 *= mixed2.mul_add(two, r2) / (mixed2 + r2.mul_add(two, zero));

        // Iteration 2
        r0 = t0 * t0 * t0;
        r1 = t1 * t1 * t1;
        r2 = t2 * t2 * t2;
        t0 *= mixed0.mul_add(two, r0) / (mixed0 + r0.mul_add(two, zero));
        t1 *= mixed1.mul_add(two, r1) / (mixed1 + r1.mul_add(two, zero));
        t2 *= mixed2.mul_add(two, r2) / (mixed2 + r2.mul_add(two, zero));

        let mixed0 = t0 + absorb_bias;
        let mixed1 = t1 + absorb_bias;
        let mixed2 = t2 + absorb_bias;

        // XYB transform
        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let b_out = mixed2;

        // SoA -> AoS transpose and store
        let x_arr = x.to_array();
        let y_arr = y.to_array();
        let b_arr = b_out.to_array();
        for i in 0..LANES {
            input[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Scalar remainder
    for pix in &mut input[chunks * LANES..] {
        convert_pixel_scalar(pix, absorbance_bias);
    }
}

/// Converts linear RGB to XYB in place using SIMD with automatic runtime dispatch.
#[inline]
pub fn linear_rgb_to_xyb_simd(input: &mut [[f32; 3]]) {
    incant!(linear_rgb_to_xyb_inner(input), [v3, neon, wasm128, scalar])
}

/// Converts linear RGB to XYB in place with no SIMD at all.
///
/// Bit-identical to [`linear_rgb_to_xyb_simd`]: the same matrix, the same
/// operation order, and the same [`cbrtf_halley_f32`] cube root, evaluated one
/// pixel at a time. `SimdImpl::Scalar` selects this so that the two backends
/// differ only in how the arithmetic is scheduled, never in what arithmetic is
/// performed. It previously called `yuvxyb`'s conversion, whose f64 cube root
/// left the two backends computing measurably different scores (up to 0.88 on
/// the 0..100 scale — see `benchmarks/cpp_parity_2026-08-31.md`).
pub(crate) fn linear_rgb_to_xyb_scalar(input: &mut [[f32; 3]]) {
    let absorbance_bias = -cbrtf_fast(OPSIN_ABSORBANCE_BIAS);
    for pix in input.iter_mut() {
        convert_pixel_scalar(pix, absorbance_bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbrtf_fast_zero_not_nan() {
        // cbrtf_fast(0.0) must return a finite value (ideally 0.0).
        // Halley iterations on f32 can produce NaN for x=0 when t*r
        // underflows below f32 min subnormal. The f64 path used here
        // avoids that, but this test guards against regressions.
        let result = cbrtf_fast(0.0);
        assert!(
            result.is_finite(),
            "cbrtf_fast(0.0) = {result} (expected finite)"
        );
        assert!(
            result.abs() < 1e-6,
            "cbrtf_fast(0.0) = {result} (expected ~0.0)"
        );
    }
}
