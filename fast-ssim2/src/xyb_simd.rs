//! SIMD-optimized RGB to XYB conversion.
//!
//! Uses archmage/magetypes for cross-platform SIMD. The cbrt initial estimate stays
//! scalar (integer bit manipulation), but Halley refinement iterations run in SIMD.
//! Matrix multiply, clamp, and XYB transform are also fully vectorized.

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::i32x8 as GenericI32x8;

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

/// jpegli `CubeRootAndAdd` (`lib/base/fast_math-inl.h`) — `cbrt(x) + add`, and
/// the exact expression the C++ SSIMULACRA2 binary evaluates. `libjxl` and
/// `cloudinary/ssimulacra2` carry byte-identical copies of it.
///
/// Newton–Raphson on the *reciprocal* cube root: `r` converges to `x^(-1/3)`
/// and the final `r*r*x` recovers the root, so there is no division anywhere
/// and the seed is pure integer arithmetic. That is why it is both faster than
/// the two f32 Halley steps this crate used through 0.9.0 (measured: 2.7–2.9%
/// on NEON, 6–8% on AVX2 in the XYB kernel) and bit-identical to the reference
/// rather than 3e-8 away from it. See `benchmarks/cbrt_perf_2026-09-09.md`.
///
/// Highway's `NegMulAdd(a, b, c)` is the fused `c - a*b` and `MulAdd(a, b, c)`
/// is `a*b + c`; the association below matches it operation for operation, and
/// the vectorised body in [`linear_rgb_to_xyb_inner`] matches this one, so the
/// scalar arm, the `len % 8` remainder and the vector arm all agree bit-for-bit.
///
/// Accuracy is deliberately *not* the goal here — this approximation is the
/// least accurate of the three we have measured (0.72 mean / 5 max ulp against
/// f64 `cbrt`, versus 0.49/3 for the old Halley pair and 0 for a f64 Newton
/// pair). Matching the reference is the goal, and the reference computes this.
#[inline(always)]
fn cbrt_and_add_jpegli(x: f32, add: f32) -> f32 {
    const K_EXP_BIAS: i32 = 0x5480_0000; // cast(1.) + cast(1.) / 3
    const K_EXP_MUL: i32 = 0x002A_AAAA; // shifted 1/3
    const K1_3: f32 = 1.0 / 3.0;
    const K4_3: f32 = 4.0 / 3.0;

    let xa_3 = K1_3 * x;
    let m1 = x.to_bits() as i32;
    // Special case for 0, exactly as the C++ does (`IfThenZeroElse`): an
    // exponent of 0 makes `kExpBias - exp/3` wrong and would feed NaN forward.
    let m2 = if m1 == 0 {
        0
    } else {
        K_EXP_BIAS - (m1 >> 23) * K_EXP_MUL
    };
    let mut r = f32::from_bits(m2 as u32);

    for _ in 0..3 {
        let r2 = r * r;
        r = (-xa_3).mul_add(r2 * r2, K4_3 * r);
    }
    let mut r2 = r * r;
    r = K1_3.mul_add((-x).mul_add(r2 * r2, r), r);
    r2 = r * r;
    // The C++ folds the additive constant into this last operation rather than
    // rounding `cbrt(x)` first and adding after.
    r2.mul_add(x, add)
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
    // rather than the same metric more slowly. The additive constant is folded
    // into the cube root's last operation, as the C++ does.
    mixed0 = cbrt_and_add_jpegli(mixed0, absorbance_bias);
    mixed1 = cbrt_and_add_jpegli(mixed1, absorbance_bias);
    mixed2 = cbrt_and_add_jpegli(mixed2, absorbance_bias);

    pix[0] = 0.5 * (mixed0 - mixed1);
    pix[1] = 0.5 * (mixed0 + mixed1);
    pix[2] = mixed2;
}

/// Generic XYB conversion — processes 8 pixels at a time on all platforms.
#[magetypes(v3, neon, wasm128, scalar)]
fn linear_rgb_to_xyb_inner(token: Token, input: &mut [[f32; 3]]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;
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
    let absorb_bias = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);
    // jpegli `CubeRootAndAdd` constants.
    let k1_3 = f32x8::splat(token, 1.0 / 3.0);
    let k4_3 = f32x8::splat(token, 4.0 / 3.0);
    let exp_bias = i32x8::splat(token, 0x5480_0000);
    let exp_mul = i32x8::splat(token, 0x002A_AAAA);
    let izero = i32x8::zero(token);

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

        // jpegli `CubeRootAndAdd`, vectorised: same operations, same order and
        // the same FMA association as `cbrt_and_add_jpegli`, so the two arms
        // and the `len % 8` remainder all produce identical bits. The seed is
        // integer-only, so unlike the Halley pair it needs no per-lane scalar
        // round trip, and the iteration converges on `x^(-1/3)` so there is no
        // division to wait on.
        let cbrt_and_add = |x: f32x8| -> f32x8 {
            let xa_3 = k1_3 * x;
            let m1 = x.bitcast_to_i32();
            let m2 = exp_bias - m1.shr_arithmetic_const::<23>() * exp_mul;
            // `IfThenZeroElse(m1 == 0, ...)`
            let m2 = i32x8::blend(m1.simd_eq(izero), izero, m2);
            let mut r = m2.bitcast_to_f32();
            for _ in 0..3 {
                let r2 = r * r;
                // `NegMulAdd(xa_3, r2*r2, k4_3 * r)`
                r = (zero - xa_3).mul_add(r2 * r2, k4_3 * r);
            }
            let r2 = r * r;
            r = k1_3.mul_add((zero - x).mul_add(r2 * r2, r), r);
            let r2 = r * r;
            r2.mul_add(x, absorb_bias)
        };

        let mixed0 = cbrt_and_add(mixed0);
        let mixed1 = cbrt_and_add(mixed1);
        let mixed2 = cbrt_and_add(mixed2);

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
/// Pixels are independent here, so this splits cleanly. The chunk size is a
/// multiple of the vector body's lane count, so every chunk but the last runs
/// the same vector path it would have run inside one big call and the scalar
/// remainder still only appears once, at the end of the plane — the result is
/// bit-identical to the serial path, which `simd_consistency` and the pinned
/// `implementation_parity` scores both check.
#[cfg(feature = "rayon")]
pub fn linear_rgb_to_xyb_simd(input: &mut [[f32; 3]]) {
    use rayon::prelude::*;
    // 8 lanes x 512 = one chunk per ~12 KB of pixels: big enough that the
    // dispatch and join overhead disappears, small enough to keep every worker
    // fed on a small image.
    const CHUNK: usize = 8 * 512;
    if input.len() < crate::simd_ops::PAR_MIN_SAMPLES {
        incant!(linear_rgb_to_xyb_inner(input), [v3, neon, wasm128, scalar]);
        return;
    }
    input.par_chunks_mut(CHUNK).for_each(|chunk| {
        incant!(linear_rgb_to_xyb_inner(chunk), [v3, neon, wasm128, scalar]);
    });
}

#[cfg(not(feature = "rayon"))]
pub fn linear_rgb_to_xyb_simd(input: &mut [[f32; 3]]) {
    incant!(linear_rgb_to_xyb_inner(input), [v3, neon, wasm128, scalar])
}

/// Converts linear RGB to XYB in place with no SIMD at all.
///
/// Bit-identical to [`linear_rgb_to_xyb_simd`]: the same matrix, the same
/// operation order, and the same [`cbrt_and_add_jpegli`] cube root, evaluated one
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
