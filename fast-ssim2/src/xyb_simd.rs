//! SIMD-optimized RGB to XYB conversion.
//!
//! Uses archmage/magetypes for cross-platform SIMD. The cbrt function stays scalar
//! (bit manipulation + Newton-Raphson in f64 doesn't vectorize well); SIMD is used
//! for the matrix multiply, clamp, and XYB transform surrounding it.

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;

// XYB color space constants from jpegli
const K_M02: f32 = 0.078f32;
const K_M00: f32 = 0.30f32;
const K_M01: f32 = 1.0f32 - K_M02 - K_M00;
const K_M12: f32 = 0.078f32;
const K_M10: f32 = 0.23f32;
const K_M11: f32 = 1.0f32 - K_M12 - K_M10;
const K_M20: f32 = 0.243_422_69_f32;
const K_M21: f32 = 0.204_767_45_f32;
const K_M22: f32 = 1.0f32 - K_M20 - K_M21;
const K_B0: f32 = 0.003_793_073_4_f32;

const OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];

const OPSIN_ABSORBANCE_BIAS: f32 = K_B0;

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

    let mut mixed0 = OPSIN_ABSORBANCE_MATRIX[0].mul_add(
        r,
        OPSIN_ABSORBANCE_MATRIX[1].mul_add(
            g,
            OPSIN_ABSORBANCE_MATRIX[2].mul_add(b, OPSIN_ABSORBANCE_BIAS),
        ),
    );
    let mut mixed1 = OPSIN_ABSORBANCE_MATRIX[3].mul_add(
        r,
        OPSIN_ABSORBANCE_MATRIX[4].mul_add(
            g,
            OPSIN_ABSORBANCE_MATRIX[5].mul_add(b, OPSIN_ABSORBANCE_BIAS),
        ),
    );
    let mut mixed2 = OPSIN_ABSORBANCE_MATRIX[6].mul_add(
        r,
        OPSIN_ABSORBANCE_MATRIX[7].mul_add(
            g,
            OPSIN_ABSORBANCE_MATRIX[8].mul_add(b, OPSIN_ABSORBANCE_BIAS),
        ),
    );

    mixed0 = mixed0.max(0.0);
    mixed1 = mixed1.max(0.0);
    mixed2 = mixed2.max(0.0);

    mixed0 = cbrtf_fast(mixed0) + absorbance_bias;
    mixed1 = cbrtf_fast(mixed1) + absorbance_bias;
    mixed2 = cbrtf_fast(mixed2) + absorbance_bias;

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

        // Matrix multiply with FMA
        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)));

        // Clamp to zero
        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        // Extract, apply scalar cbrt, reload
        let mut m0_arr = mixed0.to_array();
        let mut m1_arr = mixed1.to_array();
        let mut m2_arr = mixed2.to_array();
        for i in 0..LANES {
            m0_arr[i] = cbrtf_fast(m0_arr[i]);
            m1_arr[i] = cbrtf_fast(m1_arr[i]);
            m2_arr[i] = cbrtf_fast(m2_arr[i]);
        }

        let mixed0 = f32x8::from_array(token, m0_arr) + absorb_bias;
        let mixed1 = f32x8::from_array(token, m1_arr) + absorb_bias;
        let mixed2 = f32x8::from_array(token, m2_arr) + absorb_bias;

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
