//! PU21-native XYB conversion for HDR input — **experimental** (`hdr-pu` feature).
//!
//! Replaces the cube-root opsin nonlinearity with PU21 (Mantiuk & Azimi, PCS
//! 2021) at the perceptual-encoding layer, consuming **absolute-luminance**
//! linear RGB (cd/m²). This is the correct way to PU-adapt a metric that has
//! its own perceptual transform: feeding PU-encoded values as *input* (the
//! "outer shell" approach) lets the cube-root re-process the already-uniform
//! PU values and caps HDR correlation — measured SROCC 0.59–0.61 on UPIQ HDR
//! vs 0.694 for the integrated form (zensim) and 0.740 for classic PU-SSIM.
//! See imazen/zenmetrics#25.
//!
//! Recipe mirrors zensim's validated `linear_to_pu_xyb_planar_into`:
//! opsin LMS mix on absolute nits → PU21(banding_glare)/PU21(100 cd/m²) per
//! channel (100-nit reference white → ~1.0, the range the cube-root white
//! point sits in) → opponent formation with the positive offsets folded in.
//! The X chroma scale is 4 (not the cube-root path's 14): PU-space opsin
//! differences are already large, and HDR validation favors
//! luminance-dominant weighting (PU-SSIM, the strongest non-learned HDR
//! baseline, is luminance-only).
//!
//! PU21 coefficients are the published gfxdisp/pu21 `banding_glare` set —
//! byte-identical to `zensim::pu21` and `zenmetrics-api::hdr`, pinned by the
//! reference-parity test below (independent float64 goldens, same table as
//! zensim's `reference_parity_gfxdisp_goldens`).

use crate::xyb_simd::{K_B0, K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22};

/// PU21 `banding_glare` parameters `[p1..p7]` (gfxdisp/pu21, 2020-02-06).
#[allow(clippy::excessive_precision)]
const PU21_P: [f32; 7] = [
    0.353_487_901,
    0.373_465_862_9,
    8.277_049_286e-5,
    0.906_256_262_7,
    0.091_503_031_66,
    0.909_951_720_4,
    596.314_814_2,
];
/// Operating range of the encoding (cd/m²).
const PU21_L_MIN: f32 = 0.005;
const PU21_L_MAX: f32 = 10000.0;
/// PU21(100 cd/m²) — normalizes a 100-nit reference white to ~1.0.
const PU_WHITE: f32 = 256.3;
/// Opponent X amplification in PU space (cube-root path uses 14).
const PU_X_SCALE: f32 = 4.0;

/// PU21 encode: absolute luminance (cd/m²) → perceptually-uniform value.
/// `V = max(p7·(((p1 + p2·Y^p4)/(1 + p3·Y^p4))^p5 − p6), 0)`, `Y` clamped.
#[inline]
fn pu21_encode(y: f32) -> f32 {
    let y = y.clamp(PU21_L_MIN, PU21_L_MAX);
    let yp = y.powf(PU21_P[3]);
    let inner = (PU21_P[0] + PU21_P[1] * yp) / (1.0 + PU21_P[2] * yp);
    (PU21_P[6] * (inner.powf(PU21_P[4]) - PU21_P[5])).max(0.0)
}

/// In-place: absolute-luminance linear RGB (cd/m²) → positive PU-XYB.
///
/// Output is already positive-shifted (the cube-root path's
/// `make_positive_xyb` offsets are folded in) — do NOT apply
/// `make_positive_xyb` after this.
pub(crate) fn linear_nits_to_pu_xyb(input: &mut [[f32; 3]]) {
    for pix in input.iter_mut() {
        let (r, g, b) = (pix[0], pix[1], pix[2]);
        // Opsin LMS mix on absolute luminance — no [0,1] clamp (HDR > 1).
        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = pu21_encode(mixed0) / PU_WHITE;
        let c1 = pu21_encode(mixed1) / PU_WHITE;
        let c2 = pu21_encode(mixed2) / PU_WHITE;

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);
        pix[0] = x.mul_add(PU_X_SCALE, 0.42);
        pix[1] = y + 0.01;
        pix[2] = (c2 - y) + 0.55;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same independent float64 gfxdisp goldens as zensim's
    /// `reference_parity_gfxdisp_goldens` (generator:
    /// zensim `scripts/pu21_golden.py`) — banding_glare row. Keeps this copy
    /// from drifting from zensim / zenmetrics / the published reference.
    #[test]
    fn reference_parity_gfxdisp_goldens_banding_glare() {
        const YS: [f32; 7] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0];
        const WANT: [f64; 7] = [
            0.3722, 5.7171, 36.5439, 123.6475, 256.3839, 420.0969, 595.3939,
        ];
        for (&y, &want) in YS.iter().zip(WANT.iter()) {
            let got = pu21_encode(y) as f64;
            let tol = 0.1 + 5e-3 * want;
            assert!(
                (got - want).abs() <= tol,
                "pu21_encode({y}) = {got}, gfxdisp ref {want} (tol {tol})"
            );
        }
    }

    #[test]
    fn white_point_lands_near_one() {
        // 100 cd/m² (SDR reference white) → ~1.0 after normalization, the
        // same range the cube-root XYB white point occupies.
        let v = pu21_encode(100.0) / PU_WHITE;
        assert!((v - 1.0).abs() < 0.01, "PU(100)/PU_WHITE = {v}");
    }

    #[test]
    fn pu_xyb_outputs_are_positive_calibrated() {
        // Gray ramp from deep shadow to 4000-nit highlight: outputs must stay
        // in the positive ranges the SSIM kernels expect (X near 0.42, Y in
        // (0, ~2.5), B near 0.55).
        for nits in [0.01, 0.5, 5.0, 100.0, 1000.0, 4000.0] {
            let mut px = [[nits, nits, nits]];
            linear_nits_to_pu_xyb(&mut px);
            let [x, y, b] = px[0];
            assert!((0.3..0.6).contains(&x), "X({nits}) = {x}");
            assert!(y > 0.0 && y < 2.5, "Y({nits}) = {y}");
            assert!((0.3..0.9).contains(&b), "B({nits}) = {b}");
        }
        // Monotone luminance response on the gray axis.
        let ys: Vec<f32> = [1.0f32, 10.0, 100.0, 1000.0]
            .iter()
            .map(|&n| {
                let mut px = [[n, n, n]];
                linear_nits_to_pu_xyb(&mut px);
                px[0][1]
            })
            .collect();
        assert!(ys.windows(2).all(|w| w[1] > w[0]), "Y not monotone: {ys:?}");
    }
}
