/// SIMD-optimized operations for SSIMULACRA2 computation
///
/// Uses archmage/magetypes for cross-platform SIMD with runtime dispatch.
/// A single `#[magetypes]` generic function handles all platforms via
/// `GenericF32x8<Token>` — the polyfill emulates 8-lane on 128-bit targets.
use archmage::incant;
use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;

use crate::weights::{EDGE_HAS_WEIGHT, NUM_SCALES, SSIM_HAS_WEIGHT};

const C2: f32 = 0.0009f32;

// =============================================================================
// SSIM map
// =============================================================================

/// Generic SSIM map computation — processes 8 pixels at a time on all platforms.
#[magetypes(v3, neon, wasm128, scalar)]
fn ssim_map_inner(
    token: Token,
    scales_n: usize,
    scale_idx: usize,
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let c2_simd = f32x8::splat(token, C2);
    let one_simd = f32x8::splat(token, 1.0);
    let two_simd = f32x8::splat(token, 2.0);
    let zero_simd = f32x8::zero(token);

    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 2];

    // Skip table is parametric in `scales_n` because `score()` walks WEIGHT
    // linearly — at smaller scale counts the cells used for `(c, scale_idx)`
    // shift in the layout. See weights.rs for the worked example.
    let skip_table = SSIM_HAS_WEIGHT[scales_n.min(NUM_SCALES)];

    for c in 0..3 {
        // Lossless skip — both L1 and L4 SSIM weights for this
        // `(scales_n, channel, scale_idx)` are zero, so the per-channel
        // contribution to the final score is `0.0 * value = 0.0`. Leave the
        // two output slots at the initial 0.0.
        if scale_idx < NUM_SCALES && !skip_table[c][scale_idx] {
            continue;
        }
        let mut sum_d = 0.0f64;
        let mut sum_d4 = 0.0f64;

        let m1c = &m1[c];
        let m2c = &m2[c];
        let s11c = &s11[c];
        let s22c = &s22[c];
        let s12c = &s12[c];

        let total = m1c.len();
        let chunks = total / LANES;

        for chunk in 0..chunks {
            let base = chunk * LANES;

            let mu1 = f32x8::from_array(token, m1c[base..][..LANES].try_into().unwrap());
            let mu2 = f32x8::from_array(token, m2c[base..][..LANES].try_into().unwrap());
            let s11_vals = f32x8::from_array(token, s11c[base..][..LANES].try_into().unwrap());
            let s22_vals = f32x8::from_array(token, s22c[base..][..LANES].try_into().unwrap());
            let s12_vals = f32x8::from_array(token, s12c[base..][..LANES].try_into().unwrap());

            let mu11 = mu1 * mu1;
            let mu22 = mu2 * mu2;
            let mu12 = mu1 * mu2;
            let mu_diff = mu1 - mu2;

            // NOT `mu_diff.mul_add(-mu_diff, one)`. `magetypes`' 8-lane scalar
            // polyfill implements `mul_add` as `a * b + c` (two roundings)
            // while its NEON/AVX2/AVX-512 arms emit a real FMA, so any fused
            // expression here scores differently on a target without SIMD —
            // measured at 0.085 on a 32x32 image before this was made explicit.
            // The C++ reference writes this term unfused as well.
            let num_m = one_simd - mu_diff * mu_diff;
            // `2 * x` is exact in binary floating point, so this one rounds
            // identically fused or not; kept as an FMA for the free op.
            let num_s = two_simd.mul_add(s12_vals - mu12, c2_simd);
            let denom_s = (s11_vals - mu11) + (s22_vals - mu22) + c2_simd;

            let d = (one_simd - (num_m * num_s) / denom_s).max(zero_simd);
            let d2 = d * d;
            let d4 = d2 * d2;

            sum_d += d.reduce_add() as f64;
            sum_d4 += d4.reduce_add() as f64;
        }

        // Scalar remainder
        for x in (chunks * LANES)..total {
            let mu1 = m1c[x];
            let mu2 = m2c[x];
            let mu_diff = mu1 - mu2;

            // Same unfused form as the vectorised body above.
            let num_m = 1.0f32 - mu_diff * mu_diff;
            let num_s = 2.0f32.mul_add(s12c[x] - mu1 * mu2, C2);
            let denom_s = (s11c[x] - mu1 * mu1) + (s22c[x] - mu2 * mu2) + C2;
            let d = (1.0f32 - (num_m * num_s) / denom_s).max(0.0f32);
            let d2 = d * d;
            let d4 = d2 * d2;
            sum_d += f64::from(d);
            sum_d4 += f64::from(d4);
        }

        plane_averages[c * 2] = one_per_pixels * sum_d;
        plane_averages[c * 2 + 1] = (one_per_pixels * sum_d4).sqrt().sqrt();
    }

    plane_averages
}

/// SIMD-optimized SSIM map computation with automatic runtime dispatch.
pub(crate) fn ssim_map_simd(
    scales_n: usize,
    scale_idx: usize,
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    incant!(
        ssim_map_inner(scales_n, scale_idx, width, height, m1, m2, s11, s22, s12),
        [v3, neon, wasm128, scalar]
    )
}

// =============================================================================
// Edge difference map
// =============================================================================

/// Generic edge difference map — processes 8 pixels at a time on all platforms.
#[magetypes(v3, neon, wasm128, scalar)]
fn edge_diff_map_inner(
    token: Token,
    scales_n: usize,
    scale_idx: usize,
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 4];

    let one_simd = f32x8::splat(token, 1.0);
    let zero_simd = f32x8::zero(token);

    let skip_table = EDGE_HAS_WEIGHT[scales_n.min(NUM_SCALES)];

    for c in 0..3 {
        // Lossless skip — all four edge-diff weights for this
        // `(scales_n, c, scale_idx)` are zero.
        if scale_idx < NUM_SCALES && !skip_table[c][scale_idx] {
            continue;
        }
        let mut sum_artifact = 0.0f64;
        let mut sum_artifact4 = 0.0f64;
        let mut sum_detail = 0.0f64;
        let mut sum_detail4 = 0.0f64;

        let img1c = &img1[c];
        let mu1c = &mu1[c];
        let img2c = &img2[c];
        let mu2c = &mu2[c];

        let total = img1c.len();
        let chunks = total / LANES;

        for chunk in 0..chunks {
            let base = chunk * LANES;

            let r1 = f32x8::from_array(token, img1c[base..][..LANES].try_into().unwrap());
            let rm1 = f32x8::from_array(token, mu1c[base..][..LANES].try_into().unwrap());
            let r2 = f32x8::from_array(token, img2c[base..][..LANES].try_into().unwrap());
            let rm2 = f32x8::from_array(token, mu2c[base..][..LANES].try_into().unwrap());

            let d1_temp = r1 - rm1;
            let diff1 = d1_temp.max(-d1_temp);
            let d2_temp = r2 - rm2;
            let diff2 = d2_temp.max(-d2_temp);

            // (1 + diff2) / (1 + diff1) - 1  ==  (diff2 - diff1) / (1 + diff1)
            //
            // Algebraically identical; numerically not. The reference form
            // subtracts two nearby quantities *after* rounding them, so in f32
            // the result carries an absolute error of ~1 ulp of 1.0 (6e-8) no
            // matter how small the true value is — on smooth content, where
            // both diffs are ~1e-7, that is 100% error, and `max(d1, 0)`
            // rectifies it into a one-directional bias. The right-hand form has
            // no cancellation: its error is ~1 ulp *relative*. The C++ reference
            // computes the left form in f64, where the same cancellation costs
            // only ~1e-16 absolute, which is why it can afford it and we cannot.
            let d1 = (diff2 - diff1) / (one_simd + diff1);

            let artifact = d1.max(zero_simd);
            let detail_lost = (-d1).max(zero_simd);

            let a2 = artifact * artifact;
            let a4 = a2 * a2;
            let dl2 = detail_lost * detail_lost;
            let dl4 = dl2 * dl2;

            sum_artifact += artifact.reduce_add() as f64;
            sum_artifact4 += a4.reduce_add() as f64;
            sum_detail += detail_lost.reduce_add() as f64;
            sum_detail4 += dl4.reduce_add() as f64;
        }

        // Scalar remainder — same expression as the vectorised body above, so
        // a pixel's treatment does not depend on its index modulo LANES.
        for x in (chunks * LANES)..total {
            let diff1 = (img1c[x] - mu1c[x]).abs();
            let diff2 = (img2c[x] - mu2c[x]).abs();
            let d1 = (diff2 - diff1) / (1.0f32 + diff1);
            let artifact = d1.max(0.0);
            let detail_lost = (-d1).max(0.0);
            let a2 = artifact * artifact;
            let dl2 = detail_lost * detail_lost;
            sum_artifact += f64::from(artifact);
            sum_artifact4 += f64::from(a2 * a2);
            sum_detail += f64::from(detail_lost);
            sum_detail4 += f64::from(dl2 * dl2);
        }

        plane_averages[c * 4] = one_per_pixels * sum_artifact;
        plane_averages[c * 4 + 1] = (one_per_pixels * sum_artifact4).sqrt().sqrt();
        plane_averages[c * 4 + 2] = one_per_pixels * sum_detail;
        plane_averages[c * 4 + 3] = (one_per_pixels * sum_detail4).sqrt().sqrt();
    }

    plane_averages
}

/// SIMD-optimized edge difference map with automatic runtime dispatch.
pub(crate) fn edge_diff_map_simd(
    scales_n: usize,
    scale_idx: usize,
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    incant!(
        edge_diff_map_inner(scales_n, scale_idx, width, height, img1, mu1, img2, mu2),
        [v3, neon, wasm128, scalar]
    )
}

// =============================================================================
// Image multiplication
// =============================================================================

/// Generic image multiplication — processes 8 pixels at a time on all platforms.
#[magetypes(v3, neon, wasm128, scalar)]
fn image_multiply_inner(
    token: Token,
    img1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    out: &mut [Vec<f32>; 3],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    for c in 0..3 {
        let plane1 = &img1[c];
        let plane2 = &img2[c];
        let out_plane = &mut out[c];

        let chunks = plane1.len() / LANES;

        for chunk in 0..chunks {
            let base = chunk * LANES;
            let p1 = f32x8::from_array(token, plane1[base..][..LANES].try_into().unwrap());
            let p2 = f32x8::from_array(token, plane2[base..][..LANES].try_into().unwrap());
            let result = p1 * p2;
            out_plane[base..base + LANES].copy_from_slice(&result.to_array());
        }

        // Scalar remainder
        for i in (chunks * LANES)..plane1.len() {
            out_plane[i] = plane1[i] * plane2[i];
        }
    }
}

/// SIMD-optimized image multiplication with automatic runtime dispatch.
pub(crate) fn image_multiply_simd(
    img1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    out: &mut [Vec<f32>; 3],
) {
    incant!(
        image_multiply_inner(img1, img2, out),
        [v3, neon, wasm128, scalar]
    )
}
