#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::arcane;
/// SIMD-optimized operations for SSIMULACRA2 computation
///
/// Uses archmage/magetypes for cross-platform SIMD with runtime dispatch.
/// On x86_64: AVX2+FMA via f32x8. On aarch64/wasm32: NEON/SIMD128 via f32x4.
use archmage::incant;
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
use magetypes::simd::f32x4;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

const C2: f32 = 0.0009f32;
#[cfg(target_arch = "x86_64")]
const LANES: usize = 8;

// =============================================================================
// SSIM map
// =============================================================================

/// AVX2 SSIM map computation — processes 8 pixels at a time.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_map_inner_v3(
    token: archmage::X64V3Token,
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    let c2_simd = f32x8::splat(token, C2);
    let one_simd = f32x8::splat(token, 1.0);
    let two_simd = f32x8::splat(token, 2.0);
    let zero_simd = f32x8::zero(token);

    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 2];

    for c in 0..3 {
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

            let num_m = mu_diff.mul_add(-mu_diff, one_simd);
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

            let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
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

/// Scalar fallback for SSIM map computation.
fn ssim_map_inner_scalar(
    _token: archmage::ScalarToken,
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 2];

    for c in 0..3 {
        let mut sum_d = 0.0f64;
        let mut sum_d4 = 0.0f64;

        let m1c = &m1[c];
        let m2c = &m2[c];
        let s11c = &s11[c];
        let s22c = &s22[c];
        let s12c = &s12[c];

        for x in 0..m1c.len() {
            let mu1 = m1c[x];
            let mu2 = m2c[x];
            let mu_diff = mu1 - mu2;

            let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
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

/// 128-bit SIMD SSIM map body — shared between NEON and WASM SIMD128 variants.
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
macro_rules! ssim_map_128_body {
    ($token:ident, $width:ident, $height:ident, $m1:ident, $m2:ident,
     $s11:ident, $s22:ident, $s12:ident) => {{
        const LANES: usize = 4;
        let c2_simd = f32x4::splat($token, C2);
        let one_simd = f32x4::splat($token, 1.0);
        let two_simd = f32x4::splat($token, 2.0);
        let zero_simd = f32x4::zero($token);

        let one_per_pixels = 1.0f64 / ($width * $height) as f64;
        let mut plane_averages = [0f64; 3 * 2];

        for c in 0..3 {
            let mut sum_d = 0.0f64;
            let mut sum_d4 = 0.0f64;

            let m1c = &$m1[c];
            let m2c = &$m2[c];
            let s11c = &$s11[c];
            let s22c = &$s22[c];
            let s12c = &$s12[c];

            let total = m1c.len();
            let chunks = total / LANES;

            for chunk in 0..chunks {
                let base = chunk * LANES;

                let mu1 = f32x4::from_array($token, m1c[base..][..LANES].try_into().unwrap());
                let mu2 = f32x4::from_array($token, m2c[base..][..LANES].try_into().unwrap());
                let s11_vals = f32x4::from_array($token, s11c[base..][..LANES].try_into().unwrap());
                let s22_vals = f32x4::from_array($token, s22c[base..][..LANES].try_into().unwrap());
                let s12_vals = f32x4::from_array($token, s12c[base..][..LANES].try_into().unwrap());

                let mu11 = mu1 * mu1;
                let mu22 = mu2 * mu2;
                let mu12 = mu1 * mu2;
                let mu_diff = mu1 - mu2;

                let num_m = mu_diff.mul_add(-mu_diff, one_simd);
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

                let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
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
    }};
}

/// NEON SSIM map — 4 pixels at a time.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn ssim_map_inner_neon(
    token: archmage::NeonToken,
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    ssim_map_128_body!(token, width, height, m1, m2, s11, s22, s12)
}

/// WASM SIMD128 SSIM map — 4 pixels at a time.
#[cfg(target_arch = "wasm32")]
#[arcane]
fn ssim_map_inner_wasm128(
    token: archmage::Wasm128Token,
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    ssim_map_128_body!(token, width, height, m1, m2, s11, s22, s12)
}

/// SIMD-optimized SSIM map computation with automatic runtime dispatch.
pub(crate) fn ssim_map_simd(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    incant!(
        ssim_map_inner(width, height, m1, m2, s11, s22, s12),
        [v3, neon, wasm128, scalar]
    )
}

// =============================================================================
// Edge difference map
// =============================================================================

/// AVX2 edge difference map — processes 8 pixels at a time.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_map_inner_v3(
    token: archmage::X64V3Token,
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 4];

    let one_simd = f32x8::splat(token, 1.0);
    let zero_simd = f32x8::zero(token);

    for c in 0..3 {
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

            let d1 = (one_simd + diff2) / (one_simd + diff1) - one_simd;

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

        // Scalar remainder
        for x in (chunks * LANES)..total {
            let d1: f64 = (1.0 + f64::from((img2c[x] - mu2c[x]).abs()))
                / (1.0 + f64::from((img1c[x] - mu1c[x]).abs()))
                - 1.0;
            let artifact = d1.max(0.0);
            let detail_lost = (-d1).max(0.0);
            sum_artifact += artifact;
            sum_artifact4 += artifact.powi(4);
            sum_detail += detail_lost;
            sum_detail4 += detail_lost.powi(4);
        }

        plane_averages[c * 4] = one_per_pixels * sum_artifact;
        plane_averages[c * 4 + 1] = (one_per_pixels * sum_artifact4).sqrt().sqrt();
        plane_averages[c * 4 + 2] = one_per_pixels * sum_detail;
        plane_averages[c * 4 + 3] = (one_per_pixels * sum_detail4).sqrt().sqrt();
    }

    plane_averages
}

/// Scalar fallback for edge difference map.
fn edge_diff_map_inner_scalar(
    _token: archmage::ScalarToken,
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 4];

    for c in 0..3 {
        let mut sum_artifact = 0.0f64;
        let mut sum_artifact4 = 0.0f64;
        let mut sum_detail = 0.0f64;
        let mut sum_detail4 = 0.0f64;

        let img1c = &img1[c];
        let mu1c = &mu1[c];
        let img2c = &img2[c];
        let mu2c = &mu2[c];

        for x in 0..img1c.len() {
            let d1: f64 = (1.0 + f64::from((img2c[x] - mu2c[x]).abs()))
                / (1.0 + f64::from((img1c[x] - mu1c[x]).abs()))
                - 1.0;
            let artifact = d1.max(0.0);
            let detail_lost = (-d1).max(0.0);
            sum_artifact += artifact;
            sum_artifact4 += artifact.powi(4);
            sum_detail += detail_lost;
            sum_detail4 += detail_lost.powi(4);
        }

        plane_averages[c * 4] = one_per_pixels * sum_artifact;
        plane_averages[c * 4 + 1] = (one_per_pixels * sum_artifact4).sqrt().sqrt();
        plane_averages[c * 4 + 2] = one_per_pixels * sum_detail;
        plane_averages[c * 4 + 3] = (one_per_pixels * sum_detail4).sqrt().sqrt();
    }

    plane_averages
}

/// 128-bit SIMD edge difference map body — shared between NEON and WASM SIMD128.
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
macro_rules! edge_diff_map_128_body {
    ($token:ident, $width:ident, $height:ident, $img1:ident, $mu1:ident,
     $img2:ident, $mu2:ident) => {{
        const LANES: usize = 4;
        let one_per_pixels = 1.0f64 / ($width * $height) as f64;
        let mut plane_averages = [0f64; 3 * 4];

        let one_simd = f32x4::splat($token, 1.0);
        let zero_simd = f32x4::zero($token);

        for c in 0..3 {
            let mut sum_artifact = 0.0f64;
            let mut sum_artifact4 = 0.0f64;
            let mut sum_detail = 0.0f64;
            let mut sum_detail4 = 0.0f64;

            let img1c = &$img1[c];
            let mu1c = &$mu1[c];
            let img2c = &$img2[c];
            let mu2c = &$mu2[c];

            let total = img1c.len();
            let chunks = total / LANES;

            for chunk in 0..chunks {
                let base = chunk * LANES;

                let r1 = f32x4::from_array($token, img1c[base..][..LANES].try_into().unwrap());
                let rm1 = f32x4::from_array($token, mu1c[base..][..LANES].try_into().unwrap());
                let r2 = f32x4::from_array($token, img2c[base..][..LANES].try_into().unwrap());
                let rm2 = f32x4::from_array($token, mu2c[base..][..LANES].try_into().unwrap());

                let d1_temp = r1 - rm1;
                let diff1 = d1_temp.max(-d1_temp);
                let d2_temp = r2 - rm2;
                let diff2 = d2_temp.max(-d2_temp);

                let d1 = (one_simd + diff2) / (one_simd + diff1) - one_simd;

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

            // Scalar remainder
            for x in (chunks * LANES)..total {
                let d1: f64 = (1.0 + f64::from((img2c[x] - mu2c[x]).abs()))
                    / (1.0 + f64::from((img1c[x] - mu1c[x]).abs()))
                    - 1.0;
                let artifact = d1.max(0.0);
                let detail_lost = (-d1).max(0.0);
                sum_artifact += artifact;
                sum_artifact4 += artifact.powi(4);
                sum_detail += detail_lost;
                sum_detail4 += detail_lost.powi(4);
            }

            plane_averages[c * 4] = one_per_pixels * sum_artifact;
            plane_averages[c * 4 + 1] = (one_per_pixels * sum_artifact4).sqrt().sqrt();
            plane_averages[c * 4 + 2] = one_per_pixels * sum_detail;
            plane_averages[c * 4 + 3] = (one_per_pixels * sum_detail4).sqrt().sqrt();
        }

        plane_averages
    }};
}

/// NEON edge difference map — 4 pixels at a time.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn edge_diff_map_inner_neon(
    token: archmage::NeonToken,
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    edge_diff_map_128_body!(token, width, height, img1, mu1, img2, mu2)
}

/// WASM SIMD128 edge difference map — 4 pixels at a time.
#[cfg(target_arch = "wasm32")]
#[arcane]
fn edge_diff_map_inner_wasm128(
    token: archmage::Wasm128Token,
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    edge_diff_map_128_body!(token, width, height, img1, mu1, img2, mu2)
}

/// SIMD-optimized edge difference map with automatic runtime dispatch.
pub(crate) fn edge_diff_map_simd(
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    incant!(
        edge_diff_map_inner(width, height, img1, mu1, img2, mu2),
        [v3, neon, wasm128, scalar]
    )
}

// =============================================================================
// Image multiplication
// =============================================================================

/// AVX2 image multiplication — processes 8 pixels at a time.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn image_multiply_inner_v3(
    token: archmage::X64V3Token,
    img1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    out: &mut [Vec<f32>; 3],
) {
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

/// Scalar fallback for image multiplication.
fn image_multiply_inner_scalar(
    _token: archmage::ScalarToken,
    img1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    out: &mut [Vec<f32>; 3],
) {
    for c in 0..3 {
        let plane1 = &img1[c];
        let plane2 = &img2[c];
        let out_plane = &mut out[c];

        for i in 0..plane1.len() {
            out_plane[i] = plane1[i] * plane2[i];
        }
    }
}

/// 128-bit SIMD image multiplication body — shared between NEON and WASM SIMD128.
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
macro_rules! image_multiply_128_body {
    ($token:ident, $img1:ident, $img2:ident, $out:ident) => {{
        const LANES: usize = 4;
        for c in 0..3 {
            let plane1 = &$img1[c];
            let plane2 = &$img2[c];
            let out_plane = &mut $out[c];

            let chunks = plane1.len() / LANES;

            for chunk in 0..chunks {
                let base = chunk * LANES;
                let p1 = f32x4::from_array($token, plane1[base..][..LANES].try_into().unwrap());
                let p2 = f32x4::from_array($token, plane2[base..][..LANES].try_into().unwrap());
                let result = p1 * p2;
                out_plane[base..base + LANES].copy_from_slice(&result.to_array());
            }

            // Scalar remainder
            for i in (chunks * LANES)..plane1.len() {
                out_plane[i] = plane1[i] * plane2[i];
            }
        }
    }};
}

/// NEON image multiplication — 4 pixels at a time.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn image_multiply_inner_neon(
    token: archmage::NeonToken,
    img1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    out: &mut [Vec<f32>; 3],
) {
    image_multiply_128_body!(token, img1, img2, out)
}

/// WASM SIMD128 image multiplication — 4 pixels at a time.
#[cfg(target_arch = "wasm32")]
#[arcane]
fn image_multiply_inner_wasm128(
    token: archmage::Wasm128Token,
    img1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    out: &mut [Vec<f32>; 3],
) {
    image_multiply_128_body!(token, img1, img2, out)
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
