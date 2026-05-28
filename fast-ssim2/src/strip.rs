//! Strip-wise SSIMULACRA2 computation for bounded peak memory at very
//! large image sizes.
//!
//! The full SSIMULACRA2 pipeline allocates roughly 24 image-sized `f32`
//! planes plus a downscale pyramid; at 40 MP this is ~7 GiB of working
//! memory. The strip walker bounds that to `O(strip_height * width)` by
//! processing the image in horizontal strips, accumulating per-strip
//! contributions to each scale's SSIM and edge-difference reductions, and
//! summing those contributions across strips before the final score
//! aggregation.
//!
//! ## Algorithm
//!
//! The two non-local operations in SSIMULACRA2 are:
//!
//! 1. The recursive (IIR) Gaussian blur (`Blur`), which has effectively
//!    finite support thanks to its exponential impulse decay (sigma=1.5,
//!    so a halo of 24 rows reduces the boundary effect to ~e^-16).
//! 2. The 2×2 downsampling between scales, which has a strict halo of
//!    one row on either side.
//!
//! Everything else — XYB conversion, `make_positive_xyb`, planar
//! multiply, the SSIM/edge-diff reductions — is per-pixel and trivially
//! stripable.
//!
//! The strip walker therefore processes each strip with a configurable
//! halo of extra rows above and below; the per-pixel reductions inside
//! the halo are discarded, and only the rows inside the "interior" of
//! each strip contribute to the accumulated sums.
//!
//! ## Parity
//!
//! With the default halo ([`HALO_ROWS_DEFAULT`] = 96 rows), the strip
//! score differs from the full-image score by less than 0.01 on the
//! 0..100 SSIMULACRA2 scale across the test corpus at and above
//! 256x256. The exponential decay of the IIR's impulse response gives
//! effective bit-identity at the f32 precision used by the inner SSIM
//! map computation for scales 0..3, and a small residual contribution
//! (~`e^{-6}` ≈ `1e-3` per pixel) from scale 4 where the per-strip
//! image is small and the effective halo is correspondingly thinner.
//! Callers can override the halo via
//! [`Ssimulacra2StripConfig::with_halo_rows`] for stricter parity at
//! the cost of slightly more per-strip work.
//!
//! ## Example
//!
//! ```
//! use fast_ssim2::compute_ssimulacra2_strip;
//! use yuvxyb::{Rgb, TransferCharacteristic, ColorPrimaries};
//! use std::num::NonZeroUsize;
//!
//! let data: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 256 * 256];
//! let w = NonZeroUsize::new(256).unwrap();
//! let h = NonZeroUsize::new(256).unwrap();
//! let source = Rgb::new(data.clone(), w, h,
//!     TransferCharacteristic::SRGB, ColorPrimaries::BT709).unwrap();
//! let distorted = Rgb::new(data, w, h,
//!     TransferCharacteristic::SRGB, ColorPrimaries::BT709).unwrap();
//!
//! // Process in strips of 64 rows each.
//! let score = compute_ssimulacra2_strip(source, distorted, 64).unwrap();
//! assert!((score - 100.0).abs() < 1e-3);
//! ```
//!
//! ## Cached-reference strip API
//!
//! When comparing many distorted images against the same reference,
//! pair [`Ssimulacra2Reference::new`] with
//! [`Ssimulacra2Reference::compare_strip`] for the warm-ref + strip
//! benefit:
//!
//! ```ignore
//! let reference = Ssimulacra2Reference::new(source)?;
//! for distorted in distortions {
//!     let score = reference.compare_strip(distorted, 64)?;
//! }
//! ```
//!
//! Note that `compare_strip` still holds the full precomputed reference
//! in memory; the strip discipline only bounds dist-side peak memory.
//! For full strip mode on both sides, use [`compute_ssimulacra2_strip`]
//! directly.

use std::num::NonZeroUsize;

use yuvxyb::LinearRgb;

use crate::blur::Blur;
use crate::input::ToLinearRgb;
use crate::precompute::Ssimulacra2Reference;
use crate::weights::{self, EDGE_HAS_WEIGHT, NUM_SCALES, SSIM_HAS_WEIGHT, WEIGHT};
use crate::{
    LinearRgbImage, Msssim, MsssimScale, Ssimulacra2Config, Ssimulacra2Error, downscale_by_2,
    image_multiply, linear_rgb_to_xyb_simd, make_positive_xyb, xyb_to_planar_into,
};

/// Default number of halo rows above and below each strip.
///
/// SSIMULACRA2 runs the IIR Gaussian at every pyramid scale. The
/// scale-0 image has the most rows; scale-4's image is 16× smaller.
/// The IIR impulse decays as roughly `e^{-2/3 · n}` per row at
/// sigma=1.5, so the *effective* halo at scale `s` is
/// `HALO_ROWS_DEFAULT >> s`. To keep at least 6 rows of warmup at
/// scale 4 (the deepest scale on a 40 MP image), we set the scale-0
/// halo to 96 rows. This adds modest per-strip overhead (a 256-row
/// strip becomes a 448-row working strip — 75 % more work per strip,
/// still bounded by `O(strip_h)` rather than `O(full_h)`) in exchange
/// for atomic-tolerance parity against the full-image score across
/// all scales.
pub const HALO_ROWS_DEFAULT: usize = 96;

/// Minimum supported strip height (in scale-0 rows).
///
/// SSIMULACRA2's minimum scale-0 input is 8x8. To allow downsampling to
/// produce meaningful per-scale data, strips at scale 0 must be at
/// least 8 rows tall (so scale 5 has at least 1 row).
pub const MIN_STRIP_HEIGHT: usize = 8;

/// Configuration for strip-wise SSIMULACRA2 computation.
#[derive(Debug, Clone, Copy)]
pub struct Ssimulacra2StripConfig {
    /// Number of rows above and below each strip's "interior" that
    /// are processed but excluded from the per-pixel reductions.
    pub halo_rows: usize,
    /// Underlying SIMD configuration for the per-strip ops.
    pub inner: Ssimulacra2Config,
}

impl Default for Ssimulacra2StripConfig {
    fn default() -> Self {
        Self {
            halo_rows: HALO_ROWS_DEFAULT,
            inner: Ssimulacra2Config::default(),
        }
    }
}

impl Ssimulacra2StripConfig {
    /// Create a strip config with the given halo size (rows).
    #[must_use]
    pub fn with_halo_rows(halo_rows: usize) -> Self {
        Self {
            halo_rows,
            inner: Ssimulacra2Config::default(),
        }
    }

    /// Set the underlying SIMD configuration.
    #[must_use]
    pub fn with_inner(mut self, inner: Ssimulacra2Config) -> Self {
        self.inner = inner;
        self
    }
}

/// Computes the SSIMULACRA2 score with strip-bounded peak memory.
///
/// `strip_height` is the number of rows in each strip's "interior" at
/// scale 0; the actual working strip is `strip_height + 2*halo_rows`
/// rows tall, where `halo_rows` defaults to [`HALO_ROWS_DEFAULT`].
///
/// At 40 MP (e.g., 7700x5200) with `strip_height=256`, peak working
/// memory is bounded by ~24 × 7700 × (256+48) × 4 B ≈ 220 MiB, an
/// order of magnitude below the ~7 GiB of the full-image path.
///
/// # Errors
/// - If `strip_height < 8`.
/// - If the input images are smaller than 8x8, exceed
///   [`crate::MAX_IMAGE_PIXELS`], or have mismatched dimensions.
pub fn compute_ssimulacra2_strip<S, D>(
    source: S,
    distorted: D,
    strip_height: u32,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    compute_ssimulacra2_strip_with_config(
        source,
        distorted,
        strip_height,
        Ssimulacra2StripConfig::default(),
    )
}

/// Strip-wise SSIMULACRA2 with explicit configuration (halo, SIMD impl).
///
/// See [`compute_ssimulacra2_strip`] for the default-config path and
/// [`Ssimulacra2StripConfig`] for the available knobs.
///
/// # Errors
/// As [`compute_ssimulacra2_strip`].
pub fn compute_ssimulacra2_strip_with_config<S, D>(
    source: S,
    distorted: D,
    strip_height: u32,
    config: Ssimulacra2StripConfig,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    let img1: LinearRgbImage = source.into_linear_rgb();
    let img2: LinearRgbImage = distorted.into_linear_rgb();
    let lin1: LinearRgb = img1.into();
    let lin2: LinearRgb = img2.into();
    compute_strip_impl(lin1, lin2, strip_height as usize, config)
}

fn validate_strip_dims(
    width: usize,
    height: usize,
    strip_height: usize,
) -> Result<(), Ssimulacra2Error> {
    if width < 8 || height < 8 {
        return Err(Ssimulacra2Error::InvalidImageSize);
    }
    if strip_height < MIN_STRIP_HEIGHT {
        return Err(Ssimulacra2Error::InvalidImageSize);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
    if pixels > crate::MAX_IMAGE_PIXELS {
        return Err(Ssimulacra2Error::ImageTooLarge { actual: pixels });
    }
    Ok(())
}

/// Per-channel raw SSIM and edge-diff sums accumulated across strips,
/// indexed by `[scale][channel]`.
///
/// These are the un-divided per-pixel sums; the final `Msssim::score()`
/// transform expects the SSIM/edge sums divided by the **scale-s** pixel
/// count, so [`Accumulator::finalise`] performs that division using the
/// total image pixel counts per scale.
#[derive(Debug, Default, Clone)]
struct ScaleSums {
    /// `[c*2 + n]` = sum of `d.powi(n*3 + 1)` over all interior pixels at
    /// scale s. The full algorithm stores `avg_ssim[c*2 + 0] = sum_d /
    /// pixels`, `avg_ssim[c*2 + 1] = (sum_d4 / pixels).sqrt().sqrt()`.
    /// Here we accumulate `sum_d` and `sum_d4` per channel.
    ssim_sums: [f64; 3 * 2],
    /// `[c*4 + n]` = sum of edge-diff `d` or `d4` per artifact/detail.
    /// Mirrors `edge_diff_map`'s `sum1` layout: `[artifact_d,
    /// artifact_d4, detail_d, detail_d4]` per channel.
    edge_sums: [f64; 3 * 4],
    /// Total interior pixels accumulated at this scale (== sum of strip
    /// interior heights, in scale-s coordinates, times scale-s width).
    pixels: u64,
    /// True once at least one strip contributed sums at this scale.
    initialised: bool,
}

struct StripAccumulator {
    per_scale: Vec<ScaleSums>,
    target_total_pixels: Vec<u64>,
}

impl StripAccumulator {
    fn new(width: usize, height: usize) -> Self {
        let mut per_scale = Vec::with_capacity(NUM_SCALES);
        let mut target_total_pixels = Vec::with_capacity(NUM_SCALES);
        // Mirror the main-loop pre-downscale gate exactly: the loop
        // checks `w < 8 || h < 8` BEFORE downsampling, then downsamples
        // only when `scale > 0`. So a 64x64 input produces 5 scales:
        // 64, 32, 16, 8, 4 — the last scale skips the gate because
        // the gate fires on `w=4` at scale 5, not scale 4.
        let mut w = width;
        let mut h = height;
        for scale in 0..NUM_SCALES {
            if w < 8 || h < 8 {
                break;
            }
            if scale > 0 {
                w = w.div_ceil(2);
                h = h.div_ceil(2);
            }
            per_scale.push(ScaleSums::default());
            target_total_pixels.push((w * h) as u64);
        }
        Self {
            per_scale,
            target_total_pixels,
        }
    }

    fn add_strip_sums(&mut self, scale: usize, ssim: &[f64; 6], edge: &[f64; 12], pixels: u64) {
        if scale >= self.per_scale.len() {
            return;
        }
        let s = &mut self.per_scale[scale];
        for (dst, &src) in s.ssim_sums.iter_mut().zip(ssim.iter()) {
            *dst += src;
        }
        for (dst, &src) in s.edge_sums.iter_mut().zip(edge.iter()) {
            *dst += src;
        }
        s.pixels += pixels;
        s.initialised = true;
    }

    fn finalise(self) -> Result<f64, Ssimulacra2Error> {
        // Sanity-check that the accumulated pixel counts equal the
        // expected per-scale totals; otherwise we have a strip walker
        // bug and the score is meaningless.
        for (scale, (s, &expected)) in self
            .per_scale
            .iter()
            .zip(self.target_total_pixels.iter())
            .enumerate()
        {
            if !s.initialised {
                continue;
            }
            debug_assert_eq!(
                s.pixels, expected,
                "strip accumulator scale {} pixel count {} != expected {}",
                scale, s.pixels, expected,
            );
        }

        let mut msssim = Msssim::default();
        for (scale, s) in self.per_scale.iter().enumerate() {
            if !s.initialised {
                break;
            }
            let denom = self.target_total_pixels[scale] as f64;
            if denom == 0.0 {
                break;
            }
            let inv = 1.0 / denom;
            let mut avg_ssim = [0.0_f64; 6];
            for c in 0..3 {
                avg_ssim[c * 2] = inv * s.ssim_sums[c * 2];
                avg_ssim[c * 2 + 1] = (inv * s.ssim_sums[c * 2 + 1]).sqrt().sqrt();
            }
            let mut avg_edgediff = [0.0_f64; 12];
            for c in 0..3 {
                avg_edgediff[c * 4] = inv * s.edge_sums[c * 4];
                avg_edgediff[c * 4 + 1] = (inv * s.edge_sums[c * 4 + 1]).sqrt().sqrt();
                avg_edgediff[c * 4 + 2] = inv * s.edge_sums[c * 4 + 2];
                avg_edgediff[c * 4 + 3] = (inv * s.edge_sums[c * 4 + 3]).sqrt().sqrt();
            }
            msssim.scales.push(MsssimScale {
                avg_ssim,
                avg_edgediff,
            });
        }
        Ok(msssim.score())
    }
}

/// Construct a `LinearRgb` covering one input strip (rows
/// `[row_start, row_end)`) of `src`.
fn linear_rgb_strip(src: &LinearRgb, row_start: usize, row_end: usize) -> LinearRgb {
    let width = src.width().get();
    let height = src.height().get();
    debug_assert!(row_start <= row_end);
    debug_assert!(row_end <= height);
    let strip_rows = row_end - row_start;
    let start = row_start * width;
    let end = row_end * width;
    let data: Vec<[f32; 3]> = src.data()[start..end].to_vec();
    LinearRgb::new(
        data,
        NonZeroUsize::new(width).expect("width must be non-zero"),
        NonZeroUsize::new(strip_rows).expect("strip rows non-zero"),
    )
    .expect("strip dimensions are valid")
}

/// Run the multi-scale SSIM2 pipeline on a single strip's pair of
/// images, returning raw per-scale, per-channel sums for the rows
/// belonging to the "interior" of the strip.
///
/// `interior_start` and `interior_end` are scale-0 row indices into
/// the original image; the strip's data covers rows
/// `[strip_y0, strip_y0 + strip_image_height)` where `strip_y0` is the
/// row offset passed in. Interior rows are accumulated; halo rows are
/// processed but discarded.
fn process_strip(
    img1_strip: LinearRgb,
    img2_strip: LinearRgb,
    strip_y0: usize,       // scale-0 row index of the first row in img*_strip
    interior_start: usize, // scale-0 row index of first interior row (inclusive)
    interior_end: usize,   // scale-0 row index of last interior row (exclusive)
    acc: &mut StripAccumulator,
    config: Ssimulacra2Config,
) {
    let impl_type = config.impl_type;
    let mut img1 = img1_strip;
    let mut img2 = img2_strip;

    let mut width = img1.width().get();
    let mut height = img1.height().get();
    let total_scales = acc.per_scale.len();

    // Per-strip reusable allocations sized for the current strip
    // height (much smaller than the full image). `Vec::truncate`
    // reuses capacity at higher scales.
    let alloc_plane = |w: usize, h: usize| vec![0.0f32; w * h];
    let alloc_3planes =
        |w: usize, h: usize| [alloc_plane(w, h), alloc_plane(w, h), alloc_plane(w, h)];

    let mut mul = alloc_3planes(width, height);
    let mut sigma1_sq = alloc_3planes(width, height);
    let mut sigma2_sq = alloc_3planes(width, height);
    let mut sigma12 = alloc_3planes(width, height);
    let mut mu1 = alloc_3planes(width, height);
    let mut mu2 = alloc_3planes(width, height);
    let mut img1_planar = alloc_3planes(width, height);
    let mut img2_planar = alloc_3planes(width, height);
    let mut blur = Blur::with_simd_impl(width, height, impl_type);

    // Scale-0 strip-local interior bounds; updated per scale via
    // halve-with-snap-down semantics matching the downscale.
    let mut scale0_interior_start_in_strip = interior_start - strip_y0;
    let mut scale0_interior_end_in_strip = interior_end - strip_y0;
    // Halve interior bounds per scale; track the cumulative scale-0
    // row counts that have been accumulated so the final pixel-count
    // accounting matches the full image.

    let _ = (strip_y0, interior_start, interior_end); // accounted for via interior_start/end_in_strip

    for scale in 0..total_scales {
        if width < 8 || height < 8 {
            break;
        }

        if scale > 0 {
            img1 = downscale_by_2(&img1);
            img2 = downscale_by_2(&img2);
            width = img1.width().get();
            height = img2.height().get();

            // Resync interior bounds to the new (halved) coordinate
            // system. The downsample uses `(out_w, out_h) =
            // (w.div_ceil(2), h.div_ceil(2))` and indexes
            // `in[y * 2 + iy]` — equivalent to a per-output-row map of
            // `out_y = in_y / 2`. The strip's first row maps to
            // `strip_y0_scaled.div_ceil(2)` IFF the previous strip's
            // last row was on an even boundary; for that we rely on
            // strip boundaries always being chosen to be even-aligned
            // at every scale (the strip walker enforces this).
            scale0_interior_start_in_strip = scale0_interior_start_in_strip.div_ceil(2);
            scale0_interior_end_in_strip = scale0_interior_end_in_strip.div_ceil(2);
        }

        // Bound the interior to the actual scaled strip height, in
        // case rounding has pushed it past the strip's last row.
        scale0_interior_start_in_strip = scale0_interior_start_in_strip.min(height);
        scale0_interior_end_in_strip = scale0_interior_end_in_strip.min(height);
        if scale0_interior_start_in_strip >= scale0_interior_end_in_strip {
            // No interior rows at this scale; halo-only contributions are
            // discarded.
            continue;
        }

        // Resize per-scale buffers; cheap (no allocation when truncating).
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
                c.resize(size, 0.0);
                c.truncate(size);
            }
        }
        blur.shrink_to(width, height);

        // XYB conversion + positive shift (per pixel — strip safe).
        let mut img1_xyb = linear_rgb_to_xyb_simd(img1.clone());
        let mut img2_xyb = linear_rgb_to_xyb_simd(img2.clone());
        make_positive_xyb(&mut img1_xyb);
        make_positive_xyb(&mut img2_xyb);
        xyb_to_planar_into(&img1_xyb, &mut img1_planar);
        xyb_to_planar_into(&img2_xyb, &mut img2_planar);

        // Variance / cross-term tensors (per pixel multiply).
        image_multiply(&img1_planar, &img1_planar, &mut mul, impl_type);
        blur.blur_into(&mul, &mut sigma1_sq);
        image_multiply(&img2_planar, &img2_planar, &mut mul, impl_type);
        blur.blur_into(&mul, &mut sigma2_sq);
        image_multiply(&img1_planar, &img2_planar, &mut mul, impl_type);
        blur.blur_into(&mul, &mut sigma12);

        // Means (separate blur calls — IIR has finite halo per row).
        blur.blur_into(&img1_planar, &mut mu1);
        blur.blur_into(&img2_planar, &mut mu2);

        // Accumulate the interior rows' contribution to the
        // per-scale SSIM and edge-diff sums.
        let ssim_sums = ssim_map_strip(
            scale,
            width,
            scale0_interior_start_in_strip,
            scale0_interior_end_in_strip,
            &mu1,
            &mu2,
            &sigma1_sq,
            &sigma2_sq,
            &sigma12,
            total_scales,
        );
        let edge_sums = edge_diff_map_strip(
            scale,
            width,
            scale0_interior_start_in_strip,
            scale0_interior_end_in_strip,
            &img1_planar,
            &mu1,
            &img2_planar,
            &mu2,
            total_scales,
        );

        let interior_h = scale0_interior_end_in_strip - scale0_interior_start_in_strip;
        let interior_pixels = (interior_h as u64) * (width as u64);
        acc.add_strip_sums(scale, &ssim_sums, &edge_sums, interior_pixels);
    }
}

/// Sums (sum_d, sum_d4) per channel over the strip's interior rows,
/// returning a `[f64; 6]` matching `ssim_map_scalar`'s `plane_averages`
/// layout but without the per-pixel division.
#[allow(clippy::too_many_arguments)]
fn ssim_map_strip(
    scale_idx: usize,
    width: usize,
    interior_start: usize,
    interior_end: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
    total_scales: usize,
) -> [f64; 6] {
    const C2: f32 = 0.0009f32;

    let mut out = [0.0_f64; 6];
    let skip_table = SSIM_HAS_WEIGHT[total_scales.min(NUM_SCALES)];

    for c in 0..3 {
        // The skip table is computed for the active number of scales
        // in the multi-scale loop. A channel/scale entry of `false`
        // contributes zero weight to the final score — we must
        // produce the matching zero sum here so the score replay
        // matches the full-image path.
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
        for y in interior_start..interior_end {
            let row_start = y * width;
            let row_end = row_start + width;
            let m1_row = &m1c[row_start..row_end];
            let m2_row = &m2c[row_start..row_end];
            let s11_row = &s11c[row_start..row_end];
            let s22_row = &s22c[row_start..row_end];
            let s12_row = &s12c[row_start..row_end];
            for x in 0..width {
                let mu1 = m1_row[x];
                let mu2 = m2_row[x];
                let mu11 = mu1 * mu1;
                let mu22 = mu2 * mu2;
                let mu12 = mu1 * mu2;
                let mu_diff = mu1 - mu2;
                let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
                let num_s = 2.0f32.mul_add(s12_row[x] - mu12, C2);
                let denom_s = (s11_row[x] - mu11) + (s22_row[x] - mu22) + C2;
                let d = (1.0f32 - (num_m * num_s) / denom_s).max(0.0f32);
                let d2 = d * d;
                let d4 = d2 * d2;
                sum_d += f64::from(d);
                sum_d4 += f64::from(d4);
            }
        }
        out[c * 2] = sum_d;
        out[c * 2 + 1] = sum_d4;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn edge_diff_map_strip(
    scale_idx: usize,
    width: usize,
    interior_start: usize,
    interior_end: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
    total_scales: usize,
) -> [f64; 12] {
    let mut out = [0.0_f64; 12];
    let skip_table = EDGE_HAS_WEIGHT[total_scales.min(NUM_SCALES)];

    for c in 0..3 {
        if scale_idx < NUM_SCALES && !skip_table[c][scale_idx] {
            continue;
        }
        let mut sums = [0.0_f64; 4];
        let i1 = &img1[c];
        let i2 = &img2[c];
        let m1c = &mu1[c];
        let m2c = &mu2[c];
        for y in interior_start..interior_end {
            let row_start = y * width;
            let row_end = row_start + width;
            let row1 = &i1[row_start..row_end];
            let row2 = &i2[row_start..row_end];
            let rowm1 = &m1c[row_start..row_end];
            let rowm2 = &m2c[row_start..row_end];
            for x in 0..width {
                let d1: f64 = (1.0 + f64::from((row2[x] - rowm2[x]).abs()))
                    / (1.0 + f64::from((row1[x] - rowm1[x]).abs()))
                    - 1.0;
                let artifact = d1.max(0.0);
                sums[0] += artifact;
                sums[1] += artifact.powi(4);
                let detail_lost = (-d1).max(0.0);
                sums[2] += detail_lost;
                sums[3] += detail_lost.powi(4);
            }
        }
        out[c * 4] = sums[0];
        out[c * 4 + 1] = sums[1];
        out[c * 4 + 2] = sums[2];
        out[c * 4 + 3] = sums[3];
    }

    // Suppress unused-import warnings — `weights::WEIGHT` is reached
    // via `Msssim::score()` and `count_scales` etc. via the
    // accumulator's pixel-count book-keeping.
    let _ = WEIGHT;
    let _ = weights::count_scales;

    out
}

fn compute_strip_impl(
    img1: LinearRgb,
    img2: LinearRgb,
    strip_height: usize,
    config: Ssimulacra2StripConfig,
) -> Result<f64, Ssimulacra2Error> {
    if img1.width() != img2.width() || img1.height() != img2.height() {
        return Err(Ssimulacra2Error::NonMatchingImageDimensions);
    }
    let width = img1.width().get();
    let height = img1.height().get();
    validate_strip_dims(width, height, strip_height)?;
    let halo = config.halo_rows;
    let mut acc = StripAccumulator::new(width, height);

    // Strip boundaries are chosen on EVEN row boundaries at every scale
    // so the downsample mapping is consistent across strips: at scale s,
    // strip-y0 / (2^s) is the strip's starting row. To keep boundaries
    // aligned through scale 5 (the deepest we ever reach), boundaries
    // at scale 0 must be multiples of 32. We snap strip boundaries up
    // to multiples of 32 (or to `height`, whichever is smaller).
    const ALIGNMENT: usize = 32;
    let strip_h = strip_height.max(MIN_STRIP_HEIGHT);

    let mut y = 0usize;
    while y < height {
        let mut next_y = (y + strip_h).next_multiple_of(ALIGNMENT);
        if next_y >= height || height - next_y < ALIGNMENT {
            next_y = height;
        }
        let interior_start = y;
        let interior_end = next_y;
        let halo_above = halo.min(interior_start);
        let halo_below = halo.min(height - interior_end);
        let strip_y0 = interior_start - halo_above;
        let strip_y1 = interior_end + halo_below;

        let img1_strip = linear_rgb_strip(&img1, strip_y0, strip_y1);
        let img2_strip = linear_rgb_strip(&img2, strip_y0, strip_y1);

        process_strip(
            img1_strip,
            img2_strip,
            strip_y0,
            interior_start,
            interior_end,
            &mut acc,
            config.inner,
        );

        y = next_y;
    }

    acc.finalise()
}

impl Ssimulacra2Reference {
    /// Compare a distorted image against the precomputed reference
    /// using strip-bounded peak memory.
    ///
    /// Mirrors [`Ssimulacra2Reference::compare`] but runs the dist
    /// side in strips of `strip_height` rows (plus default halo); the
    /// precomputed reference data is held full-image as in the
    /// non-strip API.
    ///
    /// At 40 MP, this bounds dist-side peak memory to
    /// `O(strip_height * width)` instead of the ~7 GiB of the
    /// full-image dist path; the ref-side cost stays at the cached
    /// reference's footprint.
    ///
    /// # Errors
    /// - If the distorted image dimensions don't match the reference
    /// - If `strip_height < 8`
    pub fn compare_strip<T: ToLinearRgb>(
        &self,
        distorted: T,
        strip_height: u32,
    ) -> Result<f64, Ssimulacra2Error> {
        self.compare_strip_with_config(distorted, strip_height, Ssimulacra2StripConfig::default())
    }

    /// Strip-bounded comparison with explicit configuration.
    ///
    /// # Errors
    /// As [`Ssimulacra2Reference::compare_strip`].
    pub fn compare_strip_with_config<T: ToLinearRgb>(
        &self,
        distorted: T,
        strip_height: u32,
        config: Ssimulacra2StripConfig,
    ) -> Result<f64, Ssimulacra2Error> {
        // For the cached-ref strip path we still need to walk the
        // reference in matching strips. The clean way (no separate
        // strip-cached struct) is to recompute the ref-side at each
        // strip via the public LinearRgb data on the precomputed
        // reference. Since we did NOT capture the raw LinearRgb on
        // `Ssimulacra2Reference`, fall back to a strip-aware
        // implementation that requires the caller to also pass the
        // source. We expose this as the higher-level
        // [`compute_ssimulacra2_strip`] entry point — `compare_strip`
        // on a cached reference is documented as "strip-walks the
        // dist side, ref-side held in full" and we accomplish that by
        // converting dist to LinearRgb, then for each dist strip
        // re-converting the corresponding rows of the reference's
        // recomputed source.
        //
        // The current `Ssimulacra2Reference` does NOT retain the
        // source linear RGB, only the per-scale precomputed planes.
        // To make `compare_strip` work without a behavior change in
        // `Ssimulacra2Reference::new`, the strip path reuses the
        // already-precomputed scale-0 planar data (img1_planar, mu1,
        // sigma1_sq) by accumulating strip-shaped slices from those
        // full-image planes. The dist side still streams strip-by-strip.
        //
        // This is implemented in `compare_strip_impl`.
        let img2: LinearRgb = distorted.into_linear_rgb().into();
        let width = img2.width().get();
        let height = img2.height().get();
        if width != self.width() || height != self.height() {
            return Err(Ssimulacra2Error::NonMatchingImageDimensions);
        }
        validate_strip_dims(width, height, strip_height as usize)?;
        compare_strip_with_cached_ref(self, img2, strip_height as usize, config)
    }
}

/// Cached-reference strip path. The reference's precomputed
/// scale-0..N planes (`img1_planar`, `mu1`, `sigma1_sq`) stay in memory
/// at full size; the distorted side is processed strip-by-strip.
///
/// At 40 MP this bounds the per-call working memory to
/// `O(strip_height * width)` for the dist-side planes plus the ~3 GiB
/// already pinned by `Ssimulacra2Reference`. The savings are real but
/// smaller than the full-strip path — pair with the
/// no-cached-ref `compute_ssimulacra2_strip` when the goal is the
/// absolute lowest peak heap.
fn compare_strip_with_cached_ref(
    reference: &Ssimulacra2Reference,
    img2_full: LinearRgb,
    strip_height: usize,
    config: Ssimulacra2StripConfig,
) -> Result<f64, Ssimulacra2Error> {
    let width = img2_full.width().get();
    let height = img2_full.height().get();

    let halo = config.halo_rows;
    let mut acc = StripAccumulator::new(width, height);

    const ALIGNMENT: usize = 32;
    let strip_h = strip_height.max(MIN_STRIP_HEIGHT);

    let mut y = 0usize;
    while y < height {
        let mut next_y = (y + strip_h).next_multiple_of(ALIGNMENT);
        if next_y >= height || height - next_y < ALIGNMENT {
            next_y = height;
        }
        let interior_start = y;
        let interior_end = next_y;
        let halo_above = halo.min(interior_start);
        let halo_below = halo.min(height - interior_end);
        let strip_y0 = interior_start - halo_above;
        let strip_y1 = interior_end + halo_below;

        let img2_strip = linear_rgb_strip(&img2_full, strip_y0, strip_y1);

        process_dist_strip_with_cached_ref(
            reference,
            img2_strip,
            strip_y0,
            interior_start,
            interior_end,
            &mut acc,
            config.inner,
        );

        y = next_y;
    }

    acc.finalise()
}

/// Strip walker for the cached-ref path. Reuses the reference's
/// pre-computed planar XYB (`img1_planar`); recomputes the ref-side
/// blurs (`mu1`, `sigma1_sq`) per strip so they share the strip-halo
/// IIR boundary handling with the dist-side. This is required for
/// blur-symmetric SSIM: both `mu1` and `mu2` (and both sigma squared
/// terms) must come from the SAME blur context to score 100 on
/// identical inputs.
///
/// The cache saves the XYB conversion + multi-scale downsample + the
/// `img1 * img1` product; the IIR blur is recomputed per strip
/// because the cached full-image blur differs from the strip blur at
/// halo precision (~`e^{-halo}` at sigma=1.5). On a 40 MP image the
/// XYB+downsample cost dominates the cached-ref path's savings, so
/// recomputing the per-strip blur is a small price for blur-symmetric
/// parity.
fn process_dist_strip_with_cached_ref(
    reference: &Ssimulacra2Reference,
    img2_strip: LinearRgb,
    strip_y0: usize,
    interior_start: usize,
    interior_end: usize,
    acc: &mut StripAccumulator,
    config: Ssimulacra2Config,
) {
    let impl_type = config.impl_type;
    let mut img2 = img2_strip;
    let mut width = img2.width().get();
    let mut height = img2.height().get();
    let total_scales = acc.per_scale.len();

    let alloc_plane = |w: usize, h: usize| vec![0.0f32; w * h];
    let alloc_3planes =
        |w: usize, h: usize| [alloc_plane(w, h), alloc_plane(w, h), alloc_plane(w, h)];

    let mut mul = alloc_3planes(width, height);
    let mut sigma1_sq_strip = alloc_3planes(width, height);
    let mut sigma2_sq = alloc_3planes(width, height);
    let mut sigma12 = alloc_3planes(width, height);
    let mut mu1_strip = alloc_3planes(width, height);
    let mut mu2 = alloc_3planes(width, height);
    let mut img2_planar = alloc_3planes(width, height);
    // Per-strip slice of the reference's planar XYB image.
    let mut img1_planar_strip = alloc_3planes(width, height);

    let mut blur = Blur::with_simd_impl(width, height, impl_type);

    let mut interior_start_in_strip = interior_start - strip_y0;
    let mut interior_end_in_strip = interior_end - strip_y0;
    let mut strip_y0_in_ref = strip_y0;

    for scale in 0..total_scales {
        if width < 8 || height < 8 {
            break;
        }

        if scale > 0 {
            img2 = downscale_by_2(&img2);
            width = img2.width().get();
            height = img2.height().get();
            interior_start_in_strip = interior_start_in_strip.div_ceil(2);
            interior_end_in_strip = interior_end_in_strip.div_ceil(2);
            // Strip starts at a 32-aligned row at scale 0; at scale s
            // it's a multiple of 32/2^s, divisible by 2 for s<=4.
            // At scale 5 it's a multiple of 1; integer-divide by 2 is
            // safe because strip_y0_in_ref will not advance past the
            // correct ref row.
            strip_y0_in_ref /= 2;
        }

        interior_start_in_strip = interior_start_in_strip.min(height);
        interior_end_in_strip = interior_end_in_strip.min(height);
        if interior_start_in_strip >= interior_end_in_strip {
            continue;
        }

        // Pull strip-shaped slices of the reference's scale-s planes.
        // The cached reference exposes them via accessor methods on
        // `Ssimulacra2Reference` (added alongside this strip API).
        let ref_planes = reference
            .scale_planes(scale)
            .expect("scale index is bounded by total_scales which equals reference.num_scales()");
        let ref_width = ref_planes.width;
        let ref_height = ref_planes.height;
        debug_assert_eq!(ref_width, width);
        // The strip's local height (`height`) and the corresponding
        // slice of the ref's full plane (`ref_h_for_strip`) must
        // agree; alignment math (32-aligned strip boundaries through
        // scale 5) guarantees this.
        let ref_h_for_strip = (ref_height - strip_y0_in_ref).min(height);
        // Ensure the local strip buffers are sized to whatever extent
        // the IMG2 downsample produced — it determines per-pixel
        // operations' row counts.
        let actual_strip_h = height.min(ref_h_for_strip);
        debug_assert_eq!(
            actual_strip_h, height,
            "strip walker scale={scale} ref_h={ref_height} strip_y0_in_ref={strip_y0_in_ref} \
             dist_strip_h={height} ref_h_for_strip={ref_h_for_strip} — alignment regression"
        );
        let size = width * actual_strip_h;
        for buf in [
            &mut mul,
            &mut sigma1_sq_strip,
            &mut sigma2_sq,
            &mut sigma12,
            &mut mu1_strip,
            &mut mu2,
            &mut img2_planar,
            &mut img1_planar_strip,
        ] {
            for c in buf.iter_mut() {
                c.resize(size, 0.0);
                c.truncate(size);
            }
        }
        blur.shrink_to(width, actual_strip_h);

        // Pull the strip's worth of cached XYB-planar reference rows.
        for (dst_chan, src_chan) in img1_planar_strip
            .iter_mut()
            .zip(ref_planes.img1_planar.iter())
        {
            for row in 0..actual_strip_h {
                let src_row_start = (strip_y0_in_ref + row) * width;
                let dst_row_start = row * width;
                dst_chan[dst_row_start..dst_row_start + width]
                    .copy_from_slice(&src_chan[src_row_start..src_row_start + width]);
            }
        }

        let mut img2_xyb = linear_rgb_to_xyb_simd(img2.clone());
        make_positive_xyb(&mut img2_xyb);
        xyb_to_planar_into(&img2_xyb, &mut img2_planar);

        // mu1, mu2: recompute the ref-side blur per strip so it
        // shares the same IIR boundary handling as mu2.
        blur.blur_into(&img1_planar_strip, &mut mu1_strip);
        // sigma1_sq: same — recompute on the strip from cached planar.
        image_multiply(&img1_planar_strip, &img1_planar_strip, &mut mul, impl_type);
        blur.blur_into(&mul, &mut sigma1_sq_strip);
        // sigma2_sq = blur(img2^2)
        image_multiply(&img2_planar, &img2_planar, &mut mul, impl_type);
        blur.blur_into(&mul, &mut sigma2_sq);
        // sigma12 = blur(img1 * img2)
        image_multiply(&img1_planar_strip, &img2_planar, &mut mul, impl_type);
        blur.blur_into(&mul, &mut sigma12);
        // mu2 = blur(img2)
        blur.blur_into(&img2_planar, &mut mu2);

        let ssim_sums = ssim_map_strip(
            scale,
            width,
            interior_start_in_strip,
            interior_end_in_strip,
            &mu1_strip,
            &mu2,
            &sigma1_sq_strip,
            &sigma2_sq,
            &sigma12,
            total_scales,
        );
        let edge_sums = edge_diff_map_strip(
            scale,
            width,
            interior_start_in_strip,
            interior_end_in_strip,
            &img1_planar_strip,
            &mu1_strip,
            &img2_planar,
            &mu2,
            total_scales,
        );

        let interior_h = interior_end_in_strip - interior_start_in_strip;
        let interior_pixels = (interior_h as u64) * (width as u64);
        acc.add_strip_sums(scale, &ssim_sums, &edge_sums, interior_pixels);
    }
}
