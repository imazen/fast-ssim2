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

use crate::input::ToLinearRgb;
use crate::pipeline::{Kernel, Opts};
use crate::precompute::Ssimulacra2Reference;
use crate::{LinearRgbImage, Ssimulacra2Config, Ssimulacra2Error};

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
/// SSIMULACRA2's minimum scale-0 input is 8×8; a strip interior below
/// 8 rows would degenerate the per-scale halo accounting.
pub const MIN_STRIP_HEIGHT: usize = 8;

/// Configuration for strip-wise SSIMULACRA2 computation.
#[derive(Debug, Clone, Copy)]
pub struct Ssimulacra2StripConfig {
    /// Number of rows above and below each strip's "interior" that
    /// are processed but excluded from the per-pixel reductions.
    pub halo_rows: usize,
    /// Kernel selection for the per-strip ops (scalar oracle vs SIMD —
    /// bit-identical).
    pub inner: Ssimulacra2Config,
    /// Process strips in parallel (requires the `rayon` feature).
    ///
    /// Off by default because parallelism multiplies the memory bound:
    /// peak RSS becomes ~`threads × (strip+2·halo) × width × ~30 f32
    /// planes`. The concurrency cap is `min(threads, 8)` — the workload
    /// is memory-bandwidth-bound past that. On low-RAM machines keep
    /// this `false` — the memory bound is the strip path's contract.
    /// Scores are bit-identical either way (sums merge in fixed strip
    /// order).
    pub parallel_strips: bool,
}

impl Default for Ssimulacra2StripConfig {
    fn default() -> Self {
        Self {
            halo_rows: HALO_ROWS_DEFAULT,
            inner: Ssimulacra2Config::default(),
            parallel_strips: false,
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
            parallel_strips: false,
        }
    }

    /// Set the underlying SIMD configuration.
    #[must_use]
    pub fn with_inner(mut self, inner: Ssimulacra2Config) -> Self {
        self.inner = inner;
        self
    }

    /// Enable parallel strip processing (requires `rayon`; multiplies
    /// peak memory by the thread count — see [`Self::parallel_strips`]).
    #[must_use]
    pub fn with_parallel_strips(mut self, parallel: bool) -> Self {
        self.parallel_strips = parallel;
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
/// - [`Ssimulacra2Error::InvalidImageSize`] if `strip_height <
///   [`MIN_STRIP_HEIGHT`].
/// - [`Ssimulacra2Error::NonMatchingImageDimensions`],
///   [`Ssimulacra2Error::ImageTooLarge`] as in
///   [`crate::compute_ssimulacra2`]. Sub-8px inputs are reflect-padded
///   per the crate contract.
pub fn compute_ssimulacra2_strip<S, D>(
    source: S,
    distorted: D,
    strip_height: u32,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    compute_ssimulacra2_strip_with_config_and_stop(
        source,
        distorted,
        strip_height,
        Ssimulacra2StripConfig::default(),
        &enough::Unstoppable,
    )
}

/// [`compute_ssimulacra2_strip`] with cooperative cancellation.
///
/// `stop` is checked once per strip (never per-pixel); on cancellation
/// the comparison returns [`Ssimulacra2Error::Cancelled`].
///
/// # Errors
/// As [`compute_ssimulacra2_strip`], plus `Cancelled`.
pub fn compute_ssimulacra2_strip_with_stop<S, D>(
    source: S,
    distorted: D,
    strip_height: u32,
    stop: &dyn enough::Stop,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    compute_ssimulacra2_strip_with_config_and_stop(
        source,
        distorted,
        strip_height,
        Ssimulacra2StripConfig::default(),
        stop,
    )
}

/// [`compute_ssimulacra2_strip`] with explicit configuration.
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
    compute_ssimulacra2_strip_with_config_and_stop(
        source,
        distorted,
        strip_height,
        config,
        &enough::Unstoppable,
    )
}

/// Strip entry point.
///
/// Inputs with an encoded form (`to_encoded_srgb`) are sliced per strip
/// in encoded space and linearized through the reference-captured LUT —
/// the same bit-exact path as [`crate::compute_ssimulacra2`]. Inputs
/// without an encoded form take the linear-planes variant.
fn compute_ssimulacra2_strip_with_config_and_stop<S, D>(
    source: S,
    distorted: D,
    strip_height: u32,
    config: Ssimulacra2StripConfig,
    stop: &dyn enough::Stop,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    if strip_height < MIN_STRIP_HEIGHT as u32 {
        return Err(Ssimulacra2Error::InvalidImageSize);
    }
    let opts = Opts {
        kernel: Kernel::from_impl(config.inner.impl_type),
    };
    let e1 = source.to_encoded_srgb();
    let e2 = distorted.to_encoded_srgb();
    if let (Some(e1), Some(e2)) = (&e1, &e2) {
        // Same sub-8px crate contract as the pair path — pad encoded
        // planes (per-pixel LUT ⇒ U8-exact, identical to padding
        // post-linearization).
        let p1 = e1.reflect_padded(8);
        let p2 = e2.reflect_padded(8);
        return crate::pipeline::strip::compute_encoded_strip_stop(
            &p1,
            &p2,
            strip_height as usize,
            config.halo_rows,
            opts,
            config.parallel_strips,
            stop,
        );
    }
    let img1: LinearRgbImage = source.into_linear_rgb();
    let img2: LinearRgbImage = distorted.into_linear_rgb();
    let (w1, h1) = (img1.width(), img1.height());
    let (w2, h2) = (img2.width(), img2.height());
    if w1 != w2 || h1 != h2 {
        return Err(Ssimulacra2Error::NonMatchingImageDimensions);
    }
    let mut p1 = linear_strip_planes(&img1);
    let mut p2 = linear_strip_planes(&img2);
    let (w, h) = if w1 < 8 || h1 < 8 {
        let (pw, ph) = (w1.max(8), h1.max(8));
        let pad = |p: [Vec<f32>; 3]| {
            p.map(|data| {
                let mut out = Vec::with_capacity(pw * ph);
                for y in 0..ph {
                    let row = crate::reflect_index(y, h1) * w1;
                    for x in 0..pw {
                        out.push(data[row + crate::reflect_index(x, w1)]);
                    }
                }
                out
            })
        };
        p1 = pad(p1);
        p2 = pad(p2);
        (pw, ph)
    } else {
        (w1, h1)
    };
    crate::pipeline::strip::compute_linear_strip_stop(
        &p1,
        &p2,
        w,
        h,
        strip_height as usize,
        config.halo_rows,
        opts,
        config.parallel_strips,
        stop,
    )
}

/// Deinterleave a [`LinearRgbImage`] into `[r, g, b]` planes.
fn linear_strip_planes(img: &LinearRgbImage) -> [Vec<f32>; 3] {
    let n = img.width() * img.height();
    let mut p = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for px in img.data() {
        p[0].push(px[0]);
        p[1].push(px[1]);
        p[2].push(px[2]);
    }
    p
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

    /// [`Ssimulacra2Reference::compare_strip`] with cooperative cancellation.
    ///
    /// `stop` is checked once per strip (never per-pixel); on cancellation the
    /// comparison returns [`Ssimulacra2Error::Cancelled`]. `compare_strip` is the
    /// no-cancellation equivalent (it passes `enough::Unstoppable`).
    ///
    /// # Errors
    /// As [`Ssimulacra2Reference::compare_strip`], plus `Cancelled`.
    pub fn compare_strip_with_stop<T: ToLinearRgb>(
        &self,
        distorted: T,
        strip_height: u32,
        stop: &dyn enough::Stop,
    ) -> Result<f64, Ssimulacra2Error> {
        self.compare_strip_with_config_and_stop(
            distorted,
            strip_height,
            Ssimulacra2StripConfig::default(),
            stop,
        )
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
        self.compare_strip_with_config_and_stop(
            distorted,
            strip_height,
            config,
            &enough::Unstoppable,
        )
    }

    /// [`Ssimulacra2Reference::compare_strip_with_config`] with cooperative cancellation.
    ///
    /// # Errors
    /// As [`Ssimulacra2Reference::compare_strip_with_config`], plus `Cancelled`.
    pub fn compare_strip_with_config_and_stop<T: ToLinearRgb>(
        &self,
        distorted: T,
        strip_height: u32,
        config: Ssimulacra2StripConfig,
        stop: &dyn enough::Stop,
    ) -> Result<f64, Ssimulacra2Error> {
        if strip_height < MIN_STRIP_HEIGHT as u32 {
            return Err(Ssimulacra2Error::InvalidImageSize);
        }
        let cache = self.cache();
        if let Some(e2) = distorted.to_encoded_srgb() {
            let p2 = e2.reflect_padded(8);
            if p2.width != cache.width() || p2.height != cache.height() {
                return Err(Ssimulacra2Error::NonMatchingImageDimensions);
            }
            return cache.compare_strip_stop(
                &p2,
                strip_height as usize,
                config.halo_rows,
                config.parallel_strips,
                stop,
            );
        }
        // Linear-input fallback — same rules as the non-strip path.
        let padded = crate::reflect_pad_linear(distorted.into_linear_rgb(), 8);
        let (pw, ph) = (padded.width(), padded.height());
        if (pw, ph) != (cache.width(), cache.height()) {
            return Err(Ssimulacra2Error::NonMatchingImageDimensions);
        }
        cache.compare_strip_linear(
            &linear_strip_planes(&padded),
            strip_height as usize,
            config.halo_rows,
            config.parallel_strips,
            stop,
        )
    }
}
