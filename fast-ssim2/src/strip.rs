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
//! [`StripConfig::with_halo_rows`] for stricter parity at
//! the cost of slightly more per-strip work.
//!
//! ## Example
//!
//! Strip mode is selected through [`Ssimulacra2Config::strip`]:
//!
//! ```
//! use fast_ssim2::{
//!     PixelDescriptor, PixelSlice, Ssimulacra2Config, compute_ssimulacra2_with_config,
//! };
//!
//! let data: Vec<u8> = vec![128; 256 * 256 * 3];
//! let source =
//!     PixelSlice::new(&data, 256, 256, 256 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
//! let distorted =
//!     PixelSlice::new(&data, 256, 256, 256 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
//!
//! // Process in strips of 64 interior rows each.
//! let cfg = Ssimulacra2Config::strips(64);
//! let score = compute_ssimulacra2_with_config(&source, &distorted, &cfg).unwrap();
//! assert!((score - 100.0).abs() < 1e-3);
//! ```
//!
//! ## Cached-reference strip API
//!
//! When comparing many distorted images against the same reference,
//! pair [`Ssimulacra2Reference::new`] with
//! [`Ssimulacra2Reference::compare_with_config`] + `config.strip` for
//! the warm-ref + strip benefit:
//!
//! ```ignore
//! let reference = Ssimulacra2Reference::new(source)?;
//! for distorted in distortions {
//!     let score = reference.compare_with_config(distorted, &Ssimulacra2Config::strips(64))?;
//! }
//! ```
//!
//! Note that `compare_strip` still holds the full precomputed reference
//! in memory; the strip discipline only bounds dist-side peak memory.
//! For full strip mode on both sides, use [`compute_ssimulacra2_strip`]
//! directly.

use crate::pipeline::{Kernel, Opts, XybFlavor};
use crate::precompute::Ssimulacra2Reference;
use crate::{Ssimulacra2Config, Ssimulacra2Error};
use zenpixels::PixelSlice;

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

/// Strip-wise evaluation parameters, selected via
/// [`Ssimulacra2Config::strip`]. `strip_height` is in scale-0 rows.
#[derive(Debug, Clone, Copy)]
pub struct StripConfig {
    /// Interior rows per strip (min [`MIN_STRIP_HEIGHT`]).
    pub strip_height: usize,
    /// Number of rows above and below each strip's "interior" that
    /// are processed but excluded from the per-pixel reductions.
    pub halo_rows: usize,
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

impl Default for StripConfig {
    fn default() -> Self {
        Self {
            strip_height: 256,
            halo_rows: HALO_ROWS_DEFAULT,
            parallel_strips: false,
        }
    }
}

impl StripConfig {
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
/// Crate-internal strip runner — invoked from the public entry points
/// when [`Ssimulacra2Config::strip`] is `Some`. The strip parameters
/// live in [`StripConfig`]; kernel/stop come from the outer config.
///
/// Inputs with an encoded sRGB pixel format are sliced per strip
/// in encoded space and linearized through the reference-captured LUT —
/// the same bit-exact path as [`crate::compute_ssimulacra2`]. `LinearF32*`
/// inputs take the linear-planes variant.
pub(crate) fn compute_strip_inner(
    source: &PixelSlice<'_>,
    distorted: &PixelSlice<'_>,
    config: &Ssimulacra2Config<'_>,
) -> Result<f64, Ssimulacra2Error> {
    use crate::source::PreparedInput;

    let sc = config.strip.expect("strip config required");
    if sc.strip_height < MIN_STRIP_HEIGHT {
        return Err(Ssimulacra2Error::InvalidImageSize);
    }
    let stop: &dyn enough::Stop = config.stop.unwrap_or(&enough::Unstoppable);
    let opts = Opts {
        kernel: Kernel::from_impl(config.impl_type),
        flavor: XybFlavor::CubeRoot,
    };
    let strip_height = sc.strip_height;
    let p1 = crate::source::funnel(source)?;
    let p2 = crate::source::funnel(distorted)?;
    if let (PreparedInput::Encoded(e1), PreparedInput::Encoded(e2)) = (&p1, &p2) {
        // Same sub-8px crate contract as the pair path — pad encoded
        // planes (per-pixel LUT ⇒ U8-exact, identical to padding
        // post-linearization).
        let p1 = e1.reflect_padded(8);
        let p2 = e2.reflect_padded(8);
        return crate::pipeline::strip::compute_encoded_strip_stop(
            &p1,
            &p2,
            strip_height,
            sc.halo_rows,
            opts,
            sc.parallel_strips,
            stop,
        );
    }
    let (w1, h1) = p1.dims();
    let (w2, h2) = p2.dims();
    if w1 != w2 || h1 != h2 {
        return Err(Ssimulacra2Error::NonMatchingImageDimensions);
    }
    let has_alpha = crate::prepared_has_alpha(&p1) || crate::prepared_has_alpha(&p2);
    let strip_once = |bg: f32| -> Result<f64, Ssimulacra2Error> {
        let mut a = crate::linearize_prepared(&p1, w1, bg);
        let mut b = crate::linearize_prepared(&p2, w2, bg);
        let (w, h) = if w1 < 8 || h1 < 8 {
            let (pw, ph) = (w1.max(8), h1.max(8));
            a = crate::pad_planes(a, w1, h1, pw, ph);
            b = crate::pad_planes(b, w2, h2, pw, ph);
            (pw, ph)
        } else {
            (w1, h1)
        };
        crate::pipeline::strip::compute_linear_strip_stop(
            &a,
            &b,
            w,
            h,
            strip_height,
            sc.halo_rows,
            opts,
            sc.parallel_strips,
            stop,
        )
    };
    if has_alpha {
        Ok(strip_once(0.1)?.min(strip_once(0.9)?))
    } else {
        strip_once(0.5)
    }
}

impl Ssimulacra2Reference {
    /// Strip-bounded comparison — crate-internal; the public surface is
    /// [`Ssimulacra2Reference::compare_with_config`] with `config.strip`.
    pub(crate) fn compare_strip_inner(
        &self,
        distorted: &PixelSlice<'_>,
        config: &Ssimulacra2Config<'_>,
    ) -> Result<f64, Ssimulacra2Error> {
        let sc = config.strip.expect("strip config required");
        if sc.strip_height < MIN_STRIP_HEIGHT {
            return Err(Ssimulacra2Error::InvalidImageSize);
        }
        let stop: &dyn enough::Stop = config.stop.unwrap_or(&enough::Unstoppable);
        let strip_height = sc.strip_height;
        let cache = self.cache();
        let p = crate::source::funnel(distorted)?;
        match p {
            crate::source::PreparedInput::Encoded(e2) => {
                let p2 = e2.reflect_padded(8);
                if p2.width != cache.width() || p2.height != cache.height() {
                    return Err(Ssimulacra2Error::NonMatchingImageDimensions);
                }
                cache.compare_strip_stop(&p2, strip_height, sc.halo_rows, sc.parallel_strips, stop)
            }
            crate::source::PreparedInput::Linear { .. } => {
                // Linear side — compare against every ref stack; the
                // stack's bg drives the dist-side premultiply.
                let bgs: &[f32] = if cache.has_alpha() {
                    &[0.1, 0.9]
                } else {
                    &[0.5]
                };
                let mut best = f64::INFINITY;
                for (si, &bg) in bgs.iter().enumerate() {
                    let planes = crate::linearize_prepared(&p, cache.width(), bg);
                    let (pw, ph) = (cache.width(), cache.height());
                    let planes = if p.dims() != (pw, ph) {
                        crate::pad_planes(planes, p.dims().0, p.dims().1, pw, ph)
                    } else {
                        planes
                    };
                    best = best.min(cache.compare_strip_linear_stack(
                        si,
                        &planes,
                        strip_height,
                        sc.halo_rows,
                        sc.parallel_strips,
                        stop,
                    )?);
                }
                Ok(best)
            }
        }
    }
}
