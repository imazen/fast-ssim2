//! # fast-ssim2
//!
//! Fast SIMD-accelerated implementation of [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2),
//! a perceptual image quality metric.
//!
//! ## Quick Start
//!
//! The simplest way to compare two images:
//!
//! ```ignore
//! use fast_ssim2::{compute_ssimulacra2, PixelDescriptor, PixelSlice};
//!
//! // Wrap decoded 8-bit sRGB pixels (flat u8 rows).
//! let source = PixelSlice::new(&rgb_bytes, w, h, w * 3, PixelDescriptor::RGB8_SRGB)?;
//! let distorted = PixelSlice::new(&rgb_bytes2, w, h, w * 3, PixelDescriptor::RGB8_SRGB)?;
//!
//! let score = compute_ssimulacra2(&source, &distorted)?;
//! // score: 100 = identical, 90+ = imperceptible, <50 = significant degradation
//! ```
//!
//! ## Score Interpretation
//!
//! | Score | Quality |
//! |-------|---------|
//! | **100** | Identical (no difference) |
//! | **90+** | Imperceptible difference |
//! | **70-90** | Minor, subtle difference |
//! | **50-70** | Noticeable difference |
//! | **<50** | Significant degradation |
//!
//! ## Supported Input Formats
//!
//! Inputs are [`zenpixels::PixelSlice`]s — the descriptor declares
//! layout + color semantics, and the pipeline scores what the bytes
//! honestly are:
//!
//! | Descriptor | Layout | Path |
//! |------------|--------|------|
//! | `RGB8_SRGB`/`RGBA8_SRGB`/`BGRA8_SRGB`/`GRAY8_SRGB`/`RGBX8_SRGB`/`BGRX8_SRGB` | u8 | **LUT-exact** (captured lcms table) |
//! | `RGB16_SRGB`/`RGBA16_SRGB`/`GRAY16_SRGB` | u16 | sRGB poly (`linear-srgb`) |
//! | `RGBF32`/`RGBAF32` + `TransferFunction::Srgb` | f32 encoded | sRGB poly (`linear-srgb`) |
//! | `RGBF32_LINEAR`/`RGBAF32_LINEAR`/`GRAYF32_LINEAR` | f32 | linear planes |
//!
//! HDR transfers (PQ/HLG), narrow signal range, and non-BT.709 primaries
//! are rejected with [`Ssimulacra2Error::UnsupportedInput`] in the SDR
//! entry points — convert via `zenpixels-convert`, or use
//! [`compute_ssimulacra2_pu`] (`hdr-pu` feature), which accepts Linear-nits
//! f32 and Pq/Hlg descriptors natively.
//!
//! Alpha channels use the reference's dual-background compositing;
//! premultiplied alpha is un-multiplied on ingest.
//!
//! ### Plain byte buffers
//!
//! ```
//! use fast_ssim2::{compute_ssimulacra2, PixelDescriptor, PixelSlice};
//!
//! // Plain u8 sRGB pixels, 64×64 mid-gray.
//! let data: Vec<u8> = vec![128; 64 * 64 * 3];
//! let slice = || {
//!     PixelSlice::new(&data, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap()
//! };
//!
//! let score = compute_ssimulacra2(&slice(), &slice())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Batch Comparisons (2x Faster)
//!
//! When comparing multiple images against the same reference (e.g., evaluating
//! different compression levels), precompute the reference data once:
//!
//! ```
//! use fast_ssim2::{Ssimulacra2Reference, PixelDescriptor, PixelSlice};
//!
//! // Create test data
//! let data: Vec<u8> = vec![128; 64 * 64 * 3];
//! let slice = || {
//!     PixelSlice::new(&data, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap()
//! };
//!
//! // Precompute reference data (~50% of the work)
//! let reference = Ssimulacra2Reference::new(&slice())?;
//!
//! // Compare multiple distorted versions efficiently
//! let score = reference.compare(&slice())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Custom Input Types
//!
//! Inputs are [`zenpixels::PixelSlice`]s — self-describing borrowed views
//! over strided bytes. Wrap a raw buffer with `PixelSlice::new` and a
//! [`zenpixels::PixelDescriptor`] constant (e.g. `RGB8_SRGB`); `imgref`
//! types convert via `PixelSlice::from` under the `imgref` feature, and
//! owned `PixelBuffer`s expose `as_slice()`.
//!
//! ```
//! use fast_ssim2::{PixelDescriptor, PixelSlice};
//!
//! // Plain u8 sRGB pixels: flat bytes + the descriptor says the rest.
//! let data: Vec<u8> = vec![128; 64 * 64 * 3];
//! let src = PixelSlice::new(&data, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Helper functions for sRGB conversion:
//! For sRGB↔linear conversion helpers, use `linear_srgb::default::*`
//! directly (same implementation the crate uses internally).
//!
//! ## SIMD Configuration
//!
//! SIMD is enabled by default via the `archmage` crate, providing cross-platform
//! acceleration on x86_64 (AVX2, AVX-512), AArch64 (NEON), and WASM (SIMD128).
//!
//! | Backend | Speed | Platforms |
//! |---------|-------|-----------|
//! | `Scalar` | 1.0× (baseline) | All |
//! | `Simd` (default) | 2-3× | x86_64, AArch64, WASM |
//!
//! To explicitly select a backend:
//!
//! ```
//! use fast_ssim2::{compute_ssimulacra2_with_config, Ssimulacra2Config};
//!
//! # let mk = || fast_ssim2::PixelBuffer::from_vec(vec![0u8; 64 * 12], 8, 8,
//! #     fast_ssim2::PixelDescriptor::RGBF32_LINEAR).unwrap();
//! # let source = mk(); let distorted = mk();
//! let score = compute_ssimulacra2_with_config(
//!     &source.as_slice(),
//!     &distorted.as_slice(),
//!     &Ssimulacra2Config::scalar(), // or ::simd()
//! )?;
//! # Ok::<(), fast_ssim2::Ssimulacra2Error>(())
//! ```
//!
//! ## Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `imgref` | | Forwards `zenpixels/imgref` — `ImgRef`/`ImgVec` → `PixelSlice` (rgb-crate pixels) |
//! | `hdr-pu` | | HDR scoring: PU21-integrated encoding (`compute_ssimulacra2_pu`); nits/PQ/HLG inputs |
//! | `rayon` | | Parallel computation |
//!
//! ## Requirements
//!
//! - **Image size:** [`compute_ssimulacra2`], [`Ssimulacra2Reference`],
//!   and [`compute_ssimulacra2_strip`] accept any size from 1×1 up to
//!   [`MAX_IMAGE_PIXELS`] pixels; inputs below the metric's 8×8 pyramid
//!   floor are reflect(mirror)-padded before processing (a crate-level
//!   extension — the reference binary refuses such images).
//! - **MSRV:** 1.89.0

#![forbid(unsafe_code)]

mod input;
#[doc(hidden)]
pub mod pipeline;
mod precompute;
mod source;
// Reference data for parity testing (hidden from docs but accessible for tests)
#[doc(hidden)]
pub mod reference_data;
mod strip;
mod weights;

pub use precompute::Ssimulacra2Reference;
pub use strip::{HALO_ROWS_DEFAULT, MIN_STRIP_HEIGHT, StripConfig};
pub use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice, TransferFunction};

// sRGB→linear for callers: use `linear_srgb::default::*` directly.

/// SIMD implementation backend for all operations (blur, XYB conversion, SSIM computation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimdImpl {
    /// Scalar implementation (baseline, most portable)
    Scalar,
    /// Cross-platform SIMD via archmage (default, AVX2/AVX-512/NEON/WASM128)
    #[default]
    Simd,
}

impl SimdImpl {
    /// Returns the name of this implementation
    pub fn name(&self) -> &'static str {
        match self {
            SimdImpl::Scalar => "scalar",
            SimdImpl::Simd => "simd (archmage)",
        }
    }
}

/// Configuration for SSIMULACRA2 computation.
///
/// The pipeline is a bit-exact port of the reference implementation
/// (both published `ssimulacra2` binaries agree); [`SimdImpl`] selects
/// the kernel family — `Scalar` is the audit oracle, `Simd` is
/// bit-identical and ~7x fewer instructions.
/// Per-call options. `Default` = SIMD kernels, whole-image, unstoppable
/// — identical to plain [`compute_ssimulacra2`].
///
/// (`Debug` skipped: the `stop` token is a trait object.)
#[derive(Clone, Copy, Default)]
pub struct Ssimulacra2Config<'a> {
    /// Kernel backend for all operations.
    pub impl_type: SimdImpl,
    /// Strip-wise evaluation for bounded memory on large images —
    /// `Some(StripConfig)` routes to the strip pipeline.
    pub strip: Option<StripConfig>,
    /// Cooperative cancellation token — checked once per scale or per
    /// strip, never per-pixel.
    pub stop: Option<&'a dyn enough::Stop>,
}

impl<'a> Ssimulacra2Config<'a> {
    /// Create configuration with specified implementation.
    pub fn new(impl_type: SimdImpl) -> Self {
        Self {
            impl_type,
            strip: None,
            stop: None,
        }
    }

    /// Default configuration using SIMD kernels.
    pub fn simd() -> Self {
        Self::new(SimdImpl::Simd)
    }

    /// Scalar configuration — the reference-order oracle kernels.
    pub fn scalar() -> Self {
        Self::new(SimdImpl::Scalar)
    }

    /// Strip-wise evaluation (bounded memory): `strip_height` rows per
    /// strip interior, default halo, serial.
    pub fn strips(strip_height: usize) -> Self {
        Self {
            strip: Some(StripConfig {
                strip_height,
                ..Default::default()
            }),
            ..Self::default()
        }
    }

    /// Attach a cancellation token.
    pub fn with_stop(mut self, stop: &'a dyn enough::Stop) -> Self {
        self.stop = Some(stop);
        self
    }
}

/// Errors which can occur when attempting to calculate a SSIMULACRA2 score from two input images.
///
/// `#[non_exhaustive]`: downstream `match` arms must include a wildcard `_ =>`,
/// so future variants (like [`Ssimulacra2Error::Cancelled`]) can be added without
/// breaking callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Ssimulacra2Error {
    /// An input [`PixelSlice`](crate::PixelSlice)'s descriptor declares
    /// something the metric can't score honestly — HDR transfers
    /// (PQ/HLG), narrow/limited signal range, or a pixel layout with no
    /// mapping to the SDR sRGB pipeline.
    #[error("Unsupported input: {0}")]
    UnsupportedInput(&'static str),

    /// The two input images do not have the same width and height.
    #[error("Source and distorted image width and height must be equal")]
    NonMatchingImageDimensions,

    /// One of the input images is below the metric's 8×8 pyramid floor,
    /// in a code path that does not reflect-pad.
    ///
    /// The primary entry points ([`compute_ssimulacra2`],
    /// [`Ssimulacra2Reference`], and the strip APIs) reflect-pad sub-8px
    /// inputs instead of returning this error; it still signals
    /// `strip_height < 8` and empty inputs.
    #[error("Images must be at least 8x8 pixels")]
    InvalidImageSize,

    /// One of the input images exceeds the maximum supported pixel count.
    ///
    /// SSIMULACRA2 allocates roughly 24 image-sized `f32` planes of working
    /// memory plus several downscaled copies of the input, so unbounded
    /// caller-supplied dimensions are a denial-of-service vector. The current
    /// cap is [`MAX_IMAGE_PIXELS`] pixels (`width * height`), matching the
    /// largest practical web-corpus image we test against. Callers that need
    /// to compare larger images should tile and aggregate.
    #[error(
        "Image is too large: {actual} pixels exceeds limit of {} pixels",
        MAX_IMAGE_PIXELS
    )]
    ImageTooLarge {
        /// Pixel count (`width * height`) of the offending image.
        actual: usize,
    },

    /// Gaussian blur operation failed.
    #[error("Gaussian blur operation failed")]
    GaussianBlurError,

    /// The computation was cooperatively cancelled via the
    /// [`enough::Stop`] token passed to a `*_with_stop` entry point.
    ///
    /// The token is polled at the top of each multi-scale (and, in the
    /// strip APIs, per-strip) outer-loop iteration, never inside the
    /// per-pixel inner loops, so cancellation is responsive without
    /// adding overhead to the hot path.
    #[error("Computation cancelled: {0}")]
    Cancelled(enough::StopReason),
}

/// Maximum supported image size in pixels (`width * height`).
///
/// SSIMULACRA2 allocates O(24 * width * height * 4 bytes) of working memory
/// plus downscaled pyramid copies. At this cap, peak working memory stays
/// under ~6 GiB on 64-bit hosts, which is high but bounded; callers that
/// embed fast-ssim2 should treat this as the *maximum* trusted-input size.
/// Untrusted callers should impose a tighter limit upstream.
///
/// 16 384 * 16 384 = 268 435 456 pixels, comfortably above any practical
/// still-image use case (8K UHD = 33 MP, full-frame 100 MP DSLR sensors fit).
pub const MAX_IMAGE_PIXELS: usize = 16_384 * 16_384;

/// Computes the SSIMULACRA2 score from two [`PixelSlice`]s —
/// borrowed, self-describing views over pixel bytes (stride, format,
/// transfer, primaries, and alpha all live in the descriptor).
///
/// Wrap raw buffers with `PixelSlice::new(bytes, w, h, stride, descriptor)`;
/// `imgref`/`rgb`-crate images convert via `PixelSlice::from` (the
/// `imgref` feature forwards to `zenpixels/imgref`); `PixelBuffer`
/// callers pass `&buf.as_slice()`.
///
/// # Color space conventions
/// - Integer types (`u8`, `u16`) are assumed to be sRGB (gamma-encoded)
/// - Float types (`f32`) are assumed to be linear RGB
/// - Grayscale types are expanded to RGB (R=G=B)
///
/// # Example
/// ```ignore
/// use fast_ssim2::{compute_ssimulacra2, PixelDescriptor, PixelSlice};
///
/// let source = PixelSlice::new(&rgb, w, h, w * 3, PixelDescriptor::RGB8_SRGB)?;
/// let distorted = PixelSlice::new(&rgb2, w, h, w * 3, PixelDescriptor::RGB8_SRGB)?;
/// let score = compute_ssimulacra2(&source, &distorted)?;
/// ```
pub fn compute_ssimulacra2(
    source: &zenpixels::PixelSlice<'_>,
    distorted: &zenpixels::PixelSlice<'_>,
) -> Result<f64, Ssimulacra2Error> {
    compute_ssimulacra2_with_config(source, distorted, &Ssimulacra2Config::default())
}

/// Computes the SSIMULACRA2 score with custom configuration.
///
/// [`Ssimulacra2Config`] carries everything per-call: kernel family
/// (`impl_type`), bounded-memory strips (`strip`), and cooperative
/// cancellation (`stop` — checked once per scale or per strip, never
/// per-pixel; returns [`Ssimulacra2Error::Cancelled`]).
///
/// ```ignore
/// // strip mode, 128-row interiors:
/// let cfg = Ssimulacra2Config::strips(128);
/// compute_ssimulacra2_with_config(&a, &b, &cfg)
/// ```
pub fn compute_ssimulacra2_with_config(
    source: &zenpixels::PixelSlice<'_>,
    distorted: &zenpixels::PixelSlice<'_>,
    config: &Ssimulacra2Config<'_>,
) -> Result<f64, Ssimulacra2Error> {
    if config.strip.is_some() {
        return crate::strip::compute_strip_inner(source, distorted, config);
    }
    let kernel = pipeline::Kernel::from_impl(config.impl_type);
    let stop: &dyn enough::Stop = config.stop.unwrap_or(&enough::Unstoppable);
    compute_pair(source, distorted, kernel, stop)
}

/// HDR variant (`hdr-pu` feature): SSIMULACRA2 with the cube-root opsin
/// replaced by **PU21** perceptual encoding of absolute luminance (cd/m²).
///
/// Accepted inputs:
/// - `TransferFunction::Linear` f32 — pixels are absolute nits.
/// - `TransferFunction::Pq`/`Hlg` (u8/u16/f32) — EOTF-decoded to nits
///   internally (PQ: ST 2084 → 0–10 000 cd/m²; HLG: inverse-OETF +
///   system-gamma 1.2 OOTF on a 1000 cd/m² reference display).
///
/// BT.2020 primaries are *not* gamut-converted — the opsin consumes the
/// declared primaries directly (same convention as zensim's PU path;
/// SROCC ~0.69 on UPIQ HDR).
///
/// **Scores are not comparable to [`compute_ssimulacra2`] scores** — this
/// is a different perceptual-encoding regime for HDR content.
#[cfg(feature = "hdr-pu")]
pub fn compute_ssimulacra2_pu(
    source: &zenpixels::PixelSlice<'_>,
    distorted: &zenpixels::PixelSlice<'_>,
) -> Result<f64, Ssimulacra2Error> {
    compute_ssimulacra2_pu_with_config(source, distorted, &Ssimulacra2Config::default())
}

/// [`compute_ssimulacra2_pu`] with custom configuration (kernel +
/// cancellation; `strip` is ignored — HDR strip is not supported).
#[cfg(feature = "hdr-pu")]
pub fn compute_ssimulacra2_pu_with_config(
    source: &zenpixels::PixelSlice<'_>,
    distorted: &zenpixels::PixelSlice<'_>,
    config: &Ssimulacra2Config<'_>,
) -> Result<f64, Ssimulacra2Error> {
    let stop: &dyn enough::Stop = config.stop.unwrap_or(&enough::Unstoppable);
    let kernel = pipeline::Kernel::from_impl(config.impl_type);
    let p1 = source::funnel_nits(source)?;
    let p2 = source::funnel_nits(distorted)?;
    let (w1, h1) = p1.dims();
    let (w2, h2) = p2.dims();
    if w1 != w2 || h1 != h2 {
        return Err(Ssimulacra2Error::NonMatchingImageDimensions);
    }
    let opts = pipeline::Opts {
        kernel,
        flavor: pipeline::XybFlavor::Pu21,
    };
    // Alpha: composite onto the SDR dark/light-equivalent backgrounds
    // (20 / 200 cd/m²), mirroring the encoded path's 0.1/0.9 convention.
    let has_alpha = prepared_has_alpha(&p1) || prepared_has_alpha(&p2);
    let once = |bg: f32| -> Result<f64, Ssimulacra2Error> {
        let mut a = linearize_prepared(&p1, w1, bg);
        let mut b = linearize_prepared(&p2, w2, bg);
        let (w, h) = if w1 < 8 || h1 < 8 {
            let (pw, ph) = (w1.max(8), h1.max(8));
            a = pad_planes(a, w1, h1, pw, ph);
            b = pad_planes(b, w2, h2, pw, ph);
            (pw, ph)
        } else {
            (w1, h1)
        };
        pipeline::compute_planar_stop(a, b, w, h, opts, stop)
    };
    if has_alpha {
        Ok(once(20.0)?.min(once(200.0)?))
    } else {
        once(200.0)
    }
}

/// Core pair-scoring entry: the reference SSIMULACRA2.1 pipeline
/// reproduced bit-for-bit.
///
/// Encoded sRGB inputs are linearized through the captured reference
/// LUTs; already-linear inputs are used as-is (the reference binary
/// never sees such inputs, so bit-exactness there is undefined).
fn compute_pair(
    source: &zenpixels::PixelSlice<'_>,
    distorted: &zenpixels::PixelSlice<'_>,
    kernel: pipeline::Kernel,
    stop: &dyn enough::Stop,
) -> Result<f64, Ssimulacra2Error> {
    use source::PreparedInput;

    let p1 = source::funnel(source)?;
    let p2 = source::funnel(distorted)?;

    if let (PreparedInput::Encoded(e1), PreparedInput::Encoded(e2)) = (&p1, &p2) {
        // Sub-8px inputs: the reference binary refuses them, but the
        // crate's `compute_ssimulacra2` contract scores down to 1×1 via
        // mirror padding — apply it on encoded planes (per-pixel LUT ⇒
        // identical to padding post-linearization, and stays U8-exact).
        let q1 = e1.reflect_padded(8);
        let q2 = e2.reflect_padded(8);
        return pipeline::compute_encoded_stop(&q1, &q2, kernel, stop);
    }

    let (w1, h1) = p1.dims();
    let (w2, h2) = p2.dims();
    if w1 != w2 || h1 != h2 {
        return Err(Ssimulacra2Error::NonMatchingImageDimensions);
    }

    let has_alpha = prepared_has_alpha(&p1) || prepared_has_alpha(&p2);

    // Linear-side mirror padding to the 8px pyramid floor (crate contract;
    // the reference binary refuses such inputs — padding is our extension).
    let (pw, ph) = (w1.max(8), h1.max(8));
    let opts = pipeline::Opts {
        kernel,
        flavor: pipeline::XybFlavor::CubeRoot,
    };

    // Reference alpha compositing: min over two backgrounds (0.1 / 0.9
    // encoded). Linear inputs replicate it with the linearized bg.
    let eval = |bg: f32, p1: &PreparedInput, p2: &PreparedInput| -> Result<f64, Ssimulacra2Error> {
        let l1 = linearize_prepared(p1, w1, bg);
        let l2 = linearize_prepared(p2, w2, bg);
        let (l1, l2) = if pw != w1 || ph != h1 {
            (
                pad_planes(l1, w1, h1, pw, ph),
                pad_planes(l2, w2, h2, pw, ph),
            )
        } else {
            (l1, l2)
        };
        pipeline::compute_planar_stop(l1, l2, pw, ph, opts, stop)
    };

    if has_alpha {
        Ok(eval(0.1, &p1, &p2)?.min(eval(0.9, &p1, &p2)?))
    } else {
        eval(0.5, &p1, &p2)
    }
}

pub(crate) fn prepared_has_alpha(p: &source::PreparedInput) -> bool {
    match p {
        source::PreparedInput::Encoded(e) => e.alpha.is_some(),
        source::PreparedInput::Linear { alpha, .. } => alpha.is_some(),
    }
}

/// Linearize a funnelled input: encoded data goes through the reference
/// LUTs (alpha premultiplied onto `bg` in encoded space), linear planes
/// premultiply alpha onto `srgb_to_linear(bg)` for parity.
pub(crate) fn linearize_prepared(p: &source::PreparedInput, _w: usize, bg: f32) -> [Vec<f32>; 3] {
    match p {
        source::PreparedInput::Encoded(e) => pipeline::linearize(e, bg),
        source::PreparedInput::Linear { planes, alpha, .. } => match alpha {
            None => planes.clone(),
            Some(a) => {
                let bg_lin = input::srgb_to_linear(bg);
                planes.clone().map(|ch| {
                    ch.iter()
                        .zip(a.iter())
                        .map(|(&v, &af)| af * v + (1.0 - af) * bg_lin)
                        .collect()
                })
            }
        },
    }
}

/// Reflect-101 pad linear planes to `pw × ph` (original at top-left).
pub(crate) fn pad_planes(
    planes: [Vec<f32>; 3],
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
) -> [Vec<f32>; 3] {
    planes.map(|data| {
        let mut out = Vec::with_capacity(pw * ph);
        for y in 0..ph {
            let row = reflect_index(y, h) * w;
            for x in 0..pw {
                out.push(data[row + reflect_index(x, w)]);
            }
        }
        out
    })
}

/// Reflect-101 index map (OpenCV `BORDER_REFLECT_101`): fold an
/// out-of-range index `i` back into `[0, n)` by mirroring at the borders
/// without repeating the edge sample. Identity for `i < n`; `n <= 1`
/// collapses to 0.
#[inline]
pub(crate) fn reflect_index(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut k = i % period;
    if k >= n {
        k = period - k;
    }
    k
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// Wrap `&[[f32; 3]]` as an encoded-sRGB `PixelSlice`.
    fn srgb_f32(data: &[[f32; 3]], w: usize, h: usize) -> zenpixels::PixelSlice<'_> {
        zenpixels::PixelSlice::new(
            bytemuck::cast_slice(data),
            w as u32,
            h as u32,
            w * 12,
            zenpixels::PixelDescriptor::RGBF32.with_transfer(zenpixels::TransferFunction::Srgb),
        )
        .unwrap()
    }

    /// Wrap `&[[f32; 3]]` as a linear `PixelSlice`.
    fn lin_f32(data: &[[f32; 3]], w: usize, h: usize) -> zenpixels::PixelSlice<'_> {
        zenpixels::PixelSlice::new(
            bytemuck::cast_slice(data),
            w as u32,
            h as u32,
            w * 12,
            zenpixels::PixelDescriptor::RGBF32_LINEAR,
        )
        .unwrap()
    }

    /// Wrap `&[[u8; 3]]` as `RGB8_SRGB`.
    fn rgb8_slice(data: &[[u8; 3]], w: usize, h: usize) -> zenpixels::PixelSlice<'_> {
        zenpixels::PixelSlice::new(
            bytemuck::cast_slice(data),
            w as u32,
            h as u32,
            w * 3,
            zenpixels::PixelDescriptor::RGB8_SRGB,
        )
        .unwrap()
    }

    /// sRGB-encoded f32 raster — the on-grid encoded-input path.
    fn tank_rgb() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, u32, u32) {
        let mk = |name: &str| {
            let img = image::open(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("test_data")
                    .join(name),
            )
            .unwrap();
            (
                img.to_rgb32f().as_chunks::<3>().0.to_vec(),
                img.width(),
                img.height(),
            )
        };
        let (d1, w1, h1) = mk("tank_source.png");
        let (d2, w2, h2) = mk("tank_distorted.png");
        assert_eq!((w1, h1), (w2, h2));
        (d1, d2, w1, h1)
    }

    #[test]
    fn test_ssimulacra2() {
        let (s, d, w, h) = tank_rgb();
        let score = compute_ssimulacra2(
            &srgb_f32(&s, w as usize, h as usize),
            &srgb_f32(&d, w as usize, h as usize),
        )
        .unwrap();
        assert!(
            (17.0..=18.0).contains(&score),
            "tank score {score} outside expected band (ref-decoder ~17.4)"
        );
    }

    #[test]
    fn test_u8_and_f32_encoded_nearly_agree() {
        // u8 (captured LUT) vs f32-encoded sRGB (rational poly): the
        // general f32 path approximates the LUT within ~1e-7/pixel, so
        // scores stay within a small tolerance — they're *sane-equal*,
        // not bit-exact (quantized callers get exactness via u8).
        let s = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("test_data")
                .join("tank_source.png"),
        )
        .unwrap()
        .to_rgb8();
        let d = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("test_data")
                .join("tank_distorted.png"),
        )
        .unwrap()
        .to_rgb8();
        let (w, h) = s.dimensions();
        let mk = |img: &image::RgbImage| {
            img.pixels()
                .map(|p| [p[0], p[1], p[2]])
                .collect::<Vec<[u8; 3]>>()
        };
        let (a8, b8) = (mk(&s), mk(&d));
        let mkr = |img: &image::RgbImage| {
            img.pixels()
                .map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                    ]
                })
                .collect::<Vec<[f32; 3]>>()
        };
        let (af, bf) = (mkr(&s), mkr(&d));
        let score_u8 = compute_ssimulacra2(
            &rgb8_slice(&a8, w as usize, h as usize),
            &rgb8_slice(&b8, w as usize, h as usize),
        )
        .unwrap();
        let score_f32 = compute_ssimulacra2(
            &srgb_f32(&af, w as usize, h as usize),
            &srgb_f32(&bf, w as usize, h as usize),
        )
        .unwrap();
        assert!(
            (score_u8 - score_f32).abs() < 0.15,
            "u8 {score_u8} vs f32-encoded {score_f32} — poly-vs-LUT drift"
        );
    }

    /// Construct a mid-gray linear image.
    fn make_linear_rgb(width: usize, height: usize) -> Vec<[f32; 3]> {
        vec![[0.5f32, 0.5, 0.5]; width * height]
    }

    #[test]
    fn test_compute_rejects_too_large_input() {
        // We can't allocate MAX_IMAGE_PIXELS+1 floats in unit tests, so
        // verify the error variant renders and the constant is sane.
        const { assert!(MAX_IMAGE_PIXELS >= 8 * 8) };
        let err = Ssimulacra2Error::ImageTooLarge {
            actual: MAX_IMAGE_PIXELS + 1,
        };
        let msg = format!("{err}");
        assert!(msg.contains("too large"), "unexpected message: {msg}");
        assert!(
            msg.contains(&MAX_IMAGE_PIXELS.to_string()),
            "message should reference the limit: {msg}"
        );
    }

    #[test]
    fn test_compute_accepts_small_input() {
        let img = make_linear_rgb(16, 16);
        let score = compute_ssimulacra2_with_config(
            &lin_f32(&img, 16, 16),
            &lin_f32(&img, 16, 16),
            &Ssimulacra2Config::default(),
        )
        .expect("16x16 grey image must be accepted");
        assert!(
            (score - 100.0).abs() < 0.01,
            "identical images should score 100, got {score}"
        );
    }

    #[test]
    fn test_sub_8_reflect_pads_instead_of_rejecting() {
        // Sub-8px inputs reflect(mirror)-pad up to the pyramid floor and
        // score (down to 1×1) rather than erroring. Identical pairs
        // still score ~100.
        for (w, h) in [(4usize, 4usize), (1, 1), (3, 7), (7, 3)] {
            let img = make_linear_rgb(w, h);
            let score = compute_ssimulacra2_with_config(
                &lin_f32(&img, w, h),
                &lin_f32(&img, w, h),
                &Ssimulacra2Config::default(),
            )
            .unwrap_or_else(|e| panic!("{w}x{h} must score, got {e:?}"));
            assert!(
                (score - 100.0).abs() < 0.01,
                "identical {w}x{h} should score ~100, got {score}"
            );
        }
        // A real sub-8 difference yields a finite score below 100.
        let a = make_linear_rgb(5, 5);
        let b = vec![[0.9f32, 0.1, 0.2]; 25];
        let s = compute_ssimulacra2_with_config(
            &lin_f32(&a, 5, 5),
            &lin_f32(&b, 5, 5),
            &Ssimulacra2Config::default(),
        )
        .expect("5x5 differing pair must score");
        assert!(s.is_finite() && s < 100.0, "5x5 differing score {s}");
    }

    #[test]
    fn test_scalar_kernels_bit_identical() {
        let (sa, sb, w, h) = tank_rgb();
        let (w, h) = (w as usize, h as usize);
        let simd = compute_ssimulacra2_with_config(
            &srgb_f32(&sa, w, h),
            &srgb_f32(&sb, w, h),
            &Ssimulacra2Config::simd(),
        )
        .unwrap();
        let scalar = compute_ssimulacra2_with_config(
            &srgb_f32(&sa, w, h),
            &srgb_f32(&sb, w, h),
            &Ssimulacra2Config::scalar(),
        )
        .unwrap();
        assert_eq!(simd, scalar, "scalar vs simd kernel mismatch");
    }
}
