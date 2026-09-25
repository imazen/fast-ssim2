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
//! use fast_ssim2::compute_ssimulacra2;
//! use imgref::ImgVec;
//!
//! // Load your images (8-bit sRGB)
//! let source: ImgVec<[u8; 3]> = load_image("source.png");
//! let distorted: ImgVec<[u8; 3]> = load_image("distorted.png");
//!
//! let score = compute_ssimulacra2(source.as_ref(), distorted.as_ref())?;
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
//! ### With `imgref` feature (recommended for most users)
//!
//! | Type | Color Space | Notes |
//! |------|-------------|-------|
//! | `ImgRef<[u8; 3]>` | sRGB | Standard 8-bit RGB images |
//! | `ImgRef<[u16; 3]>` | sRGB | 16-bit RGB (high bit depth, SDR) |
//! | `ImgRef<[f32; 3]>` | **Linear RGB** | Already linearized data |
//! | `ImgRef<u8>` | sRGB grayscale | Expanded to R=G=B |
//! | `ImgRef<f32>` | Linear grayscale | Expanded to R=G=B |
//!
//! **Convention:** Integer types assume sRGB gamma encoding. Float types assume linear RGB.
//!
//! ### Without features (using `yuvxyb` types)
//!
//! ```
//! use fast_ssim2::compute_ssimulacra2;
//! use yuvxyb::{Rgb, TransferCharacteristic, ColorPrimaries};
//! use std::num::NonZeroUsize;
//!
//! let data: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 64 * 64];
//! let w = NonZeroUsize::new(64).unwrap();
//! let h = NonZeroUsize::new(64).unwrap();
//! let source = Rgb::new(data.clone(), w, h,
//!     TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;
//! let distorted = Rgb::new(data, w, h,
//!     TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;
//!
//! let score = compute_ssimulacra2(source, distorted)?;
//! // compute_ssimulacra2 accepts yuvxyb::Rgb, yuvxyb::LinearRgb, and more
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Batch Comparisons (2x Faster)
//!
//! When comparing multiple images against the same reference (e.g., evaluating
//! different compression levels), precompute the reference data once:
//!
//! ```
//! use fast_ssim2::Ssimulacra2Reference;
//! use yuvxyb::{Rgb, TransferCharacteristic, ColorPrimaries};
//! use std::num::NonZeroUsize;
//!
//! // Create test data
//! let data: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 64 * 64];
//! let w = NonZeroUsize::new(64).unwrap();
//! let h = NonZeroUsize::new(64).unwrap();
//! let source = Rgb::new(data.clone(), w, h,
//!     TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;
//!
//! // Precompute reference data (~50% of the work)
//! let reference = Ssimulacra2Reference::new(source)?;
//!
//! // Compare multiple distorted versions efficiently
//! let distorted = Rgb::new(data, w, h,
//!     TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;
//! let score = reference.compare(distorted)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Custom Input Types
//!
//! Implement [`ToLinearRgb`] to support your own image types:
//!
//! ```
//! use fast_ssim2::{ToLinearRgb, LinearRgbImage, srgb_u8_to_linear};
//!
//! struct MyImage {
//!     pixels: Vec<[u8; 3]>,
//!     width: usize,
//!     height: usize,
//! }
//!
//! impl ToLinearRgb for MyImage {
//!     fn to_linear_rgb(&self) -> LinearRgbImage {
//!         let data: Vec<[f32; 3]> = self.pixels.iter()
//!             .map(|[r, g, b]| [
//!                 srgb_u8_to_linear(*r),
//!                 srgb_u8_to_linear(*g),
//!                 srgb_u8_to_linear(*b),
//!             ])
//!             .collect();
//!         LinearRgbImage::new(data, self.width, self.height)
//!     }
//! }
//! ```
//!
//! Helper functions for sRGB conversion:
//! - [`srgb_u8_to_linear`] - 8-bit lookup table (fastest)
//! - [`srgb_u16_to_linear`] - 16-bit conversion
//! - [`srgb_to_linear`] - General f32 conversion
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
//! # let source = fast_ssim2::LinearRgbImage::new(vec![[0.0; 3]; 64], 8, 8);
//! # let distorted = fast_ssim2::LinearRgbImage::new(vec![[0.0; 3]; 64], 8, 8);
//! let score = compute_ssimulacra2_with_config(
//!     source,
//!     distorted,
//!     Ssimulacra2Config::scalar(), // or ::simd()
//! )?;
//! # Ok::<(), fast_ssim2::Ssimulacra2Error>(())
//! ```
//!
//! ## Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `imgref` | | Support for `imgref` image types |
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
// Reference data for parity testing (hidden from docs but accessible for tests)
#[doc(hidden)]
pub mod reference_data;
mod strip;
mod weights;

pub use input::{LinearRgbImage, LinearRgbImageError, ToLinearRgb};
pub use precompute::Ssimulacra2Reference;
pub use strip::{
    HALO_ROWS_DEFAULT, MIN_STRIP_HEIGHT, Ssimulacra2StripConfig, compute_ssimulacra2_strip,
    compute_ssimulacra2_strip_with_config, compute_ssimulacra2_strip_with_stop,
};

// Re-export sRGB conversion functions for users implementing custom input types
pub use input::{srgb_to_linear, srgb_u8_to_linear, srgb_u16_to_linear};

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
#[derive(Debug, Clone, Copy, Default)]
pub struct Ssimulacra2Config {
    /// Kernel backend for all operations.
    pub impl_type: SimdImpl,
}

impl Ssimulacra2Config {
    /// Create configuration with specified implementation.
    pub fn new(impl_type: SimdImpl) -> Self {
        Self { impl_type }
    }

    /// Default configuration using SIMD kernels.
    pub fn simd() -> Self {
        Self::new(SimdImpl::Simd)
    }

    /// Scalar configuration — the reference-order oracle kernels.
    pub fn scalar() -> Self {
        Self::new(SimdImpl::Scalar)
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
    /// The conversion from input image to [`yuvxyb::LinearRgb`] (via [TryFrom]) returned an [Err].
    #[error("Failed to convert input image to linear RGB")]
    LinearRgbConversionFailed,

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


/// Computes the SSIMULACRA2 score from any input type implementing [`ToLinearRgb`].
///
/// This is the recommended API for new code. It supports:
/// - `imgref` types (with the `imgref` feature): `ImgRef<[u8; 3]>`, `ImgRef<[f32; 3]>`, etc.
/// - `yuvxyb` types: `Rgb`, `LinearRgb`
/// - Custom types implementing [`ToLinearRgb`]
///
/// # Color space conventions
/// - Integer types (`u8`, `u16`) are assumed to be sRGB (gamma-encoded)
/// - Float types (`f32`) are assumed to be linear RGB
/// - Grayscale types are expanded to RGB (R=G=B)
///
/// # Example
/// ```ignore
/// use imgref::ImgVec;
/// use fast_ssim2::compute_ssimulacra2;
///
/// let source: ImgVec<[u8; 3]> = /* ... */;
/// let distorted: ImgVec<[u8; 3]> = /* ... */;
/// let score = compute_ssimulacra2(&source, &distorted)?;
/// ```
pub fn compute_ssimulacra2<S, D>(source: S, distorted: D) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    compute_ssimulacra2_with_config(source, distorted, Ssimulacra2Config::default())
}

/// Computes the SSIMULACRA2 score with cooperative cancellation.
///
/// Identical to [`compute_ssimulacra2`] but takes a [`enough::Stop`]
/// token. The token is checked once at the top of each multi-scale
/// outer-loop iteration (never inside the per-pixel inner loops), so
/// cancellation is responsive at scale granularity without adding any
/// cost to the hot path. On cancellation the function returns
/// [`Ssimulacra2Error::Cancelled`].
///
/// Pass [`enough::Unstoppable`] for the never-cancel path, which is
/// indistinguishable in cost from [`compute_ssimulacra2`].
pub fn compute_ssimulacra2_with_stop<S, D>(
    source: S,
    distorted: D,
    stop: &dyn enough::Stop,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    compute_ssimulacra2_with_config_and_stop(source, distorted, Ssimulacra2Config::default(), stop)
}

/// Computes the SSIMULACRA2 score with custom configuration from [`ToLinearRgb`] inputs.
pub fn compute_ssimulacra2_with_config<S, D>(
    source: S,
    distorted: D,
    config: Ssimulacra2Config,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    compute_ssimulacra2_with_config_and_stop(source, distorted, config, &enough::Unstoppable)
}

/// Computes the SSIMULACRA2 score with custom configuration and
/// cooperative cancellation from [`ToLinearRgb`] inputs.
///
/// See [`compute_ssimulacra2_with_stop`] for the cancellation semantics.
fn compute_ssimulacra2_with_config_and_stop<S, D>(
    source: S,
    distorted: D,
    config: Ssimulacra2Config,
    stop: &dyn enough::Stop,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    let kernel = pipeline::Kernel::from_impl(config.impl_type);
    compute_pair(source, distorted, kernel, stop)
}

/// Core pair-scoring entry: the reference SSIMULACRA2.1 pipeline
/// reproduced bit-for-bit.
///
/// Inputs that carry encoded sRGB data (`to_encoded_srgb`) are
/// linearized through the captured reference LUTs; already-linear
/// inputs are used as-is (the reference binary never sees such inputs,
/// so bit-exactness there is undefined).
fn compute_pair<S, D>(
    source: S,
    distorted: D,
    kernel: pipeline::Kernel,
    stop: &dyn enough::Stop,
) -> Result<f64, Ssimulacra2Error>
where
    S: ToLinearRgb,
    D: ToLinearRgb,
{
    let enc1 = source.to_encoded_srgb();
    let enc2 = distorted.to_encoded_srgb();
    if let (Some(e1), Some(e2)) = (&enc1, &enc2) {
        // Sub-8px inputs: the reference binary refuses them, but the
        // crate's `compute_ssimulacra2` contract scores down to 1×1 via
        // mirror padding — apply it on encoded planes (per-pixel LUT ⇒
        // identical to padding post-linearization, and stays U8-exact).
        let p1 = e1.reflect_padded(8);
        let p2 = e2.reflect_padded(8);
        return pipeline::compute_encoded_stop(&p1, &p2, kernel, stop);
    }

    fn planes<T: ToLinearRgb>(t: &T) -> ([Vec<f32>; 3], usize, usize) {
        let img = t.to_linear_rgb();
        let (w, h) = (img.width(), img.height());
        let mut p = [
            Vec::with_capacity(w * h),
            Vec::with_capacity(w * h),
            Vec::with_capacity(w * h),
        ];
        for px in img.data() {
            p[0].push(px[0]);
            p[1].push(px[1]);
            p[2].push(px[2]);
        }
        (p, w, h)
    }

    let (lin1, w1, h1) = planes(&source);
    let (lin2, w2, h2) = planes(&distorted);
    if w1 != w2 || h1 != h2 {
        return Err(Ssimulacra2Error::NonMatchingImageDimensions);
    }
    // Same sub-8px crate contract for the linear-input fallback: pad
    // planes via the encoded-pad equivalent on linear data (per-pixel
    // semantics already applied by to_linear_rgb).
    if w1 < 8 || h1 < 8 {
        let pw = w1.max(8);
        let ph = h1.max(8);
        let pad = |p: [Vec<f32>; 3]| {
            p.map(|data| {
                let mut out = Vec::with_capacity(pw * ph);
                for y in 0..ph {
                    let row = reflect_index(y, h1) * w1;
                    for x in 0..pw {
                        out.push(data[row + reflect_index(x, w1)]);
                    }
                }
                out
            })
        };
        return pipeline::compute_planar_stop(
            pad(lin1),
            pad(lin2),
            pw,
            ph,
            pipeline::Opts { kernel },
            stop,
        );
    }
    pipeline::compute_planar_stop(
        lin1,
        lin2,
        w1,
        h1,
        pipeline::Opts { kernel: pipeline::Kernel::Simd },
        stop,
    )
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

/// Reflect(mirror)-pad a [`LinearRgbImage`] up to `min` px on each axis
/// so SSIMULACRA2's multi-scale pyramid can form on images below the 8px
/// floor. Returns the input unchanged when already ≥ `min` on both axes
/// (or empty — that falls through to the `InvalidImageSize` check). The
/// original pixels occupy the top-left `w × h` region of the result.
pub(crate) fn reflect_pad_linear(img: LinearRgbImage, min: usize) -> LinearRgbImage {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return img;
    }
    let (pw, ph) = (w.max(min), h.max(min));
    if pw == w && ph == h {
        return img;
    }
    let src = img.data();
    let mut out = Vec::with_capacity(pw * ph);
    for y in 0..ph {
        let row = reflect_index(y, h) * w;
        for x in 0..pw {
            out.push(src[row + reflect_index(x, w)]);
        }
    }
    LinearRgbImage::new(out, pw, ph)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::path::PathBuf;

    use super::*;
    use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

    fn tank_rgb() -> (Rgb, Rgb) {
        let mk = |name: &str| {
            let img = image::open(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("test_data")
                    .join(name),
            )
            .unwrap();
            let data = img
                .to_rgb32f()
                .as_chunks::<3>()
                .0
                .to_vec();
            Rgb::new(
                data,
                NonZeroUsize::new(img.width() as usize).unwrap(),
                NonZeroUsize::new(img.height() as usize).unwrap(),
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap()
        };
        (mk("tank_source.png"), mk("tank_distorted.png"))
    }

    #[test]
    fn test_ssimulacra2() {
        let (source, distorted) = tank_rgb();
        let score = compute_ssimulacra2(source, distorted).unwrap();
        assert!(
            (17.0..=18.0).contains(&score),
            "tank score {score} outside expected band (ref-decoder ~17.4)"
        );
    }

    #[cfg(feature = "imgref")]
    #[test]
    fn test_u8_and_on_grid_f32_agree() {
        // The captured-LUT u8 path and `EncodedData::F32` values on the
        // u8 grid (k/255) must produce bit-identical scores — the grid
        // snap is what keeps u8-widened-to-f32 callers exact.
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
            let px: Vec<[u8; 3]> = img.pixels().map(|p| [p[0], p[1], p[2]]).collect();
            imgref::ImgVec::new(px, w as usize, h as usize)
        };
        let (a8, b8) = (mk(&s), mk(&d));
        let mkr = |img: &image::RgbImage| {
            let data: Vec<[f32; 3]> = img
                .pixels()
                .map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
                .collect();
            Rgb::new(
                data,
                NonZeroUsize::new(w as usize).unwrap(),
                NonZeroUsize::new(h as usize).unwrap(),
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap()
        };
        let score_u8 = compute_ssimulacra2(a8.as_ref(), b8.as_ref()).unwrap();
        let score_f32 = compute_ssimulacra2(mkr(&s), mkr(&d)).unwrap();
        assert_eq!(
            score_u8, score_f32,
            "u8 {score_u8} vs on-grid f32 {score_f32} differ — grid snap broken"
        );
    }

    /// Construct a `yuvxyb::LinearRgb` of the requested dimensions filled
    /// with mid-gray.
    fn make_linear_rgb(width: usize, height: usize) -> yuvxyb::LinearRgb {
        let data = vec![[0.5f32, 0.5, 0.5]; width * height];
        yuvxyb::LinearRgb::new(
            data,
            NonZeroUsize::new(width).unwrap(),
            NonZeroUsize::new(height).unwrap(),
        )
        .unwrap()
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
        let score = compute_ssimulacra2_with_config(img.clone(), img, Ssimulacra2Config::default())
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
            let score =
                compute_ssimulacra2_with_config(img.clone(), img, Ssimulacra2Config::default())
                    .unwrap_or_else(|e| panic!("{w}x{h} must score, got {e:?}"));
            assert!(
                (score - 100.0).abs() < 0.01,
                "identical {w}x{h} should score ~100, got {score}"
            );
        }
        // A real sub-8 difference yields a finite score below 100.
        let a = make_linear_rgb(5, 5);
        let b = yuvxyb::LinearRgb::new(
            vec![[0.9f32, 0.1, 0.2]; 25],
            NonZeroUsize::new(5).unwrap(),
            NonZeroUsize::new(5).unwrap(),
        )
        .unwrap();
        let s = compute_ssimulacra2_with_config(a, b, Ssimulacra2Config::default())
            .expect("5x5 differing pair must score");
        assert!(s.is_finite() && s < 100.0, "5x5 differing score {s}");
    }

    #[test]
    fn test_scalar_kernels_bit_identical() {
        let (a, b) = tank_rgb();
        let simd = compute_ssimulacra2_with_config(
            a.clone(),
            b.clone(),
            Ssimulacra2Config::simd(),
        )
        .unwrap();
        let scalar = compute_ssimulacra2_with_config(a, b, Ssimulacra2Config::scalar()).unwrap();
        assert_eq!(simd, scalar, "scalar vs simd kernel mismatch");
    }
}
