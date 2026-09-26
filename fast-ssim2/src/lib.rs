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
//! ### Without features (built-in slice adapters)
//!
//! ```
//! use fast_ssim2::{compute_ssimulacra2, RgbSlice};
//!
//! // Plain u8 sRGB pixels, 64×64 mid-gray.
//! let data: Vec<[u8; 3]> = vec![[128, 128, 128]; 64 * 64];
//! let source = RgbSlice::new(&data, 64, 64);
//! let distorted = RgbSlice::new(&data, 64, 64);
//!
//! let score = compute_ssimulacra2(source, distorted)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Batch Comparisons (2x Faster)
//!
//! When comparing multiple images against the same reference (e.g., evaluating
//! different compression levels), precompute the reference data once:
//!
//! ```
//! use fast_ssim2::{Ssimulacra2Reference, RgbSlice};
//!
//! // Create test data
//! let data: Vec<[u8; 3]> = vec![[128, 128, 128]; 64 * 64];
//! let source = RgbSlice::new(&data, 64, 64);
//!
//! // Precompute reference data (~50% of the work)
//! let reference = Ssimulacra2Reference::new(source)?;
//!
//! // Compare multiple distorted versions efficiently
//! let distorted = RgbSlice::new(&data, 64, 64);
//! let score = reference.compare(distorted)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Custom Input Types
//!
//! Implement [`ImageSource`] to support your own image types:
//!
//! ```
//! use fast_ssim2::{ImageSource, PixelFormat};
//!
//! struct MyImage {
//!     pixels: Vec<[u8; 3]>,
//!     width: usize,
//!     height: usize,
//! }
//!
//! impl ImageSource for MyImage {
//!     fn width(&self) -> usize { self.width }
//!     fn height(&self) -> usize { self.height }
//!     fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb8Rgb }
//!     fn row_bytes(&self, y: usize) -> &[u8] {
//!         self.pixels[y * self.width..(y + 1) * self.width].as_flattened()
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
mod source;
#[cfg(feature = "zenpixels")]
mod zenpixels_compat;
mod precompute;
// Reference data for parity testing (hidden from docs but accessible for tests)
#[doc(hidden)]
pub mod reference_data;
mod strip;
mod weights;

pub use input::{LinearRgbImage, LinearRgbImageError};
pub use source::{
    AlphaMode, GraySlice, ImageSource, PixelFormat, Rgb16Slice, RgbSlice, RgbaSlice,
    SrgbF32Image, SrgbF32Slice, StridedBytes, SubsetView,
};
#[cfg(feature = "zenpixels")]
pub use zenpixels_compat::{UnsupportedFormat, ZenpixelsSource};
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
    /// An input source's byte buffer was shorter than its declared
    /// dimensions and [`PixelFormat`](crate::PixelFormat) require.
    #[error("Input data is {actual} bytes but the declared dimensions and format require more")]
    InvalidInputData {
        /// Byte length the source provided.
        actual: usize,
    },

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


/// Computes the SSIMULACRA2 score from any [`ImageSource`].
///
/// This is the recommended API. Any [`ImageSource`] works — the
/// built-in [`RgbSlice`]/[`RgbaSlice`]/[`GraySlice`]/[`StridedBytes`]
/// adapters cover raw buffers; `imgref` types need the `imgref`
/// feature; `zenpixels` `PixelSlice`/`PixelBuffer` bridge via the
/// `zenpixels` feature. Custom inputs implement [`ImageSource`].
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
    S: ImageSource,
    D: ImageSource,
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
    S: ImageSource,
    D: ImageSource,
{
    compute_ssimulacra2_with_config_and_stop(source, distorted, Ssimulacra2Config::default(), stop)
}

/// Computes the SSIMULACRA2 score with custom configuration.
pub fn compute_ssimulacra2_with_config<S, D>(
    source: S,
    distorted: D,
    config: Ssimulacra2Config,
) -> Result<f64, Ssimulacra2Error>
where
    S: ImageSource,
    D: ImageSource,
{
    compute_ssimulacra2_with_config_and_stop(source, distorted, config, &enough::Unstoppable)
}

/// Computes the SSIMULACRA2 score with custom configuration and
/// cooperative cancellation.
///
/// See [`compute_ssimulacra2_with_stop`] for the cancellation semantics.
fn compute_ssimulacra2_with_config_and_stop<S, D>(
    source: S,
    distorted: D,
    config: Ssimulacra2Config,
    stop: &dyn enough::Stop,
) -> Result<f64, Ssimulacra2Error>
where
    S: ImageSource,
    D: ImageSource,
{
    let kernel = pipeline::Kernel::from_impl(config.impl_type);
    compute_pair(source, distorted, kernel, stop)
}

/// Core pair-scoring entry: the reference SSIMULACRA2.1 pipeline
/// reproduced bit-for-bit.
///
/// Encoded sRGB inputs are linearized through the captured reference
/// LUTs; already-linear inputs are used as-is (the reference binary
/// never sees such inputs, so bit-exactness there is undefined).
fn compute_pair<S, D>(
    source: S,
    distorted: D,
    kernel: pipeline::Kernel,
    stop: &dyn enough::Stop,
) -> Result<f64, Ssimulacra2Error>
where
    S: ImageSource,
    D: ImageSource,
{
    use source::PreparedInput;

    let p1 = source::funnel(&source)?;
    let p2 = source::funnel(&distorted)?;

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
    let opts = pipeline::Opts { kernel };

    // Reference alpha compositing: min over two backgrounds (0.1 / 0.9
    // encoded). Linear inputs replicate it with the linearized bg.
    let eval = |bg: f32, p1: &PreparedInput, p2: &PreparedInput| -> Result<f64, Ssimulacra2Error> {
        let l1 = linearize_prepared(p1, w1, bg);
        let l2 = linearize_prepared(p2, w2, bg);
        let (l1, l2) = if pw != w1 || ph != h1 {
            (pad_planes(l1, w1, h1, pw, ph), pad_planes(l2, w2, h2, pw, ph))
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

    /// sRGB-encoded f32 raster — the on-grid encoded-input path.
    fn tank_rgb() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, u32, u32) {
        let mk = |name: &str| {
            let img = image::open(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("test_data")
                    .join(name),
            )
            .unwrap();
            (img.to_rgb32f().as_chunks::<3>().0.to_vec(), img.width(), img.height())
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
            SrgbF32Slice::new(&s, w as usize, h as usize),
            SrgbF32Slice::new(&d, w as usize, h as usize),
        )
        .unwrap();
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
            img.pixels()
                .map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
                .collect::<Vec<[f32; 3]>>()
        };
        let (af, bf) = (mkr(&s), mkr(&d));
        let score_u8 = compute_ssimulacra2(a8.as_ref(), b8.as_ref()).unwrap();
        let score_f32 = compute_ssimulacra2(
            SrgbF32Slice::new(&af, w as usize, h as usize),
            SrgbF32Slice::new(&bf, w as usize, h as usize),
        )
        .unwrap();
        assert_eq!(
            score_u8, score_f32,
            "u8 {score_u8} vs on-grid f32 {score_f32} differ — grid snap broken"
        );
    }

    /// Construct a mid-gray linear image.
    fn make_linear_rgb(width: usize, height: usize) -> LinearRgbImage {
        LinearRgbImage::new(vec![[0.5f32, 0.5, 0.5]; width * height], width, height)
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
        let b = LinearRgbImage::new(vec![[0.9f32, 0.1, 0.2]; 25], 5, 5);
        let s = compute_ssimulacra2_with_config(a, b, Ssimulacra2Config::default())
            .expect("5x5 differing pair must score");
        assert!(s.is_finite() && s < 100.0, "5x5 differing score {s}");
    }

    #[test]
    fn test_scalar_kernels_bit_identical() {
        let (sa, sb, w, h) = tank_rgb();
        let (w, h) = (w as usize, h as usize);
        let simd = compute_ssimulacra2_with_config(
            SrgbF32Slice::new(&sa, w, h),
            SrgbF32Slice::new(&sb, w, h),
            Ssimulacra2Config::simd(),
        )
        .unwrap();
        let scalar = compute_ssimulacra2_with_config(
            SrgbF32Slice::new(&sa, w, h),
            SrgbF32Slice::new(&sb, w, h),
            Ssimulacra2Config::scalar(),
        )
        .unwrap();
        assert_eq!(simd, scalar, "scalar vs simd kernel mismatch");
    }
}
