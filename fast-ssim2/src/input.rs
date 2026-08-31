//! Input image types and conversion to linear RGB.
//!
//! This module provides the [`ToLinearRgb`] trait for converting various image
//! formats to the internal linear RGB representation used by SSIMULACRA2.
//!
//! ## Supported input formats (with `imgref` feature)
//!
//! | Type | Color Space | Conversion |
//! |------|-------------|------------|
//! | `ImgRef<[u8; 3]>` | sRGB (gamma) | `/255` + linearize |
//! | `ImgRef<[u16; 3]>` | sRGB (gamma) | `/65535` + linearize |
//! | `ImgRef<[f32; 3]>` | Linear RGB | none |
//! | `ImgRef<u8>` | sRGB grayscale | `/255` + linearize + expand |
//! | `ImgRef<f32>` | Linear grayscale | expand to RGB |
//!
//! ## Supported `yuvxyb` input formats (always available)
//!
//! | Type | Fallible? | Conversion |
//! |------|-----------|------------|
//! | [`yuvxyb::LinearRgb`] | no | none |
//! | [`yuvxyb::Rgb`] | yes | transfer function → linear, primaries → BT.709 |
//! | [`yuvxyb::Xyb`] | no | inverse opsin |
//! | [`yuvxyb::Yuv<T>`], `&Yuv<T>` (`T` = `u8`/`u16`) | yes | matrix coefficients → RGB, then as `Rgb` |
//!
//! ## Convention
//!
//! - Integer types (u8, u16) are assumed to be **sRGB** (gamma-encoded)
//! - Float types (f32) are assumed to be **linear**

use crate::Ssimulacra2Error;

/// Internal linear RGB image representation.
///
/// Stores pixels as `[f32; 3]` in linear RGB color space (0.0-1.0 range).
#[derive(Clone, Debug)]
pub struct LinearRgbImage {
    pub(crate) data: Vec<[f32; 3]>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Errors returned by [`LinearRgbImage::try_new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LinearRgbImageError {
    /// `width` or `height` was zero.
    #[error("LinearRgbImage dimensions must be nonzero")]
    ZeroDimension,
    /// `width * height` overflowed `usize`.
    #[error("LinearRgbImage dimensions overflow usize")]
    DimensionOverflow,
    /// `data.len()` did not match `width * height`.
    #[error("LinearRgbImage data length {actual} does not match width * height = {expected}")]
    DataLengthMismatch {
        /// Expected pixel count (`width * height`).
        expected: usize,
        /// Actual `data.len()`.
        actual: usize,
    },
}

impl LinearRgbImage {
    /// Creates a new linear RGB image from raw data.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` is `0`, if `width * height` overflows
    /// `usize`, or if `data.len()` does not equal `width * height`.
    /// For a non-panicking constructor, use [`LinearRgbImage::try_new`].
    pub fn new(data: Vec<[f32; 3]>, width: usize, height: usize) -> Self {
        Self::try_new(data, width, height)
            .expect("LinearRgbImage::new: invalid dimensions or data length")
    }

    /// Fallible constructor for [`LinearRgbImage`].
    ///
    /// Returns `Err` if `width` or `height` is `0`, if `width * height`
    /// overflows `usize`, or if `data.len()` does not equal `width * height`.
    pub fn try_new(
        data: Vec<[f32; 3]>,
        width: usize,
        height: usize,
    ) -> Result<Self, LinearRgbImageError> {
        if width == 0 || height == 0 {
            return Err(LinearRgbImageError::ZeroDimension);
        }
        let expected = width
            .checked_mul(height)
            .ok_or(LinearRgbImageError::DimensionOverflow)?;
        if data.len() != expected {
            return Err(LinearRgbImageError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Returns the image width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the image height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the pixel data.
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }

    /// Returns mutable pixel data.
    pub fn data_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.data
    }
}

/// Build a [`LinearRgbImage`] inside a [`ToLinearRgb`] implementation.
///
/// Every failure mode of [`LinearRgbImage::try_new`] is a dimension problem
/// (zero axis, `width * height` overflow, or a pixel count that disagrees with
/// the stated dimensions), so they all map to
/// [`Ssimulacra2Error::InvalidImageSize`]. Using this instead of
/// [`LinearRgbImage::new`] keeps a zero-sized input — which `imgref` will
/// happily hand us — a recoverable error rather than a panic.
fn build(
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
) -> Result<LinearRgbImage, Ssimulacra2Error> {
    LinearRgbImage::try_new(data, width, height).map_err(|_| Ssimulacra2Error::InvalidImageSize)
}

/// Trait for converting image types to linear RGB.
///
/// Implement this trait to add support for custom image types.
///
/// # Why conversion is fallible
///
/// Conversion is a `Result` because it genuinely fails for some inputs. A
/// [`yuvxyb::Yuv`] frame or a [`yuvxyb::Rgb`] image carries caller-supplied
/// matrix coefficients, transfer characteristics and primaries; `yuvxyb` has
/// no conversion for several spec-legal values (H.273 MC=12
/// `ChromaticityDerivedNonConstantLuminance`, TC=17 `ST428`), so for those the
/// conversion has nothing to return. That is ordinary user input, not a
/// programming error, so it must be recoverable. Making the fallible form the
/// required method means no implementation is ever tempted to turn a
/// recoverable `Err` into a panic — which is what the infallible
/// `to_linear_rgb` it replaced did, via `.expect()`, for any `Rgb` with an
/// unsupported transfer function.
///
/// Implementations whose conversion cannot fail simply return `Ok`.
///
/// Override [`try_into_linear_rgb`](ToLinearRgb::try_into_linear_rgb) for
/// owned types that can convert in-place without allocating a new pixel
/// buffer.
///
/// # Example
///
/// ```
/// use fast_ssim2::{LinearRgbImage, Ssimulacra2Error, ToLinearRgb, srgb_u8_to_linear};
///
/// struct MyImage {
///     pixels: Vec<u8>, // RGB8, tightly packed
///     width: usize,
///     height: usize,
/// }
///
/// impl ToLinearRgb for MyImage {
///     fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
///         let data = self
///             .pixels
///             .chunks_exact(3)
///             .map(|c| {
///                 [
///                     srgb_u8_to_linear(c[0]),
///                     srgb_u8_to_linear(c[1]),
///                     srgb_u8_to_linear(c[2]),
///                 ]
///             })
///             .collect();
///         LinearRgbImage::try_new(data, self.width, self.height)
///             .map_err(|_| Ssimulacra2Error::InvalidImageSize)
///     }
/// }
/// ```
pub trait ToLinearRgb {
    /// Convert to linear RGB image (borrowing).
    ///
    /// # Errors
    ///
    /// Returns [`Ssimulacra2Error::LinearRgbConversionFailed`] if the input's
    /// color signaling cannot be converted to linear RGB, or
    /// [`Ssimulacra2Error::InvalidImageSize`] if it has a zero dimension.
    fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error>;

    /// Convert to linear RGB image, consuming self.
    ///
    /// The default implementation calls
    /// [`try_to_linear_rgb`](ToLinearRgb::try_to_linear_rgb). Override this
    /// for owned types that can reuse their pixel buffer to avoid allocation.
    ///
    /// # Errors
    ///
    /// Same as [`try_to_linear_rgb`](ToLinearRgb::try_to_linear_rgb).
    fn try_into_linear_rgb(self) -> Result<LinearRgbImage, Ssimulacra2Error>
    where
        Self: Sized,
    {
        self.try_to_linear_rgb()
    }
}

/// Identity implementation for already-converted images.
impl ToLinearRgb for LinearRgbImage {
    fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        Ok(self.clone())
    }

    fn try_into_linear_rgb(self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        Ok(self)
    }
}

// =============================================================================
// sRGB conversion functions
// =============================================================================

/// Convert sRGB (gamma-encoded) value to linear.
///
/// Uses a degree-4/4 rational polynomial approximation matching libjxl's
/// `TF_SRGB::DisplayFromEncoded`. Coefficients computed via `af_cheb_rational`
/// (k=100), approximation error ~5e-7. Evaluated with Horner's scheme using
/// FMA to match HWY's `EvalRationalPolynomial`.
#[inline]
pub fn srgb_to_linear(s: f32) -> f32 {
    const THRESH: f32 = 0.04045;
    const LOW_DIV_INV: f32 = 1.0 / 12.92;

    // Rational polynomial coefficients from libjxl TF_SRGB
    const P: [f32; 5] = [
        2.200_248_3e-4,
        1.043_637_6e-2,
        1.624_820_4e-1,
        7.961_565e-1,
        8.210_153e-1,
    ];
    const Q: [f32; 5] = [
        2.631_847e-1,
        1.076_976_5,
        4.987_528_3e-1,
        -5.512_498_3e-2,
        6.521_209e-3,
    ];

    let x = s.abs();
    if x <= THRESH {
        x * LOW_DIV_INV
    } else {
        // Horner's: p[4]*x^4 + p[3]*x^3 + p[2]*x^2 + p[1]*x + p[0]
        let num = P[4]
            .mul_add(x, P[3])
            .mul_add(x, P[2])
            .mul_add(x, P[1])
            .mul_add(x, P[0]);
        let den = Q[4]
            .mul_add(x, Q[3])
            .mul_add(x, Q[2])
            .mul_add(x, Q[1])
            .mul_add(x, Q[0]);
        num / den
    }
}

/// Convert 8-bit sRGB value to linear f32.
#[inline]
pub fn srgb_u8_to_linear(v: u8) -> f32 {
    // Use lookup table for performance
    SRGB_TO_LINEAR_LUT[v as usize]
}

/// Convert 16-bit sRGB value to linear f32.
#[inline]
pub fn srgb_u16_to_linear(v: u16) -> f32 {
    srgb_to_linear(v as f32 / 65535.0)
}

// Precomputed lookup table for sRGB u8 -> linear f32
// Generated with: (0..256).map(|i| srgb_to_linear(i as f32 / 255.0))
static SRGB_TO_LINEAR_LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = srgb_to_linear(i as f32 / 255.0);
    }
    lut
});

// =============================================================================
// imgref implementations
// =============================================================================

#[cfg(feature = "imgref")]
mod imgref_impl {
    use super::*;
    use imgref::ImgRef;

    /// RGB u8 (sRGB) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, [u8; 3]> {
        fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
            let data: Vec<[f32; 3]> = self
                .pixels()
                .map(|[r, g, b]| {
                    [
                        srgb_u8_to_linear(r),
                        srgb_u8_to_linear(g),
                        srgb_u8_to_linear(b),
                    ]
                })
                .collect();
            build(data, self.width(), self.height())
        }
    }

    /// RGB u16 (sRGB) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, [u16; 3]> {
        fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
            let data: Vec<[f32; 3]> = self
                .pixels()
                .map(|[r, g, b]| {
                    [
                        srgb_u16_to_linear(r),
                        srgb_u16_to_linear(g),
                        srgb_u16_to_linear(b),
                    ]
                })
                .collect();
            build(data, self.width(), self.height())
        }
    }

    /// RGB f32 (already linear) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, [f32; 3]> {
        fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
            let data: Vec<[f32; 3]> = self.pixels().collect();
            build(data, self.width(), self.height())
        }
    }

    /// Grayscale u8 (sRGB) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, u8> {
        fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
            let data: Vec<[f32; 3]> = self
                .pixels()
                .map(|v| {
                    let l = srgb_u8_to_linear(v);
                    [l, l, l]
                })
                .collect();
            build(data, self.width(), self.height())
        }
    }

    /// Grayscale f32 (linear) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, f32> {
        fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
            let data: Vec<[f32; 3]> = self.pixels().map(|v| [v, v, v]).collect();
            build(data, self.width(), self.height())
        }
    }
}

// =============================================================================
// yuvxyb compatibility
// =============================================================================

impl ToLinearRgb for yuvxyb::LinearRgb {
    fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        build(
            self.data().to_vec(),
            self.width().get(),
            self.height().get(),
        )
    }

    fn try_into_linear_rgb(self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        let width = self.width().get();
        let height = self.height().get();
        build(self.into_data(), width, height)
    }
}

// =============================================================================
// Conversion to yuvxyb::LinearRgb (for internal pipeline)
// =============================================================================

impl From<LinearRgbImage> for yuvxyb::LinearRgb {
    fn from(img: LinearRgbImage) -> Self {
        use std::num::NonZeroUsize;
        // `LinearRgbImage::try_new` enforces nonzero dimensions and
        // `data.len() == width * height`, so the conversions below cannot fail.
        // We assert defensively in case `LinearRgbImage` was constructed
        // without going through the validated constructor (e.g., by an
        // internal `pub(crate)` field assignment that bypassed validation).
        let width = NonZeroUsize::new(img.width)
            .expect("LinearRgbImage width is nonzero (try_new invariant)");
        let height = NonZeroUsize::new(img.height)
            .expect("LinearRgbImage height is nonzero (try_new invariant)");
        assert_eq!(
            img.data.len(),
            width.get().saturating_mul(height.get()),
            "LinearRgbImage data length must equal width * height (try_new invariant)"
        );
        yuvxyb::LinearRgb::new(img.data, width, height)
            .expect("LinearRgbImage dimensions are valid (try_new invariant)")
    }
}

impl ToLinearRgb for yuvxyb::Rgb {
    fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        if self.transfer() == yuvxyb::TransferCharacteristic::SRGB {
            // Use our own IEC 61966-2-1 sRGB linearization (standard constants)
            // instead of yuvxyb's smoothed variant, for C++ ssimulacra2 parity.
            let data: Vec<[f32; 3]> = self
                .data()
                .iter()
                .map(|&[r, g, b]| [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)])
                .collect();
            build(data, self.width().get(), self.height().get())
        } else {
            // For any other transfer, defer to yuvxyb. It rejects transfer
            // characteristics and primaries it has no conversion for
            // (`Reserved`, and anything the primaries table does not cover),
            // which is caller-supplied signaling, so the error is returned
            // rather than unwrapped.
            let linear = yuvxyb::LinearRgb::try_from(self.clone())
                .map_err(|_| Ssimulacra2Error::LinearRgbConversionFailed)?;
            linear.try_into_linear_rgb()
        }
    }

    fn try_into_linear_rgb(self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        let width = self.width().get();
        let height = self.height().get();
        if self.transfer() == yuvxyb::TransferCharacteristic::SRGB {
            // Consume the Rgb, linearize in-place — zero allocation
            let mut data = self.into_data();
            for pixel in &mut data {
                pixel[0] = srgb_to_linear(pixel[0]);
                pixel[1] = srgb_to_linear(pixel[1]);
                pixel[2] = srgb_to_linear(pixel[2]);
            }
            build(data, width, height)
        } else {
            let linear = yuvxyb::LinearRgb::try_from(self)
                .map_err(|_| Ssimulacra2Error::LinearRgbConversionFailed)?;
            linear.try_into_linear_rgb()
        }
    }
}

/// XYB -> Linear RGB. Infallible (yuvxyb models it as [`From`]).
impl ToLinearRgb for yuvxyb::Xyb {
    fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        yuvxyb::LinearRgb::from(self.clone()).try_into_linear_rgb()
    }

    fn try_into_linear_rgb(self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        yuvxyb::LinearRgb::from(self).try_into_linear_rgb()
    }
}

/// YUV -> Linear RGB, shared by the owned and borrowed impls.
///
/// Routed through `yuvxyb::LinearRgb` (matrix coefficients → RGB, then that
/// `Rgb`'s transfer function → linear) rather than through this crate's
/// [`ToLinearRgb`] impl for [`yuvxyb::Rgb`], which substitutes our own sRGB
/// linearization for C++ parity. Keeping yuvxyb's path here makes the scores
/// for YUV input bit-identical to what the removed `compute_frame_ssimulacra2`
/// produced, so 0.9.0 is an API change and not a metric change. Callers who
/// want the parity linearization can convert `Yuv` → `Rgb` themselves and pass
/// the `Rgb`.
fn yuv_to_linear_rgb<T: yuvxyb::Pixel>(
    yuv: &yuvxyb::Yuv<T>,
) -> Result<LinearRgbImage, Ssimulacra2Error> {
    let linear = yuvxyb::LinearRgb::try_from(yuv)
        .map_err(|_| Ssimulacra2Error::LinearRgbConversionFailed)?;
    linear.try_into_linear_rgb()
}

/// YUV -> Linear RGB.
///
/// Fails with [`Ssimulacra2Error::LinearRgbConversionFailed`] when
/// `yuvxyb` has no conversion for the frame's
/// [`MatrixCoefficients`](yuvxyb::MatrixCoefficients),
/// [`TransferCharacteristic`](yuvxyb::TransferCharacteristic), or
/// [`ColorPrimaries`](yuvxyb::ColorPrimaries) — for example H.273 MC=12
/// (`ChromaticityDerivedNonConstantLuminance`) or TC=17 (`ST428`, digital
/// cinema). Those are values a decoder hands you off a real bitstream, so
/// they are an error, not a panic.
impl<T: yuvxyb::Pixel> ToLinearRgb for yuvxyb::Yuv<T> {
    fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        yuv_to_linear_rgb(self)
    }
}

/// Borrowed YUV -> Linear RGB. The conversion reads the frame either way, so
/// this costs nothing over the owned impl and saves callers a clone.
impl<T: yuvxyb::Pixel> ToLinearRgb for &yuvxyb::Yuv<T> {
    fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
        yuv_to_linear_rgb(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_linear_bounds() {
        assert!((srgb_to_linear(0.0) - 0.0).abs() < 1e-6);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_srgb_to_linear_midpoint() {
        // sRGB 0.5 should be approximately 0.214 in linear
        let linear = srgb_to_linear(0.5);
        assert!((linear - 0.214).abs() < 0.01);
    }

    #[test]
    fn test_srgb_u8_to_linear() {
        assert!((srgb_u8_to_linear(0) - 0.0).abs() < 1e-6);
        assert!((srgb_u8_to_linear(255) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_rgb_image_accessors() {
        let data = vec![[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]];
        let img = LinearRgbImage::new(data.clone(), 2, 1);

        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 1);
        assert_eq!(img.data(), &data[..]);
    }

    #[test]
    fn test_try_new_rejects_zero_dimension() {
        let err = LinearRgbImage::try_new(vec![], 0, 4).unwrap_err();
        assert_eq!(err, LinearRgbImageError::ZeroDimension);
        let err = LinearRgbImage::try_new(vec![], 4, 0).unwrap_err();
        assert_eq!(err, LinearRgbImageError::ZeroDimension);
    }

    #[test]
    fn test_try_new_rejects_dimension_overflow() {
        // usize::MAX * 2 always overflows on every supported target.
        let err = LinearRgbImage::try_new(vec![], usize::MAX, 2).unwrap_err();
        assert_eq!(err, LinearRgbImageError::DimensionOverflow);
    }

    #[test]
    fn test_try_new_rejects_data_length_mismatch() {
        let err = LinearRgbImage::try_new(vec![[0.0; 3]; 3], 2, 2).unwrap_err();
        assert!(matches!(
            err,
            LinearRgbImageError::DataLengthMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn test_try_new_accepts_valid_input() {
        let img = LinearRgbImage::try_new(vec![[0.5, 0.3, 0.1]; 6], 3, 2).unwrap();
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 2);
    }

    #[test]
    #[should_panic(expected = "LinearRgbImage::new: invalid dimensions or data length")]
    fn test_new_panics_on_zero_dimension_in_release() {
        // This panic now fires in release as well as debug builds — previously
        // only `debug_assert_eq!` validated, so release-mode misuse silently
        // produced a malformed image that would later panic deep in
        // `From<LinearRgbImage> for yuvxyb::LinearRgb`.
        let _ = LinearRgbImage::new(vec![], 0, 4);
    }

    #[test]
    fn test_yuvxyb_linearrgb_roundtrip() {
        use std::num::NonZeroUsize;
        let data = vec![[0.5, 0.3, 0.1]; 4];
        let yuvxyb_img = yuvxyb::LinearRgb::new(
            data.clone(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        )
        .expect("valid dimensions");

        let our_img = yuvxyb_img.try_to_linear_rgb().unwrap();
        assert_eq!(our_img.width(), 2);
        assert_eq!(our_img.height(), 2);
        assert_eq!(our_img.data(), &data[..]);

        // Convert back
        let back: yuvxyb::LinearRgb = our_img.into();
        assert_eq!(back.data(), &data[..]);
    }
}

#[cfg(test)]
mod yuv_tests {
    use super::*;
    use yuvxyb::{
        ChromaSubsampling, ColorPrimaries, Frame, FrameBuilder, MatrixCoefficients,
        TransferCharacteristic, Yuv, YuvConfig,
    };

    /// A 16x16 4:4:4 8-bit YUV frame with a deterministic ramp, tagged with
    /// the caller's matrix coefficients.
    fn make_yuv(mc: MatrixCoefficients) -> Yuv<u8> {
        let dim = std::num::NonZeroUsize::new(16).unwrap();
        let bit_depth = std::num::NonZeroU8::new(8).unwrap();
        let mut data: Frame<u8> = FrameBuilder::new(dim, dim, ChromaSubsampling::Yuv444, bit_depth)
            .build::<u8>()
            .unwrap();
        for (i, val) in data.y_plane.pixels_mut().enumerate() {
            *val = (i * 7 % 220 + 16) as u8;
        }
        for plane in [data.u_plane.as_mut(), data.v_plane.as_mut()]
            .into_iter()
            .flatten()
        {
            for (i, val) in plane.pixels_mut().enumerate() {
                *val = (i * 11 % 200 + 20) as u8;
            }
        }
        Yuv::new(
            data,
            YuvConfig {
                bit_depth: 8,
                subsampling_x: 0,
                subsampling_y: 0,
                full_range: true,
                matrix_coefficients: mc,
                transfer_characteristics: TransferCharacteristic::SRGB,
                color_primaries: ColorPrimaries::BT709,
            },
        )
        .unwrap()
    }

    /// The YUV path must reproduce `yuvxyb`'s own `Yuv -> LinearRgb`
    /// conversion bit-for-bit. That is what the removed
    /// `compute_frame_ssimulacra2` did, so this pins 0.9.0 as an API change
    /// rather than a metric change — and it fires if someone later reroutes
    /// `Yuv` through the `Rgb` impl, which substitutes our own sRGB
    /// linearization.
    #[test]
    fn yuv_conversion_matches_yuvxyb_bit_for_bit() {
        let yuv = make_yuv(MatrixCoefficients::BT709);

        let ours = yuv.try_to_linear_rgb().expect("BT709 YUV converts");
        let theirs = yuvxyb::LinearRgb::try_from(&yuv).expect("BT709 YUV converts");

        assert_eq!(ours.width(), theirs.width().get());
        assert_eq!(ours.height(), theirs.height().get());
        assert_eq!(ours.data(), theirs.data());
    }

    /// Route through a generic bound, which is the only way to select the
    /// `&Yuv<T>` impl — method-call syntax on a `&Yuv<T>` receiver autorefs
    /// into the *owned* impl, so `(&yuv).try_to_linear_rgb()` would not test
    /// this at all. `compute_ssimulacra2(&yuv, ...)` monomorphises exactly
    /// like `via_bound` does.
    fn via_bound<T: ToLinearRgb>(input: T) -> Result<LinearRgbImage, Ssimulacra2Error> {
        input.try_into_linear_rgb()
    }

    /// The borrowed impl exists so callers need not clone a frame; it must be
    /// indistinguishable from the owned one.
    #[test]
    fn borrowed_yuv_matches_owned_yuv() {
        let yuv = make_yuv(MatrixCoefficients::BT709);

        let borrowed = via_bound(&yuv).expect("BT709 YUV converts");
        let owned = via_bound(yuv).expect("BT709 YUV converts");

        assert_eq!(borrowed.data(), owned.data());
    }

    /// The reason `ToLinearRgb` is fallible. H.273 MC=12
    /// (`ChromaticityDerivedNonConstantLuminance`) and MC=3 (`Reserved`) are
    /// values a decoder can hand you off a real bitstream, and `yuvxyb` has no
    /// YUV->RGB matrix for either, so conversion must return `Err` rather than
    /// panicking.
    #[test]
    fn unsupported_matrix_coefficients_are_an_error_not_a_panic() {
        for mc in [
            MatrixCoefficients::ChromaticityDerivedNonConstantLuminance,
            MatrixCoefficients::Reserved,
        ] {
            let yuv = make_yuv(mc);
            assert_eq!(
                yuv.try_to_linear_rgb().unwrap_err(),
                Ssimulacra2Error::LinearRgbConversionFailed,
                "owned Yuv with {mc:?}"
            );
            assert_eq!(
                via_bound(&yuv).unwrap_err(),
                Ssimulacra2Error::LinearRgbConversionFailed,
                "borrowed Yuv with {mc:?}"
            );
        }
    }

    /// The same failure must surface from the public entry point rather than
    /// aborting the process.
    #[test]
    fn compute_ssimulacra2_propagates_yuv_conversion_failure() {
        let yuv = make_yuv(MatrixCoefficients::ChromaticityDerivedNonConstantLuminance);

        assert_eq!(
            crate::compute_ssimulacra2(&yuv, &yuv).unwrap_err(),
            Ssimulacra2Error::LinearRgbConversionFailed
        );
    }

    /// The `Rgb` impl had the same reachable panic before 0.9.0: its non-sRGB
    /// arm unwrapped `LinearRgb::try_from` with
    /// `.expect("... should not fail")`. It does fail — H.273 TC=17 (`ST428`,
    /// digital cinema) has no `to_linear` in `yuvxyb` — so an ST428-tagged
    /// image aborted the process instead of returning an error.
    #[test]
    fn unsupported_transfer_on_rgb_is_an_error_not_a_panic() {
        let dim = std::num::NonZeroUsize::new(16).unwrap();
        let rgb = yuvxyb::Rgb::new(
            vec![[0.5, 0.4, 0.3]; 16 * 16],
            dim,
            dim,
            TransferCharacteristic::ST428,
            ColorPrimaries::BT709,
        )
        .unwrap();

        assert_eq!(
            rgb.try_to_linear_rgb().unwrap_err(),
            Ssimulacra2Error::LinearRgbConversionFailed
        );
        assert_eq!(
            rgb.try_into_linear_rgb().unwrap_err(),
            Ssimulacra2Error::LinearRgbConversionFailed
        );
    }

    /// End to end: a supported YUV pair scores, and scores 100 against itself.
    #[test]
    fn identical_yuv_frames_score_100() {
        let yuv = make_yuv(MatrixCoefficients::BT709);
        let score = crate::compute_ssimulacra2(&yuv, &yuv).expect("BT709 YUV scores");
        assert!(
            (score - 100.0).abs() < 0.01,
            "identical YUV frames should score 100, got {score}"
        );
    }
}

#[cfg(all(test, feature = "imgref"))]
mod imgref_tests {
    use super::*;
    use imgref::{Img, ImgVec};

    #[test]
    fn test_imgref_u8_srgb_conversion() {
        // Create a 2x2 sRGB image
        let pixels: Vec<[u8; 3]> = vec![
            [0, 0, 0],       // black
            [255, 255, 255], // white
            [128, 128, 128], // mid gray
            [255, 0, 0],     // red
        ];
        let img: ImgVec<[u8; 3]> = Img::new(pixels, 2, 2);

        let linear = img.as_ref().try_to_linear_rgb().unwrap();
        assert_eq!(linear.width(), 2);
        assert_eq!(linear.height(), 2);

        // Black should be [0, 0, 0]
        assert!((linear.data()[0][0] - 0.0).abs() < 1e-6);
        // White should be [1, 1, 1]
        assert!((linear.data()[1][0] - 1.0).abs() < 1e-6);
        assert!((linear.data()[1][1] - 1.0).abs() < 1e-6);
        // Mid gray (sRGB 128) should be ~0.215 in linear
        assert!((linear.data()[2][0] - 0.215).abs() < 0.01);
        // Red should have R=1, G=B=0
        assert!((linear.data()[3][0] - 1.0).abs() < 1e-6);
        assert!((linear.data()[3][1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_imgref_f32_passthrough() {
        // f32 is assumed to already be linear - should pass through unchanged
        let pixels: Vec<[f32; 3]> = vec![[0.5, 0.3, 0.1], [0.9, 0.8, 0.7]];
        let img: ImgVec<[f32; 3]> = Img::new(pixels.clone(), 2, 1);

        let linear = img.as_ref().try_to_linear_rgb().unwrap();
        assert_eq!(linear.data(), &pixels[..]);
    }

    #[test]
    fn test_imgref_grayscale_u8_expansion() {
        // Grayscale u8 should expand to R=G=B and apply sRGB conversion
        let pixels: Vec<u8> = vec![0, 255, 128];
        let img: ImgVec<u8> = Img::new(pixels, 3, 1);

        let linear = img.as_ref().try_to_linear_rgb().unwrap();

        // Black
        let black = linear.data()[0];
        assert!((black[0] - 0.0).abs() < 1e-6);
        assert_eq!(black[0], black[1]);
        assert_eq!(black[1], black[2]);

        // White
        let white = linear.data()[1];
        assert!((white[0] - 1.0).abs() < 1e-6);
        assert_eq!(white[0], white[1]);

        // Mid gray
        let gray = linear.data()[2];
        assert!((gray[0] - 0.215).abs() < 0.01);
        assert_eq!(gray[0], gray[1]);
    }

    #[test]
    fn test_imgref_grayscale_f32_expansion() {
        // Grayscale f32 should expand to R=G=B (already linear)
        let pixels: Vec<f32> = vec![0.0, 1.0, 0.5];
        let img: ImgVec<f32> = Img::new(pixels, 3, 1);

        let linear = img.as_ref().try_to_linear_rgb().unwrap();

        assert_eq!(linear.data()[0], [0.0, 0.0, 0.0]);
        assert_eq!(linear.data()[1], [1.0, 1.0, 1.0]);
        assert_eq!(linear.data()[2], [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_compute_ssimulacra2_with_imgref_u8() {
        use crate::compute_ssimulacra2;

        // Create two 16x16 images (minimum viable for SSIMULACRA2)
        let pixels1: Vec<[u8; 3]> = vec![[128, 128, 128]; 16 * 16];
        let pixels2: Vec<[u8; 3]> = vec![[130, 130, 130]; 16 * 16]; // slightly different

        let img1: ImgVec<[u8; 3]> = Img::new(pixels1, 16, 16);
        let img2: ImgVec<[u8; 3]> = Img::new(pixels2, 16, 16);

        // Should compute successfully
        let score = compute_ssimulacra2(img1.as_ref(), img2.as_ref()).unwrap();
        // Small difference should result in high score (close to 100)
        assert!(
            score > 90.0,
            "Score {score} should be > 90 for very similar images"
        );
    }

    #[test]
    fn test_compute_ssimulacra2_identical_imgref() {
        use crate::compute_ssimulacra2;

        // Identical images should score 100
        let pixels: Vec<[u8; 3]> = vec![[100, 150, 200]; 16 * 16];
        let img: ImgVec<[u8; 3]> = Img::new(pixels, 16, 16);

        let score = compute_ssimulacra2(img.as_ref(), img.as_ref()).unwrap();
        assert!(
            (score - 100.0).abs() < 0.01,
            "Identical images should score 100, got {score}"
        );
    }

    #[test]
    fn test_precompute_with_imgref() {
        use crate::Ssimulacra2Reference;

        // Create source and distorted images
        let source_pixels: Vec<[u8; 3]> = vec![[128, 128, 128]; 32 * 32];
        let distorted_pixels: Vec<[u8; 3]> = vec![[130, 128, 126]; 32 * 32];

        let source: ImgVec<[u8; 3]> = Img::new(source_pixels, 32, 32);
        let distorted: ImgVec<[u8; 3]> = Img::new(distorted_pixels, 32, 32);

        // Use precompute API with imgref
        let reference = Ssimulacra2Reference::new(source.as_ref()).unwrap();
        let score = reference.compare(distorted.as_ref()).unwrap();

        // Should compute successfully with reasonable score
        assert!(
            score > 80.0,
            "Score {score} should be > 80 for similar images"
        );
    }
}
