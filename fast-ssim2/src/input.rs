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
//! ## Convention
//!
//! - Integer types (u8, u16) are assumed to be **sRGB** (gamma-encoded)
//! - Float types (f32) are assumed to be **linear**

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

/// Trait for converting image types to linear RGB.
///
/// Implement this trait to add support for custom image types.
///
/// Override [`into_linear_rgb`](ToLinearRgb::into_linear_rgb) for owned types
/// that can convert in-place without allocating a new pixel buffer.
pub trait ToLinearRgb {
    /// Convert to linear RGB image (borrowing).
    fn to_linear_rgb(&self) -> LinearRgbImage;

    /// Convert to linear RGB image, consuming self.
    ///
    /// The default implementation calls [`to_linear_rgb`](ToLinearRgb::to_linear_rgb).
    /// Override this for owned types that can reuse their pixel buffer
    /// to avoid allocation.
    fn into_linear_rgb(self) -> LinearRgbImage
    where
        Self: Sized,
    {
        self.to_linear_rgb()
    }
}

/// Identity implementation for already-converted images.
impl ToLinearRgb for LinearRgbImage {
    fn to_linear_rgb(&self) -> LinearRgbImage {
        self.clone()
    }

    fn into_linear_rgb(self) -> LinearRgbImage {
        self
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
        fn to_linear_rgb(&self) -> LinearRgbImage {
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
            LinearRgbImage::new(data, self.width(), self.height())
        }
    }

    /// RGB u16 (sRGB) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, [u16; 3]> {
        fn to_linear_rgb(&self) -> LinearRgbImage {
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
            LinearRgbImage::new(data, self.width(), self.height())
        }
    }

    /// RGB f32 (already linear) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, [f32; 3]> {
        fn to_linear_rgb(&self) -> LinearRgbImage {
            let data: Vec<[f32; 3]> = self.pixels().collect();
            LinearRgbImage::new(data, self.width(), self.height())
        }
    }

    /// Grayscale u8 (sRGB) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, u8> {
        fn to_linear_rgb(&self) -> LinearRgbImage {
            let data: Vec<[f32; 3]> = self
                .pixels()
                .map(|v| {
                    let l = srgb_u8_to_linear(v);
                    [l, l, l]
                })
                .collect();
            LinearRgbImage::new(data, self.width(), self.height())
        }
    }

    /// Grayscale f32 (linear) -> Linear RGB
    impl ToLinearRgb for ImgRef<'_, f32> {
        fn to_linear_rgb(&self) -> LinearRgbImage {
            let data: Vec<[f32; 3]> = self.pixels().map(|v| [v, v, v]).collect();
            LinearRgbImage::new(data, self.width(), self.height())
        }
    }
}

// =============================================================================
// yuvxyb compatibility
// =============================================================================

impl ToLinearRgb for yuvxyb::LinearRgb {
    fn to_linear_rgb(&self) -> LinearRgbImage {
        LinearRgbImage::new(
            self.data().to_vec(),
            self.width().get(),
            self.height().get(),
        )
    }

    fn into_linear_rgb(self) -> LinearRgbImage {
        let width = self.width().get();
        let height = self.height().get();
        LinearRgbImage::new(self.into_data(), width, height)
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
    fn to_linear_rgb(&self) -> LinearRgbImage {
        if self.transfer() == yuvxyb::TransferCharacteristic::SRGB {
            // Use our own IEC 61966-2-1 sRGB linearization (standard constants)
            // instead of yuvxyb's smoothed variant, for C++ ssimulacra2 parity.
            let data: Vec<[f32; 3]> = self
                .data()
                .iter()
                .map(|&[r, g, b]| [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)])
                .collect();
            LinearRgbImage::new(data, self.width().get(), self.height().get())
        } else {
            // For non-sRGB transfers, fall back to yuvxyb's conversion
            let linear: yuvxyb::LinearRgb = yuvxyb::LinearRgb::try_from(self.clone())
                .expect("Rgb to LinearRgb conversion should not fail");
            linear.to_linear_rgb()
        }
    }

    fn into_linear_rgb(self) -> LinearRgbImage {
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
            LinearRgbImage::new(data, width, height)
        } else {
            let linear: yuvxyb::LinearRgb = yuvxyb::LinearRgb::try_from(self)
                .expect("Rgb to LinearRgb conversion should not fail");
            linear.into_linear_rgb()
        }
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

        let our_img = yuvxyb_img.to_linear_rgb();
        assert_eq!(our_img.width(), 2);
        assert_eq!(our_img.height(), 2);
        assert_eq!(our_img.data(), &data[..]);

        // Convert back
        let back: yuvxyb::LinearRgb = our_img.into();
        assert_eq!(back.data(), &data[..]);
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

        let linear = img.as_ref().to_linear_rgb();
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

        let linear = img.as_ref().to_linear_rgb();
        assert_eq!(linear.data(), &pixels[..]);
    }

    #[test]
    fn test_imgref_grayscale_u8_expansion() {
        // Grayscale u8 should expand to R=G=B and apply sRGB conversion
        let pixels: Vec<u8> = vec![0, 255, 128];
        let img: ImgVec<u8> = Img::new(pixels, 3, 1);

        let linear = img.as_ref().to_linear_rgb();

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

        let linear = img.as_ref().to_linear_rgb();

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
