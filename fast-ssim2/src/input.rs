//! Input image types: the [`LinearRgbImage`] linear container and the
//! public sRGB↔linear conversion helpers.
//!
//! The image-input surface lives in [`crate::source`] — the
//! [`crate::ImageSource`] trait plus slice/stride adapters. This module
//! holds the owned linear container and the family-standard transfer
//! functions ([`linear_srgb`]).

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

// =============================================================================
// sRGB conversion functions
// =============================================================================

/// Convert sRGB (gamma-encoded) value to linear f32.
///
/// Delegates to [`linear_srgb::default::srgb_to_linear`] — the
/// family-standard C0-continuous rational polynomial (≤14 ULP vs the
/// IEC curve). Quantized input formats don't reach this fn — they go
/// through the captured lcms LUTs in `pipeline::lut8`.
#[inline]
pub fn srgb_to_linear(s: f32) -> f32 {
    linear_srgb::default::srgb_to_linear(s)
}

/// Convert 8-bit sRGB value to linear f32.
#[inline]
pub fn srgb_u8_to_linear(v: u8) -> f32 {
    linear_srgb::default::srgb_u8_to_linear(v)
}

/// Convert 16-bit sRGB value to linear f32.
#[inline]
pub fn srgb_u16_to_linear(v: u16) -> f32 {
    linear_srgb::default::srgb_u16_to_linear(v)
}

