//! Adapter for `zenpixels` pixel types.
//!
//! [`ZenpixelsSource`] validates a [`PixelSlice`]/[`PixelBuffer`] and
//! implements [`ImageSource`], borrowed zero-copy where possible
//! (premultiplied alpha is un-premultiplied into an owned buffer).
//!
//! # Supported formats
//!
//! | zenpixels format | transfer | result |
//! |------------------|----------|--------|
//! | Rgb8 | sRGB/BT.709 | [`PixelFormat::Srgb8Rgb`], opaque |
//! | Rgba8 / Rgbx8 | sRGB/BT.709 | [`PixelFormat::Srgb8Rgba`] |
//! | Bgra8 / Bgrx8 | sRGB/BT.709 | [`PixelFormat::Srgb8Bgra`] |
//! | Rgb16 | sRGB/BT.709 | [`PixelFormat::Srgb16Rgb`], opaque |
//! | Rgba16 | sRGB/BT.709 | [`PixelFormat::Srgb16Rgba`] |
//! | RgbF32 (sRGB) | sRGB/BT.709 | [`PixelFormat::SrgbF32Rgb`] |
//! | RgbaF32 | Linear | [`PixelFormat::LinearF32Rgba`] |
//! | RgbF32 | Linear | [`PixelFormat::LinearF32Rgb`] |
//!
//! # Rejected
//!
//! - HDR transfers (PQ/HLG) — the metric's SDR pipeline can't score them
//!   honestly; tonemap/convert upstream (`zenpixels-convert`) first.
//! - Narrow (limited) signal range — expand to full range upstream.
//! - Non-BT.709/sRGB primaries — gamut-convert upstream; the reference
//!   metric assumes sRGB primaries.
//! - Grayscale zenpixels formats — wrap raw gray bytes in
//!   [`GraySlice`](crate::GraySlice) instead.
//! - Unknown transfer functions.

use std::borrow::Cow;

use zenpixels::{
    AlphaMode as ZpAlpha, ColorPrimaries as ZpPrimaries, PixelBuffer, PixelDescriptor, PixelSlice,
    TransferFunction, PixelFormat as ZpFormat, SignalRange,
};

use crate::source::{AlphaMode, ImageSource, PixelFormat};
use crate::Ssimulacra2Error;

/// A [`PixelSlice`]/[`PixelBuffer`] couldn't be mapped to an
/// [`ImageSource`] — see the module docs for the supported/rejected lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unsupported zenpixels format: {0}")]
pub struct UnsupportedFormat(pub &'static str);

/// Validated [`ImageSource`] adapter for zenpixels pixel data.
///
/// Create via [`try_from_slice`](Self::try_from_slice) or
/// [`try_from_buffer`](Self::try_from_buffer). Premultiplied alpha is
/// un-premultiplied into an internal buffer (one allocation); every other
/// supported layout is borrowed zero-copy.
pub struct ZenpixelsSource<'a> {
    data: Cow<'a, [u8]>,
    width: usize,
    height: usize,
    stride: usize,
    pixel_format: PixelFormat,
    alpha_mode: AlphaMode,
}

impl<'a> ZenpixelsSource<'a> {
    /// Create from a borrowed [`PixelSlice`].
    ///
    /// # Errors
    /// [`UnsupportedFormat`] for descriptors this metric can't score
    /// honestly (see the module docs).
    pub fn try_from_slice(slice: &'a PixelSlice<'a>) -> Result<Self, UnsupportedFormat> {
        let desc = slice.descriptor();
        let (pixel_format, alpha_mode) = map_descriptor(&desc)?;
        let width = slice.width() as usize;
        let height = slice.rows() as usize;
        let stride = slice.stride();
        let raw = slice.as_strided_bytes();
        let data = if matches!(desc.alpha, Some(ZpAlpha::Premultiplied)) {
            Cow::Owned(unpremultiply(raw, width, height, stride, &desc))
        } else {
            Cow::Borrowed(raw)
        };
        Ok(Self {
            data,
            width,
            height,
            stride,
            pixel_format,
            alpha_mode,
        })
    }

    /// Create from an owned [`PixelBuffer`].
    ///
    /// Same validation as [`try_from_slice`](Self::try_from_slice).
    /// The buffer's backing memory is borrowed for the source's lifetime.
    ///
    /// # Errors
    /// [`UnsupportedFormat`] for non-contiguous buffers or unsupported
    /// descriptors.
    pub fn try_from_buffer(buf: &'a PixelBuffer) -> Result<Self, UnsupportedFormat> {
        let desc = buf.descriptor();
        let (pixel_format, alpha_mode) = map_descriptor(&desc)?;
        let width = buf.width() as usize;
        let height = buf.height() as usize;
        let stride = buf.stride();
        let raw = match buf.as_contiguous_bytes() {
            Some(b) => b,
            None => return Err(UnsupportedFormat("non-contiguous PixelBuffer layout")),
        };
        let data = if matches!(desc.alpha, Some(ZpAlpha::Premultiplied)) {
            Cow::Owned(unpremultiply(raw, width, height, stride, &desc))
        } else {
            Cow::Borrowed(raw)
        };
        Ok(Self {
            data,
            width,
            height,
            stride,
            pixel_format,
            alpha_mode,
        })
    }
}

impl ImageSource for ZenpixelsSource<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        let bpp = self.pixel_format.bytes_per_pixel();
        let start = y * self.stride;
        &self.data[start..start + self.width * bpp]
    }
}

impl<'a> TryFrom<&'a PixelSlice<'a>> for ZenpixelsSource<'a> {
    type Error = UnsupportedFormat;
    fn try_from(s: &'a PixelSlice<'a>) -> Result<Self, UnsupportedFormat> {
        Self::try_from_slice(s)
    }
}

impl<'a> TryFrom<&'a PixelBuffer> for ZenpixelsSource<'a> {
    type Error = UnsupportedFormat;
    fn try_from(b: &'a PixelBuffer) -> Result<Self, UnsupportedFormat> {
        Self::try_from_buffer(b)
    }
}

impl From<UnsupportedFormat> for Ssimulacra2Error {
    fn from(e: UnsupportedFormat) -> Self {
        let _ = e;
        Ssimulacra2Error::InvalidInputData { actual: 0 }
    }
}

/// Map a zenpixels [`PixelDescriptor`] to [`PixelFormat`]/[`AlphaMode`].
fn map_descriptor(
    desc: &PixelDescriptor,
) -> Result<(PixelFormat, AlphaMode), UnsupportedFormat> {
    match desc.transfer {
        TransferFunction::Srgb | TransferFunction::Bt709 => {}
        TransferFunction::Linear => {
            if !matches!(desc.format, ZpFormat::RgbaF32 | ZpFormat::RgbF32) {
                return Err(UnsupportedFormat(
                    "linear transfer requires an f32 format",
                ));
            }
        }
        TransferFunction::Pq | TransferFunction::Hlg => {
            return Err(UnsupportedFormat(
                "HDR transfers (PQ, HLG) — convert to sRGB or linear via \
                 zenpixels-convert first",
            ));
        }
        _ => {
            return Err(UnsupportedFormat("unknown transfer function"));
        }
    }

    if desc.signal_range != SignalRange::Full {
        return Err(UnsupportedFormat(
            "narrow/limited signal range — expand to full range upstream",
        ));
    }

    match desc.primaries {
        ZpPrimaries::Bt709 => {}
        _ => {
            return Err(UnsupportedFormat(
                "non-BT.709 primaries — gamut-convert via zenpixels-convert first",
            ));
        }
    }

    let pixel_format = match (desc.format, desc.transfer) {
        (ZpFormat::Rgb8, _) => PixelFormat::Srgb8Rgb,
        (ZpFormat::Rgba8 | ZpFormat::Rgbx8, _) => PixelFormat::Srgb8Rgba,
        (ZpFormat::Bgra8 | ZpFormat::Bgrx8, _) => PixelFormat::Srgb8Bgra,
        (ZpFormat::Rgb16, _) => PixelFormat::Srgb16Rgb,
        (ZpFormat::Rgba16, _) => PixelFormat::Srgb16Rgba,
        (ZpFormat::RgbF32, TransferFunction::Linear) => PixelFormat::LinearF32Rgb,
        (ZpFormat::RgbaF32, TransferFunction::Linear) => PixelFormat::LinearF32Rgba,
        (ZpFormat::RgbF32, TransferFunction::Srgb | TransferFunction::Bt709) => {
            PixelFormat::SrgbF32Rgb
        }
        (ZpFormat::Gray8 | ZpFormat::Gray16 | ZpFormat::GrayF32
        | ZpFormat::GrayA8 | ZpFormat::GrayA16 | ZpFormat::GrayAF32, _) => {
            return Err(UnsupportedFormat(
                "grayscale — wrap bytes in fast_ssim2::GraySlice instead",
            ));
        }
        _ => return Err(UnsupportedFormat("unsupported pixel format")),
    };

    let alpha_mode = match desc.alpha {
        None | Some(ZpAlpha::Undefined) | Some(ZpAlpha::Opaque) => AlphaMode::Opaque,
        Some(ZpAlpha::Straight) => AlphaMode::Straight,
        Some(ZpAlpha::Premultiplied) => AlphaMode::Straight, // un-premultiplied on ingest
        _ => AlphaMode::Straight,
    };
    Ok((pixel_format, alpha_mode))
}

/// Un-premultiply to straight alpha, native domain (ported from zensim).
fn unpremultiply(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    desc: &PixelDescriptor,
) -> Vec<u8> {
    let mut out = data.to_vec();
    match desc.format {
        ZpFormat::Rgba8 | ZpFormat::Bgra8 | ZpFormat::Rgbx8 | ZpFormat::Bgrx8 => {
            for y in 0..height {
                let row_start = y * stride;
                for x in 0..width {
                    let off = row_start + x * 4;
                    let a = out[off + 3];
                    if a == 0 {
                        out[off] = 0;
                        out[off + 1] = 0;
                        out[off + 2] = 0;
                    } else if a < 255 {
                        let inv = 255.0 / a as f32;
                        for c in 0..3 {
                            out[off + c] =
                                (out[off + c] as f32 * inv).round().min(255.0) as u8;
                        }
                    }
                }
            }
        }
        ZpFormat::Rgba16 => {
            for y in 0..height {
                let row_start = y * stride;
                for x in 0..width {
                    let off = row_start + x * 8;
                    let a = u16::from_ne_bytes([out[off + 6], out[off + 7]]);
                    if a == 0 {
                        out[off..off + 6].fill(0);
                    } else if a < 65535 {
                        let inv = 65535.0 / a as f32;
                        for c in 0..3 {
                            let co = off + c * 2;
                            let v = u16::from_ne_bytes([out[co], out[co + 1]]);
                            let unpremul = (v as f32 * inv).round().min(65535.0) as u16;
                            out[co..co + 2].copy_from_slice(&unpremul.to_ne_bytes());
                        }
                    }
                }
            }
        }
        ZpFormat::RgbaF32 => {
            for y in 0..height {
                let row_start = y * stride;
                for x in 0..width {
                    let off = row_start + x * 16;
                    let a = f32::from_ne_bytes([
                        out[off + 12],
                        out[off + 13],
                        out[off + 14],
                        out[off + 15],
                    ]);
                    if a <= 0.0 {
                        out[off..off + 12].fill(0);
                    } else if a < 1.0 {
                        let inv = 1.0 / a;
                        for c in 0..3 {
                            let co = off + c * 4;
                            let v = f32::from_ne_bytes([
                                out[co],
                                out[co + 1],
                                out[co + 2],
                                out[co + 3],
                            ]);
                            out[co..co + 4].copy_from_slice(&(v * inv).to_ne_bytes());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    out
}
