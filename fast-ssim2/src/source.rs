//! `ImageSource` — zero-copy, row-level image input for SSIMULACRA2.
#![allow(clippy::chunks_exact_to_as_chunks)]
//!
//! Same shape as zensim's `ImageSource` (family-convergent): callers provide
//! borrowed pixel rows plus a [`PixelFormat`]/[`AlphaMode`] descriptor; the
//! metric pulls rows directly into its pipeline — encoded (sRGB-quantized)
//! formats go through the captured lcms LUT path bit-exactly, linear f32
//! formats pass through unchanged.
//!
//! Concrete adapters in this module cover contiguous `&[[u8; N]]` slices,
//! strided byte buffers, grayscale, and [`LinearRgbImage`]. The optional
//! `zenpixels` feature adds [`ZenpixelsSource`](crate::ZenpixelsSource)
//! (bridging `zenpixels::PixelSlice`/`PixelBuffer`), and `imgref` adds
//! impls for `ImgRef`/`ImgVec` pixel types.
//!
//! For sources that are not already sRGB (YUV video, wide-gamut, HDR),
//! convert upstream — e.g. with `zenpixels-convert` — then score the
//! sRGB result here. The metric has no color management of its own.

use crate::pipeline::{EncodedData, EncodedSrgb};
use crate::{LinearRgbImage, Ssimulacra2Error};

/// Pixel layout + encoding of a [`ImageSource`]'s rows.
///
/// `Srgb*` formats are sRGB-encoded (gamma) — they linearize through the
/// captured lcms LUT, the same values the reference implementation reads.
/// `LinearF32*` formats are already linear light in `[0, 1]` and pass
/// through untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// sRGB 8-bit RGB, 3 bpp: `[R, G, B]`.
    Srgb8Rgb,
    /// sRGB 8-bit RGBA, 4 bpp: `[R, G, B, A]`. Alpha per [`ImageSource::alpha_mode`].
    Srgb8Rgba,
    /// sRGB 8-bit BGRA, 4 bpp: `[B, G, R, A]` — common on Windows surfaces.
    Srgb8Bgra,
    /// sRGB 8-bit grayscale, 1 bpp — expanded to RGB per the reference.
    Srgb8Gray,
    /// sRGB 16-bit RGB, 6 bpp: `[R, G, B]` as native-endian `u16`.
    Srgb16Rgb,
    /// sRGB 16-bit RGBA, 8 bpp. Alpha per [`ImageSource::alpha_mode`].
    Srgb16Rgba,
    /// sRGB-encoded f32 RGB, 12 bpp: `[f32; 3]` in `0..=1`. On-grid values
    /// (multiples of 1/255) snap to the u8 LUT bit-exactly; off-grid values
    /// use the family-standard rational polynomial.
    SrgbF32Rgb,
    /// Linear-light f32 RGB, 12 bpp.
    LinearF32Rgb,
    /// Linear-light f32 RGBA, 16 bpp. Alpha per [`ImageSource::alpha_mode`].
    LinearF32Rgba,
    /// Linear-light f32 grayscale, 4 bpp — expanded to RGB.
    LinearF32Gray,
}

impl PixelFormat {
    /// Bytes per pixel.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Srgb8Rgb => 3,
            Self::Srgb8Rgba | Self::Srgb8Bgra => 4,
            Self::Srgb8Gray => 1,
            Self::Srgb16Rgb => 6,
            Self::Srgb16Rgba => 8,
            Self::SrgbF32Rgb | Self::LinearF32Rgb => 12,
            Self::LinearF32Rgba => 16,
            Self::LinearF32Gray => 4,
        }
    }

    /// Whether the format carries an alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Srgb8Rgba | Self::Srgb8Bgra | Self::Srgb16Rgba | Self::LinearF32Rgba)
    }
}

/// Alpha-channel interpretation for [`PixelFormat`]s with alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    /// No meaningful alpha — alpha bytes are ignored.
    Opaque,
    /// Straight (unassociated) alpha — the reference's two-background
    /// compositing (0.1/0.9, min score) applies.
    Straight,
    /// Premultiplied alpha — un-premultiplied to straight on ingest
    /// (one extra pass over the input).
    Premultiplied,
}

/// Zero-copy access to image pixel data, row by row.
///
/// Implementors provide row-level access with arbitrary stride. Width and
/// height come from the trait — no separate dimension parameters.
///
/// `Sync` so strip scoring can pull rows from multiple threads.
pub trait ImageSource: Sync {
    /// Image width in pixels.
    fn width(&self) -> usize;
    /// Image height in pixels.
    fn height(&self) -> usize;
    /// Pixel format (layout + encoding) of every row.
    fn pixel_format(&self) -> PixelFormat;
    /// Alpha interpretation; must be [`AlphaMode::Opaque`] for
    /// formats without alpha.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    /// Raw bytes for row `y` — at least `width() * pixel_format().bytes_per_pixel()`
    /// bytes, native-endian for u16/f32 elements.
    fn row_bytes(&self, y: usize) -> &[u8];
}

impl<T: ImageSource + ?Sized> ImageSource for &T {
    fn width(&self) -> usize {
        (**self).width()
    }
    fn height(&self) -> usize {
        (**self).height()
    }
    fn pixel_format(&self) -> PixelFormat {
        (**self).pixel_format()
    }
    fn alpha_mode(&self) -> AlphaMode {
        (**self).alpha_mode()
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        (**self).row_bytes(y)
    }
}

// ============================================================================
// Concrete adapters
// ============================================================================

/// Contiguous `&[[u8; 3]]` sRGB pixels.
#[derive(Clone, Copy, Debug)]
pub struct RgbSlice<'a> {
    data: &'a [[u8; 3]],
    width: usize,
    height: usize,
}

impl<'a> RgbSlice<'a> {
    /// Fallible constructor — rejects `data.len() < width * height`.
    pub fn try_new(data: &'a [[u8; 3]], width: usize, height: usize) -> Result<Self, Ssimulacra2Error> {
        let required = width.checked_mul(height).ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
        if data.len() < required {
            return Err(Ssimulacra2Error::InvalidInputData {
                actual: data.len() * 3,
            });
        }
        Ok(Self { data, width, height })
    }

    /// Infallible constructor — panics if `data.len() < width * height`.
    pub fn new(data: &'a [[u8; 3]], width: usize, height: usize) -> Self {
        Self::try_new(data, width, height).expect("RgbSlice: data.len() < width*height")
    }
}

impl ImageSource for RgbSlice<'_> {
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb8Rgb }
    fn row_bytes(&self, y: usize) -> &[u8] {
        self.data[y * self.width..(y + 1) * self.width].as_flattened()
    }
}

/// Contiguous `&[[u8; 4]]` sRGB RGBA pixels.
#[derive(Clone, Copy, Debug)]
pub struct RgbaSlice<'a> {
    data: &'a [[u8; 4]],
    width: usize,
    height: usize,
    alpha_mode: AlphaMode,
}

impl<'a> RgbaSlice<'a> {
    /// Fallible constructor.
    pub fn try_new(data: &'a [[u8; 4]], width: usize, height: usize, alpha_mode: AlphaMode) -> Result<Self, Ssimulacra2Error> {
        let required = width.checked_mul(height).ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
        if data.len() < required {
            return Err(Ssimulacra2Error::InvalidInputData {
                actual: data.len() * 4,
            });
        }
        Ok(Self { data, width, height, alpha_mode })
    }

    /// Straight-alpha constructor.
    pub fn new(data: &'a [[u8; 4]], width: usize, height: usize) -> Self {
        Self::try_new(data, width, height, AlphaMode::Straight).expect("RgbaSlice: data.len() < width*height")
    }
}

impl ImageSource for RgbaSlice<'_> {
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb8Rgba }
    fn alpha_mode(&self) -> AlphaMode { self.alpha_mode }
    fn row_bytes(&self, y: usize) -> &[u8] {
        self.data[y * self.width..(y + 1) * self.width].as_flattened()
    }
}

/// Contiguous `&[u8]` sRGB grayscale pixels.
#[derive(Clone, Copy, Debug)]
pub struct GraySlice<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
}

impl<'a> GraySlice<'a> {
    /// Fallible constructor.
    pub fn try_new(data: &'a [u8], width: usize, height: usize) -> Result<Self, Ssimulacra2Error> {
        let required = width.checked_mul(height).ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
        if data.len() < required {
            return Err(Ssimulacra2Error::InvalidInputData { actual: data.len() });
        }
        Ok(Self { data, width, height })
    }

    /// Infallible constructor.
    pub fn new(data: &'a [u8], width: usize, height: usize) -> Self {
        Self::try_new(data, width, height).expect("GraySlice: data.len() < width*height")
    }
}

impl ImageSource for GraySlice<'_> {
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb8Gray }
    fn row_bytes(&self, y: usize) -> &[u8] {
        &self.data[y * self.width..(y + 1) * self.width]
    }
}

/// Contiguous `&[[u16; 3]]` sRGB pixels (16-bit PNGs, TIFFs).
#[derive(Clone, Copy, Debug)]
pub struct Rgb16Slice<'a> {
    data: &'a [[u16; 3]],
    width: usize,
    height: usize,
}

impl<'a> Rgb16Slice<'a> {
    /// Fallible constructor.
    pub fn try_new(data: &'a [[u16; 3]], width: usize, height: usize) -> Result<Self, Ssimulacra2Error> {
        let required = width.checked_mul(height).ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
        if data.len() < required {
            return Err(Ssimulacra2Error::InvalidInputData { actual: data.len() * 6 });
        }
        Ok(Self { data, width, height })
    }

    /// Infallible constructor.
    pub fn new(data: &'a [[u16; 3]], width: usize, height: usize) -> Self {
        Self::try_new(data, width, height).expect("Rgb16Slice: data.len() < width*height")
    }
}

impl ImageSource for Rgb16Slice<'_> {
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb16Rgb }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(self.data[y * self.width..(y + 1) * self.width].as_flattened())
    }
}

/// Contiguous `&[[f32; 3]]` **sRGB-encoded** pixels in `0..=1` — the f32
/// raster that quantized decode pipelines hand off when they keep
/// float headroom (e.g. `Rgb<f32>` images). On-grid values snap to the
/// u8 LUT bit-exactly.
#[derive(Clone, Copy, Debug)]
pub struct SrgbF32Slice<'a> {
    data: &'a [[f32; 3]],
    width: usize,
    height: usize,
}

impl<'a> SrgbF32Slice<'a> {
    /// Fallible constructor.
    pub fn try_new(data: &'a [[f32; 3]], width: usize, height: usize) -> Result<Self, Ssimulacra2Error> {
        let required = width.checked_mul(height).ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
        if data.len() < required {
            return Err(Ssimulacra2Error::InvalidInputData { actual: data.len() * 12 });
        }
        Ok(Self { data, width, height })
    }

    /// Infallible constructor.
    pub fn new(data: &'a [[f32; 3]], width: usize, height: usize) -> Self {
        Self::try_new(data, width, height).expect("SrgbF32Slice: data.len() < width*height")
    }
}

impl ImageSource for SrgbF32Slice<'_> {
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { PixelFormat::SrgbF32Rgb }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(self.data[y * self.width..(y + 1) * self.width].as_flattened())
    }
}

/// Arbitrary byte-layout source: raw data + explicit stride + format.
///
/// Covers everything the typed slices don't — padded rows, channel
/// orderings, planar-interleaved buffers from foreign decoders.
#[derive(Clone, Copy, Debug)]
pub struct StridedBytes<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
    stride: usize,
    pixel_format: PixelFormat,
    alpha_mode: AlphaMode,
}

impl<'a> StridedBytes<'a> {
    /// Fallible constructor. `stride` is bytes per row; the last row may
    /// be short (no trailing pad required).
    pub fn try_new(
        data: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
        pixel_format: PixelFormat,
        alpha_mode: AlphaMode,
    ) -> Result<Self, Ssimulacra2Error> {
        let bpp = pixel_format.bytes_per_pixel();
        let row_bytes = width.checked_mul(bpp).ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
        if stride < row_bytes {
            return Err(Ssimulacra2Error::InvalidInputData { actual: stride });
        }
        let required = if height == 0 {
            0
        } else {
            stride
                .checked_mul(height - 1)
                .and_then(|v| v.checked_add(row_bytes))
                .ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?
        };
        if data.len() < required {
            return Err(Ssimulacra2Error::InvalidInputData { actual: data.len() });
        }
        Ok(Self { data, width, height, stride, pixel_format, alpha_mode })
    }

    /// Contiguous shorthand — `stride = width * bpp`, opaque alpha.
    pub fn contiguous(data: &'a [u8], width: usize, height: usize, pixel_format: PixelFormat) -> Self {
        Self::try_new(
            data,
            width,
            height,
            width * pixel_format.bytes_per_pixel(),
            pixel_format,
            AlphaMode::Opaque,
        )
        .expect("StridedBytes::contiguous: data too short")
    }
}

impl ImageSource for StridedBytes<'_> {
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { self.pixel_format }
    fn alpha_mode(&self) -> AlphaMode { self.alpha_mode }
    fn row_bytes(&self, y: usize) -> &[u8] {
        let bpp = self.pixel_format.bytes_per_pixel();
        let start = y * self.stride;
        &self.data[start..start + self.width * bpp]
    }
}

/// A Y-range view over another [`ImageSource`] — no copy.
#[derive(Clone, Copy, Debug)]
pub struct SubsetView<'a, S: ImageSource + ?Sized> {
    parent: &'a S,
    y_start: usize,
    height: usize,
}

impl<'a, S: ImageSource + ?Sized> SubsetView<'a, S> {
    /// Fallible constructor — rejects ranges outside the parent.
    pub fn try_new(parent: &'a S, y_start: usize, height: usize) -> Result<Self, Ssimulacra2Error> {
        if height == 0 || y_start + height > parent.height() {
            return Err(Ssimulacra2Error::InvalidImageSize);
        }
        Ok(Self { parent, y_start, height })
    }
}

impl<S: ImageSource + ?Sized> ImageSource for SubsetView<'_, S> {
    fn width(&self) -> usize { self.parent.width() }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { self.parent.pixel_format() }
    fn alpha_mode(&self) -> AlphaMode { self.parent.alpha_mode() }
    fn row_bytes(&self, y: usize) -> &[u8] {
        self.parent.row_bytes(self.y_start + y)
    }
}

impl ImageSource for LinearRgbImage {
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { PixelFormat::LinearF32Rgb }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(self.data[y * self.width..(y + 1) * self.width].as_flattened())
    }
}

// ============================================================================
// Funnel: ImageSource → pipeline input
// ============================================================================

/// What the funnel produced — mirrors the pipeline's two input shapes.
pub(crate) enum PreparedInput {
    /// sRGB-quantized raster — LUT-exact encoded path.
    Encoded(EncodedSrgb),
    /// Already-linear planes + optional straight alpha.
    Linear {
        /// Channel-separated planes, `width * height` each.
        planes: [Vec<f32>; 3],
        /// Straight alpha as f32 in `[0, 1]`, if the source carries it.
        alpha: Option<Vec<f32>>,
        /// Row width.
        width: usize,
        /// Row count.
        height: usize,
    },
}

impl PreparedInput {
    /// Dimensions of the funnelled input.
    pub(crate) fn dims(&self) -> (usize, usize) {
        match self {
            Self::Encoded(e) => (e.width, e.height),
            Self::Linear { width, height, .. } => (*width, *height),
        }
    }
}

/// Read a source into the pipeline input it belongs on.
///
/// Encoded formats de-interleave RGB (and alpha) verbatim — the LUT does
/// the linearization. Linear formats read f32 pixels into planes.
pub(crate) fn funnel(src: &dyn ImageSource) -> Result<PreparedInput, Ssimulacra2Error> {
    let w = src.width();
    let h = src.height();
    if w == 0 || h == 0 {
        return Err(Ssimulacra2Error::InvalidImageSize);
    }
    let npix = w
        .checked_mul(h)
        .ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
    if npix > crate::MAX_IMAGE_PIXELS {
        return Err(Ssimulacra2Error::ImageTooLarge { actual: npix });
    }
    let fmt = src.pixel_format();
    if fmt.has_alpha() && src.alpha_mode() == AlphaMode::Opaque {
        // treat alpha bytes as padding — strip the channel
    }
    match fmt {
        PixelFormat::Srgb8Rgb
        | PixelFormat::Srgb8Rgba
        | PixelFormat::Srgb8Bgra
        | PixelFormat::Srgb8Gray => {
            let npix = w * h;
            let mut rgb = Vec::with_capacity(npix * 3);
            let mut alpha = (fmt.has_alpha() && src.alpha_mode() != AlphaMode::Opaque)
                .then(|| Vec::with_capacity(npix));
            for y in 0..h {
                let row = src.row_bytes(y);
                match fmt {
                    PixelFormat::Srgb8Rgb => rgb.extend_from_slice(&row[..w * 3]),
                    PixelFormat::Srgb8Gray => {
                        for &v in &row[..w] {
                            rgb.extend_from_slice(&[v, v, v]);
                        }
                    }
                    PixelFormat::Srgb8Rgba => {
                        for px in row[..w * 4].chunks_exact(4) {
                            rgb.extend_from_slice(&px[..3]);
                            if let Some(a) = &mut alpha {
                                a.push(px[3] as f32 * (1.0 / 255.0));
                            }
                        }
                    }
                    PixelFormat::Srgb8Bgra => {
                        for px in row[..w * 4].chunks_exact(4) {
                            rgb.extend_from_slice(&[px[2], px[1], px[0]]);
                            if let Some(a) = &mut alpha {
                                a.push(px[3] as f32 * (1.0 / 255.0));
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            if src.alpha_mode() == AlphaMode::Premultiplied {
                unpremultiply_u8(&mut rgb, alpha.as_deref().unwrap_or(&[]));
            }
            Ok(PreparedInput::Encoded(EncodedSrgb {
                width: w,
                height: h,
                data: EncodedData::U8(rgb),
                alpha,
            }))
        }
        PixelFormat::Srgb16Rgb | PixelFormat::Srgb16Rgba => {
            let npix = w * h;
            let mut rgb = Vec::with_capacity(npix * 3);
            let mut alpha = (fmt == PixelFormat::Srgb16Rgba && src.alpha_mode() != AlphaMode::Opaque)
                .then(|| Vec::with_capacity(npix));
            for y in 0..h {
                let row: &[u16] = bytemuck::cast_slice(src.row_bytes(y));
                match fmt {
                    PixelFormat::Srgb16Rgb => {
                        for px in row[..w * 3].chunks_exact(3) {
                            rgb.extend_from_slice(&px[..3]);
                        }
                    }
                    PixelFormat::Srgb16Rgba => {
                        for px in row[..w * 4].chunks_exact(4) {
                            rgb.extend_from_slice(&px[..3]);
                            if let Some(a) = &mut alpha {
                                a.push(px[3] as f32 * (1.0 / 65535.0));
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            if src.alpha_mode() == AlphaMode::Premultiplied {
                unpremultiply_u16(&mut rgb, alpha.as_deref().unwrap_or(&[]));
            }
            Ok(PreparedInput::Encoded(EncodedSrgb {
                width: w,
                height: h,
                data: EncodedData::U16(rgb),
                alpha,
            }))
        }
        PixelFormat::SrgbF32Rgb => {
            let mut rgb = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                let row: &[f32] = bytemuck::cast_slice(src.row_bytes(y));
                rgb.extend_from_slice(&row[..w * 3]);
            }
            Ok(PreparedInput::Encoded(EncodedSrgb {
                width: w,
                height: h,
                data: EncodedData::F32(rgb),
                alpha: None,
            }))
        }
        PixelFormat::LinearF32Rgb
        | PixelFormat::LinearF32Rgba
        | PixelFormat::LinearF32Gray => {
            let npix = w * h;
            let (mut r, mut g, mut b) = (
                Vec::with_capacity(npix),
                Vec::with_capacity(npix),
                Vec::with_capacity(npix),
            );
            let mut alpha = (fmt == PixelFormat::LinearF32Rgba
                && src.alpha_mode() != AlphaMode::Opaque)
                .then(|| Vec::with_capacity(npix));
            for y in 0..h {
                let row: &[f32] = bytemuck::cast_slice(src.row_bytes(y));
                match fmt {
                    PixelFormat::LinearF32Rgb => {
                        for px in row[..w * 3].chunks_exact(3) {
                            r.push(px[0]);
                            g.push(px[1]);
                            b.push(px[2]);
                        }
                    }
                    PixelFormat::LinearF32Rgba => {
                        for px in row[..w * 4].chunks_exact(4) {
                            r.push(px[0]);
                            g.push(px[1]);
                            b.push(px[2]);
                            if let Some(a) = &mut alpha {
                                a.push(px[3]);
                            }
                        }
                    }
                    PixelFormat::LinearF32Gray => {
                        for &v in &row[..w] {
                            r.push(v);
                            g.push(v);
                            b.push(v);
                        }
                    }
                    _ => unreachable!(),
                }
            }
            if src.alpha_mode() == AlphaMode::Premultiplied {
                unpremultiply_f32([&mut r, &mut g, &mut b], alpha.as_deref().unwrap_or(&[]));
            }
            Ok(PreparedInput::Linear {
                planes: [r, g, b],
                alpha,
                width: w,
                height: h,
            })
        }
    }
}

/// Un-premultiply encoded u8 RGB by its straight-alpha fraction
/// (quantized back to u8, like zensim's adapter).
fn unpremultiply_u8(rgb: &mut [u8], alpha: &[f32]) {
    for (px, &a) in rgb.chunks_exact_mut(3).zip(alpha.iter()) {
        if a > 0.0 {
            for c in px.iter_mut() {
                *c = ((*c as f32 / 255.0) / a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        } else {
            px.fill(0);
        }
    }
}

fn unpremultiply_u16(rgb: &mut [u16], alpha: &[f32]) {
    for (px, &a) in rgb.chunks_exact_mut(3).zip(alpha.iter()) {
        if a > 0.0 {
            for c in px.iter_mut() {
                *c = ((*c as f32 / 65535.0) / a * 65535.0).round().clamp(0.0, 65535.0) as u16;
            }
        } else {
            px.fill(0);
        }
    }
}

fn unpremultiply_f32(rgb: [&mut Vec<f32>; 3], alpha: &[f32]) {
    let [r, g, b] = rgb;
    for (i, &a) in alpha.iter().enumerate() {
        if a > 0.0 {
            r[i] /= a;
            g[i] /= a;
            b[i] /= a;
        } else {
            r[i] = 0.0;
            g[i] = 0.0;
            b[i] = 0.0;
        }
    }
}

// ============================================================================
// imgref impls
// ============================================================================

#[cfg(feature = "imgref")]
mod imgref_impl {
    use super::*;

    macro_rules! imgref_source {
        ($t:ty, $fmt:expr) => {
            impl ImageSource for imgref::ImgRef<'_, $t> {
                fn width(&self) -> usize {
                    imgref::Img::width(self)
                }
                fn height(&self) -> usize {
                    imgref::Img::height(self)
                }
                fn pixel_format(&self) -> PixelFormat {
                    $fmt
                }
                fn alpha_mode(&self) -> AlphaMode {
                    if self.pixel_format().has_alpha() {
                        AlphaMode::Straight
                    } else {
                        AlphaMode::Opaque
                    }
                }
                fn row_bytes(&self, y: usize) -> &[u8] {
                    let buf = imgref::Img::buf(self);
                    let stride = imgref::Img::stride(self);
                    let w = imgref::Img::width(self);
                    bytemuck::cast_slice(&buf[y * stride..y * stride + w])
                }
            }
        };
    }
    imgref_source!([u8; 3], PixelFormat::Srgb8Rgb);
    imgref_source!([u16; 3], PixelFormat::Srgb16Rgb);
    imgref_source!([u8; 4], PixelFormat::Srgb8Rgba);
    imgref_source!([u16; 4], PixelFormat::Srgb16Rgba);
    imgref_source!([f32; 3], PixelFormat::LinearF32Rgb);
    imgref_source!(u8, PixelFormat::Srgb8Gray);
    imgref_source!(f32, PixelFormat::LinearF32Gray);

    impl ImageSource for imgref::ImgVec<[u8; 3]> {
        fn width(&self) -> usize { imgref::ImgVec::width(self) }
        fn height(&self) -> usize { imgref::ImgVec::height(self) }
        fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb8Rgb }
        fn row_bytes(&self, y: usize) -> &[u8] {
            let buf = imgref::Img::buf(self);
            let stride = imgref::Img::stride(self);
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[y * stride..y * stride + w])
        }
    }
    impl ImageSource for imgref::ImgVec<[u16; 3]> {
        fn width(&self) -> usize { imgref::ImgVec::width(self) }
        fn height(&self) -> usize { imgref::ImgVec::height(self) }
        fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb16Rgb }
        fn row_bytes(&self, y: usize) -> &[u8] {
            let buf = imgref::Img::buf(self);
            let stride = imgref::Img::stride(self);
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[y * stride..y * stride + w])
        }
    }
    impl ImageSource for imgref::ImgVec<[u8; 4]> {
        fn width(&self) -> usize { imgref::ImgVec::width(self) }
        fn height(&self) -> usize { imgref::ImgVec::height(self) }
        fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb8Rgba }
        fn alpha_mode(&self) -> AlphaMode { AlphaMode::Straight }
        fn row_bytes(&self, y: usize) -> &[u8] {
            let buf = imgref::Img::buf(self);
            let stride = imgref::Img::stride(self);
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[y * stride..y * stride + w])
        }
    }
    impl ImageSource for imgref::ImgVec<[u16; 4]> {
        fn width(&self) -> usize { imgref::ImgVec::width(self) }
        fn height(&self) -> usize { imgref::ImgVec::height(self) }
        fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb16Rgba }
        fn alpha_mode(&self) -> AlphaMode { AlphaMode::Straight }
        fn row_bytes(&self, y: usize) -> &[u8] {
            let buf = imgref::Img::buf(self);
            let stride = imgref::Img::stride(self);
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[y * stride..y * stride + w])
        }
    }
    impl ImageSource for imgref::ImgVec<[f32; 3]> {
        fn width(&self) -> usize { imgref::ImgVec::width(self) }
        fn height(&self) -> usize { imgref::ImgVec::height(self) }
        fn pixel_format(&self) -> PixelFormat { PixelFormat::LinearF32Rgb }
        fn row_bytes(&self, y: usize) -> &[u8] {
            let buf = imgref::Img::buf(self);
            let stride = imgref::Img::stride(self);
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[y * stride..y * stride + w])
        }
    }
    impl ImageSource for imgref::ImgVec<u8> {
        fn width(&self) -> usize { imgref::ImgVec::width(self) }
        fn height(&self) -> usize { imgref::ImgVec::height(self) }
        fn pixel_format(&self) -> PixelFormat { PixelFormat::Srgb8Gray }
        fn row_bytes(&self, y: usize) -> &[u8] {
            let buf = imgref::Img::buf(self);
            let stride = imgref::Img::stride(self);
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[y * stride..y * stride + w])
        }
    }
    impl ImageSource for imgref::ImgVec<f32> {
        fn width(&self) -> usize { imgref::ImgVec::width(self) }
        fn height(&self) -> usize { imgref::ImgVec::height(self) }
        fn pixel_format(&self) -> PixelFormat { PixelFormat::LinearF32Gray }
        fn row_bytes(&self, y: usize) -> &[u8] {
            let buf = imgref::Img::buf(self);
            let stride = imgref::Img::stride(self);
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[y * stride..y * stride + w])
        }
    }
}

/// Owned `Vec<[f32; 3]>` **sRGB-encoded** raster — the owned counterpart
/// of [`SrgbF32Slice`] for callers building images at runtime.
#[derive(Clone, Debug)]
pub struct SrgbF32Image {
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
}

impl SrgbF32Image {
    /// Fallible constructor.
    pub fn try_new(data: Vec<[f32; 3]>, width: usize, height: usize) -> Result<Self, Ssimulacra2Error> {
        let required = width.checked_mul(height).ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
        if data.len() != required {
            return Err(Ssimulacra2Error::InvalidInputData { actual: data.len() * 12 });
        }
        Ok(Self { data, width, height })
    }

    /// Infallible constructor.
    pub fn new(data: Vec<[f32; 3]>, width: usize, height: usize) -> Self {
        Self::try_new(data, width, height).expect("SrgbF32Image: data.len() != width*height")
    }

    /// Pixel data.
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }
}

impl ImageSource for SrgbF32Image {
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn pixel_format(&self) -> PixelFormat { PixelFormat::SrgbF32Rgb }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(self.data[y * self.width..(y + 1) * self.width].as_flattened())
    }
}
