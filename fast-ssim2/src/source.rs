//! Input funnel: a [`zenpixels::PixelView`] (borrowed, self-describing,
//! strided) becomes either an [`EncodedSrgb`] (quantized/encoded inputs —
//! u8 takes the reference-captured LUT for bit-exact parity; u16/f32
//! evaluate the sRGB polynomial) or deinterleaved linear `f32` planes
//! (already-linear inputs).
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::borrow::Cow;

use zenpixels::{
    AlphaMode, BitPacking, ChannelOrder, ColorSpace, DiffuseWhite, PixelFormat, PixelView,
    Primaries, SampleType, SignalRange, Transfer,
};

use crate::Ssimulacra2Error;
use crate::pipeline::{EncodedData, EncodedSrgb};

mod native_hdr;
mod native_hdr_pow;

pub(crate) struct ColorParams {
    pub(crate) primaries: Primaries,
    pub(crate) transfer: Transfer,
    pub(crate) range: SignalRange,
    pub(crate) alpha: AlphaMode,
    pub(crate) white: Option<DiffuseWhite>,
    pub(crate) is_gray: bool,
}

pub(crate) fn extract_color(color: &ColorSpace) -> Result<ColorParams, Ssimulacra2Error> {
    match *color {
        ColorSpace::Rgb {
            primaries,
            transfer,
            range,
            alpha,
            white,
        } => Ok(ColorParams {
            primaries,
            transfer,
            range,
            alpha,
            white,
            is_gray: false,
        }),
        ColorSpace::Gray {
            transfer,
            range,
            alpha,
            white,
        } => Ok(ColorParams {
            primaries: Primaries::Bt709,
            transfer,
            range,
            alpha,
            white,
            is_gray: true,
        }),
        _ => Err(Ssimulacra2Error::UnsupportedInput(
            "unsupported color space — only RGB and Gray are supported; convert via zenpixels-convert first",
        )),
    }
}

/// Result of [`funnel`] — the encoded path preserves the original bytes
/// (LUT-exact); the linear path holds premultiplied-ready planes.
pub(crate) enum PreparedInput {
    /// Encoded sRGB quantized/f32-on-grid input → reference LUT path.
    Encoded(EncodedSrgb),
    /// Native reference color conversion, retaining the intensity target.
    Native(native_hdr::NativeInput),
    /// Already-linear input — planes + optional straight-alpha plane.
    Linear {
        planes: [Vec<f32>; 3],
        alpha: Option<Vec<f32>>,
        width: usize,
        height: usize,
    },
}

impl PreparedInput {
    /// (width, height) of the source, pre-padding.
    pub(crate) fn dims(&self) -> (usize, usize) {
        match self {
            Self::Encoded(e) => (e.width, e.height),
            Self::Native(n) => (n.width, n.height),
            Self::Linear { width, height, .. } => (*width, *height),
        }
    }

    pub(crate) fn intensity_target(&self) -> f32 {
        match self {
            Self::Native(n) => n.intensity_target,
            _ => 255.0,
        }
    }
}

/// Validate original and padded dimensions before allocating input planes.
pub(crate) fn checked_dimensions(w: usize, h: usize) -> Result<usize, Ssimulacra2Error> {
    if w == 0 || h == 0 {
        return Err(Ssimulacra2Error::InvalidImageSize);
    }
    let npix = w
        .checked_mul(h)
        .ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
    if npix > crate::MAX_IMAGE_PIXELS {
        return Err(Ssimulacra2Error::ImageTooLarge { actual: npix });
    }
    let padded_pixels = w
        .max(8)
        .checked_mul(h.max(8))
        .ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
    if padded_pixels > crate::MAX_IMAGE_PIXELS {
        return Err(Ssimulacra2Error::ImageTooLarge {
            actual: padded_pixels,
        });
    }
    Ok(npix)
}

fn validate_alpha_layout(fmt: &PixelFormat) -> Result<(), Ssimulacra2Error> {
    if matches!(
        fmt.color.alpha(),
        AlphaMode::Straight | AlphaMode::Premultiplied
    ) && !fmt.layout.order.has_alpha_slot()
    {
        return Err(Ssimulacra2Error::UnsupportedInput(
            "alpha semantics require a stored alpha channel",
        ));
    }
    Ok(())
}

/// Interpret `view` per its [`PixelFormat`], materializing the
/// planes the pipeline needs. See the crate docs for which formats
/// are rejected. Native PQ/HLG and supported linear HDR formats use the
/// reference color conversion; the encoded sRGB path keeps its captured LUT.
pub(crate) fn funnel(view: &PixelView<'_>) -> Result<PreparedInput, Ssimulacra2Error> {
    let fmt = view.format();
    let (w, h) = (view.width() as usize, view.height() as usize);
    let npix = checked_dimensions(w, h)?;
    if fmt.layout.packing != BitPacking::Full {
        return Err(Ssimulacra2Error::UnsupportedInput(
            "bit-packed pixel layouts are not supported",
        ));
    }
    let color = extract_color(&fmt.color)?;

    if matches!(color.transfer, Transfer::Pq | Transfer::Hlg(_))
        || (color.transfer == Transfer::Linear
            && ((!color.is_gray && color.primaries != Primaries::Bt709) || color.white.is_some()))
    {
        return native_hdr::NativeInput::new(view).map(PreparedInput::Native);
    }
    validate_alpha_layout(fmt)?;

    // Format gates — reject what the SDR metric can't honestly score.
    match color.transfer {
        Transfer::Srgb => {}
        Transfer::Linear => {
            if fmt.layout.sample != SampleType::F32
                || !matches!(
                    fmt.layout.order,
                    ChannelOrder::Rgba
                        | ChannelOrder::Rgb
                        | ChannelOrder::Gray
                        | ChannelOrder::GrayAlpha
                )
            {
                return Err(Ssimulacra2Error::UnsupportedInput(
                    "linear transfer requires an f32 format",
                ));
            }
        }
        Transfer::Pq | Transfer::Hlg(_) => {
            return Err(Ssimulacra2Error::UnsupportedInput(
                "HDR transfers (PQ, HLG) — convert via zenpixels-convert first",
            ));
        }
        Transfer::Unknown => {
            return Err(Ssimulacra2Error::UnsupportedInput(
                "unknown transfer — state it explicitly (e.g. RGB8_SRGB) so the bytes aren't silently misread",
            ));
        }
        _ => {
            return Err(Ssimulacra2Error::UnsupportedInput(
                "non-sRGB encoded transfer — the LUT is sRGB-shaped; convert upstream",
            ));
        }
    }
    if !color.is_gray && color.primaries != Primaries::Bt709 {
        return Err(Ssimulacra2Error::UnsupportedInput(
            "non-BT.709 primaries — gamut-convert via zenpixels-convert first",
        ));
    }
    if color.range != SignalRange::Full {
        return Err(Ssimulacra2Error::UnsupportedInput(
            "narrow/limited signal range — expand to full range upstream",
        ));
    }

    // Premultiplied alpha is un-multiplied into an owned buffer; every
    // other layout is consumed borrowed (zero-copy).
    let data: Cow<'_, [u8]> = if color.alpha == AlphaMode::Premultiplied {
        Cow::Owned(unpremultiply(view))
    } else {
        Cow::Borrowed(view.as_strided_bytes())
    };
    let stride = view.stride_bytes();
    let premult_alpha = matches!(color.alpha, AlphaMode::Straight | AlphaMode::Premultiplied);

    let f32_alpha_of = |data: &[u8], stride: usize, bpp: usize| -> Vec<f32> {
        let mut a = Vec::with_capacity(npix);
        for y in 0..h {
            let row = &data[y * stride..];
            for px in row[..w * bpp].chunks_exact(bpp) {
                a.push(f32::from_ne_bytes([px[12], px[13], px[14], px[15]]));
            }
        }
        a
    };

    Ok(
        match (fmt.layout.sample, fmt.layout.order, color.transfer) {
            // ---------------------------------------------------- encoded u8
            (SampleType::U8, ChannelOrder::Rgb, _) => {
                let mut rgb = Vec::with_capacity(npix * 3);
                for y in 0..h {
                    let row = &data[y * stride..];
                    rgb.extend_from_slice(&row[..w * 3]);
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::U8(rgb),
                    alpha: None,
                })
            }
            (SampleType::U8, ChannelOrder::Rgba | ChannelOrder::Rgbx, _) => {
                let mut rgb = Vec::with_capacity(npix * 3);
                let mut alpha = premult_alpha.then(|| Vec::with_capacity(npix));
                for y in 0..h {
                    let row = &data[y * stride..];
                    for px in row[..w * 4].chunks_exact(4) {
                        rgb.extend_from_slice(&px[..3]);
                        if let Some(a) = alpha.as_mut() {
                            a.push(px[3] as f32 * (1.0 / 255.0));
                        }
                    }
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::U8(rgb),
                    alpha,
                })
            }
            (SampleType::U8, ChannelOrder::Bgra | ChannelOrder::Bgrx, _) => {
                let mut rgb = Vec::with_capacity(npix * 3);
                let mut alpha = premult_alpha.then(|| Vec::with_capacity(npix));
                for y in 0..h {
                    let row = &data[y * stride..];
                    for px in row[..w * 4].chunks_exact(4) {
                        rgb.extend_from_slice(&[px[2], px[1], px[0]]);
                        if let Some(a) = alpha.as_mut() {
                            a.push(px[3] as f32 * (1.0 / 255.0));
                        }
                    }
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::U8(rgb),
                    alpha,
                })
            }
            (SampleType::U8, ChannelOrder::Gray, _) => {
                let mut g = Vec::with_capacity(npix);
                for y in 0..h {
                    let row = &data[y * stride..];
                    g.extend_from_slice(&row[..w]);
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::U8(g.into_iter().flat_map(|v| [v; 3]).collect()),
                    alpha: None,
                })
            }
            // --------------------------------------------------- encoded u16
            (SampleType::U16, ChannelOrder::Rgb, _) => {
                let mut rgb = Vec::with_capacity(npix * 3);
                for y in 0..h {
                    let row = &data[y * stride..];
                    for c in row[..w * 6].chunks_exact(2) {
                        rgb.push(u16::from_ne_bytes([c[0], c[1]]));
                    }
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::U16(rgb),
                    alpha: None,
                })
            }
            (SampleType::U16, ChannelOrder::Rgba, _) => {
                let mut rgb = Vec::with_capacity(npix * 3);
                let mut alpha = premult_alpha.then(|| Vec::with_capacity(npix));
                for y in 0..h {
                    let row = &data[y * stride..];
                    for px in row[..w * 8].chunks_exact(8) {
                        for c in 0..3 {
                            rgb.push(u16::from_ne_bytes([px[2 * c], px[2 * c + 1]]));
                        }
                        if let Some(a) = alpha.as_mut() {
                            a.push(u16::from_ne_bytes([px[6], px[7]]) as f32 * (1.0 / 65535.0));
                        }
                    }
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::U16(rgb),
                    alpha,
                })
            }
            (SampleType::U16, ChannelOrder::Gray, _) => {
                let mut g = Vec::with_capacity(npix);
                for y in 0..h {
                    let row = &data[y * stride..];
                    for c in row[..w * 2].chunks_exact(2) {
                        g.push(u16::from_ne_bytes([c[0], c[1]]));
                    }
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::U16(g.into_iter().flat_map(|v| [v; 3]).collect()),
                    alpha: None,
                })
            }
            // -------------------------------------- encoded f32
            // sRGB-encoded f32 is the *general* path — a rational polynomial
            // (`linear-srgb`), no grid tricks: encoded-f32 is a marginal input
            // class (decoders emit integers; linear pipelines emit Linear-f32),
            // and callers with u8-quantized data get bit-exact results by
            // handing us u8 directly.
            (SampleType::F32, ChannelOrder::Rgb, Transfer::Srgb) => {
                let mut rgb = Vec::with_capacity(npix * 3);
                for y in 0..h {
                    let row = &data[y * stride..];
                    for c in row[..w * 12].chunks_exact(4) {
                        rgb.push(f32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
                    }
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::F32(rgb),
                    alpha: None,
                })
            }
            (SampleType::F32, ChannelOrder::Rgba, Transfer::Srgb) => {
                let mut rgb = Vec::with_capacity(npix * 3);
                let mut alpha = premult_alpha.then(|| Vec::with_capacity(npix));
                for y in 0..h {
                    let row = &data[y * stride..];
                    for px in row[..w * 16].chunks_exact(16) {
                        for c in 0..3 {
                            rgb.push(f32::from_ne_bytes([
                                px[4 * c],
                                px[4 * c + 1],
                                px[4 * c + 2],
                                px[4 * c + 3],
                            ]));
                        }
                        if let Some(a) = alpha.as_mut() {
                            a.push(f32::from_ne_bytes([px[12], px[13], px[14], px[15]]));
                        }
                    }
                }
                PreparedInput::Encoded(EncodedSrgb {
                    width: w,
                    height: h,
                    data: EncodedData::F32(rgb),
                    alpha,
                })
            }
            // ------------------------------------------------- linear f32
            (SampleType::F32, ChannelOrder::Rgb, Transfer::Linear) => {
                let mut planes = [
                    Vec::with_capacity(npix),
                    Vec::with_capacity(npix),
                    Vec::with_capacity(npix),
                ];
                for y in 0..h {
                    let row = &data[y * stride..];
                    for px in row[..w * 12].chunks_exact(12) {
                        for (c, ch) in px.chunks_exact(4).enumerate() {
                            planes[c].push(f32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]));
                        }
                    }
                }
                PreparedInput::Linear {
                    planes,
                    alpha: None,
                    width: w,
                    height: h,
                }
            }
            (SampleType::F32, ChannelOrder::Rgba, Transfer::Linear) => PreparedInput::Linear {
                planes: {
                    let mut planes = [
                        Vec::with_capacity(npix),
                        Vec::with_capacity(npix),
                        Vec::with_capacity(npix),
                    ];
                    for y in 0..h {
                        let row = &data[y * stride..];
                        for px in row[..w * 16].chunks_exact(16) {
                            for c in 0..3 {
                                planes[c].push(f32::from_ne_bytes([
                                    px[4 * c],
                                    px[4 * c + 1],
                                    px[4 * c + 2],
                                    px[4 * c + 3],
                                ]));
                            }
                        }
                    }
                    planes
                },
                alpha: premult_alpha.then(|| f32_alpha_of(&data, stride, 16)),
                width: w,
                height: h,
            },
            (SampleType::F32, ChannelOrder::Gray, Transfer::Linear) => {
                let mut g = Vec::with_capacity(npix);
                for y in 0..h {
                    let row = &data[y * stride..];
                    for c in row[..w * 4].chunks_exact(4) {
                        g.push(f32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
                    }
                }
                PreparedInput::Linear {
                    planes: [g.clone(), g.clone(), g],
                    alpha: None,
                    width: w,
                    height: h,
                }
            }
            (SampleType::F32, ChannelOrder::GrayAlpha, Transfer::Linear) => {
                let mut g = Vec::with_capacity(npix);
                let mut a = Vec::with_capacity(npix);
                for y in 0..h {
                    let row = &data[y * stride..];
                    for px in row[..w * 8].chunks_exact(8) {
                        g.push(f32::from_ne_bytes([px[0], px[1], px[2], px[3]]));
                        a.push(f32::from_ne_bytes([px[4], px[5], px[6], px[7]]));
                    }
                }
                PreparedInput::Linear {
                    planes: [g.clone(), g.clone(), g],
                    alpha: premult_alpha.then_some(a),
                    width: w,
                    height: h,
                }
            }
            _ => {
                return Err(Ssimulacra2Error::UnsupportedInput(
                    "unsupported pixel format for SSIMULACRA2 — RGB/RGBA/BGRA/Gray \
                 in u8/u16/f32-sRGB or f32-linear only",
                ));
            }
        },
    )
}

/// Un-premultiply to straight alpha, native domain (ported from zensim's
/// zenpixels adapter — quantizes back for integer formats).
fn unpremultiply(view: &PixelView<'_>) -> Vec<u8> {
    let (w, h) = (view.width() as usize, view.height() as usize);
    let stride = view.stride_bytes();
    let layout = view.format().layout;
    let mut out = view.as_strided_bytes().to_vec();
    match (layout.sample, layout.order) {
        (
            SampleType::U8,
            ChannelOrder::Rgba | ChannelOrder::Bgra | ChannelOrder::Rgbx | ChannelOrder::Bgrx,
        ) => {
            for y in 0..h {
                let row_start = y * stride;
                for x in 0..w {
                    let off = row_start + x * 4;
                    let a = out[off + 3];
                    if a == 0 {
                        out[off] = 0;
                        out[off + 1] = 0;
                        out[off + 2] = 0;
                    } else if a < 255 {
                        let inv = 255.0 / a as f32;
                        for c in 0..3 {
                            out[off + c] = (out[off + c] as f32 * inv).round().min(255.0) as u8;
                        }
                    }
                }
            }
        }
        (SampleType::U16, ChannelOrder::Rgba | ChannelOrder::GrayAlpha) => {
            let bpp = layout.bytes_per_pixel();
            let alpha_offset = bpp - 2;
            for y in 0..h {
                let row_start = y * stride;
                for x in 0..w {
                    let off = row_start + x * bpp;
                    let a =
                        u16::from_ne_bytes([out[off + alpha_offset], out[off + alpha_offset + 1]]);
                    if a == 0 {
                        out[off..off + alpha_offset].fill(0);
                    } else if a < 65535 {
                        let inv = 65535.0 / a as f32;
                        for c in 0..alpha_offset / 2 {
                            let co = off + c * 2;
                            let v = u16::from_ne_bytes([out[co], out[co + 1]]);
                            let unpremul = (v as f32 * inv).round().min(65535.0) as u16;
                            out[co..co + 2].copy_from_slice(&unpremul.to_ne_bytes());
                        }
                    }
                }
            }
        }
        (SampleType::F32, ChannelOrder::Rgba | ChannelOrder::GrayAlpha) => {
            let bpp = layout.bytes_per_pixel();
            let alpha_offset = bpp - 4;
            for y in 0..h {
                let row_start = y * stride;
                for x in 0..w {
                    let off = row_start + x * bpp;
                    let a = f32::from_ne_bytes([
                        out[off + alpha_offset],
                        out[off + alpha_offset + 1],
                        out[off + alpha_offset + 2],
                        out[off + alpha_offset + 3],
                    ]);
                    if a <= 0.0 {
                        out[off..off + alpha_offset].fill(0);
                    } else if a < 1.0 {
                        let inv = 1.0 / a;
                        for c in 0..alpha_offset / 4 {
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

// ============================================================================
// HDR/PU21 funnel (`hdr-pu` feature)
// ============================================================================

/// [`funnel`] variant for the PU21 path: produces linear planes of
/// **absolute luminance in cd/m²**.
///
/// Accepted formats:
/// - `Linear` f32 — treated as absolute nits directly.
/// - `Pq`/`Hlg` (u8/u16/f32) — EOTF-decoded to nits. BT.2020 primaries
///   are *not* gamut-converted (the opsin matrix consumes whatever
///   primaries the source declares — same convention as zensim's PU path).
#[cfg(feature = "hdr-pu")]
pub(crate) fn funnel_nits(view: &PixelView<'_>) -> Result<PreparedInput, Ssimulacra2Error> {
    use crate::pipeline::pu21;

    let fmt = view.format();
    let (w, h) = (view.width() as usize, view.height() as usize);
    let npix = checked_dimensions(w, h)?;
    if fmt.layout.packing != BitPacking::Full {
        return Err(Ssimulacra2Error::UnsupportedInput(
            "bit-packed pixel layouts are not supported",
        ));
    }
    validate_alpha_layout(fmt)?;
    let color = extract_color(&fmt.color)?;
    if color.range != SignalRange::Full {
        return Err(Ssimulacra2Error::UnsupportedInput(
            "narrow/limited signal range — expand to full range upstream",
        ));
    }

    let pq = matches!(color.transfer, Transfer::Pq);
    let hlg = matches!(color.transfer, Transfer::Hlg(_));
    let linear = matches!(color.transfer, Transfer::Linear);
    if !(pq || hlg || linear) {
        return Err(Ssimulacra2Error::UnsupportedInput(
            "PU path needs Linear f32 (nits) or Pq/Hlg formats",
        ));
    }
    if linear
        && !(fmt.layout.sample == SampleType::F32
            && matches!(
                fmt.layout.order,
                ChannelOrder::Rgb
                    | ChannelOrder::Rgba
                    | ChannelOrder::Gray
                    | ChannelOrder::GrayAlpha
            ))
    {
        return Err(Ssimulacra2Error::UnsupportedInput(
            "linear transfer requires an f32 format",
        ));
    }

    // Premultiplied alpha un-multiplied in code-value space first.
    let data: Cow<'_, [u8]> = if color.alpha == AlphaMode::Premultiplied {
        Cow::Owned(unpremultiply(view))
    } else {
        Cow::Borrowed(view.as_strided_bytes())
    };
    let stride = view.stride_bytes();
    let alpha_present = matches!(color.alpha, AlphaMode::Straight | AlphaMode::Premultiplied);

    // Decode one channel's code value → [0,1], then to nits. For HLG the
    // per-pixel OOTF needs the full triple — handled in the px loop below.
    let decode01 = |v01: f32| -> f32 {
        if pq {
            pu21::pq_channel_to_nits(v01)
        } else {
            v01 // Linear: already nits; HLG handled per-triple
        }
    };

    let rgb_planes = |data: &[u8],
                      bpp: usize,
                      channel_bytes: usize,
                      read: &dyn Fn(&[u8]) -> f32,
                      gray: bool,
                      has_alpha: bool,
                      alpha_off: usize,
                      bgr: bool| {
        let mut planes = [
            Vec::with_capacity(npix),
            Vec::with_capacity(npix),
            Vec::with_capacity(npix),
        ];
        let mut alpha = has_alpha.then(|| Vec::with_capacity(npix));
        for y in 0..h {
            let row = &data[y * stride..];
            for px in row[..w * bpp].chunks_exact(bpp) {
                if hlg {
                    let triple = if gray {
                        [read(px); 3]
                    } else {
                        core::array::from_fn(|c| {
                            let src = if bgr { 2 - c } else { c };
                            read(&px[src * channel_bytes..])
                        })
                    };
                    let n = pu21::hlg_triple_to_nits(triple);
                    for c in 0..3 {
                        planes[c].push(if gray { n[0] } else { n[c] });
                    }
                } else if gray {
                    let g = decode01(read(px));
                    planes[0].push(g);
                    planes[1].push(g);
                    planes[2].push(g);
                } else {
                    let cbytes = channel_bytes;
                    for (c, plane) in planes.iter_mut().enumerate() {
                        let src = if bgr { 2 - c } else { c };
                        plane.push(decode01(read(&px[src * cbytes..])));
                    }
                }
                if let Some(a) = alpha.as_mut() {
                    a.push(read(&px[alpha_off..]));
                }
            }
        }
        (planes, alpha)
    };

    let bpp = fmt.layout.bytes_per_pixel();
    let (planes, alpha) = match (fmt.layout.sample, fmt.layout.order) {
        (
            SampleType::U8,
            ChannelOrder::Rgb
            | ChannelOrder::Rgba
            | ChannelOrder::Rgbx
            | ChannelOrder::Bgra
            | ChannelOrder::Bgrx
            | ChannelOrder::Gray,
        ) => {
            let (a_off, gray) = match fmt.layout.order {
                ChannelOrder::Gray => (0, true),
                _ => (3, false),
            };
            let rd = |p: &[u8]| p[0] as f32 * (1.0 / 255.0);
            let bgr = matches!(fmt.layout.order, ChannelOrder::Bgra | ChannelOrder::Bgrx);
            rgb_planes(&data, bpp, 1, &rd, gray, alpha_present, a_off, bgr)
        }
        (
            SampleType::U16,
            ChannelOrder::Rgb | ChannelOrder::Rgba | ChannelOrder::Gray | ChannelOrder::GrayAlpha,
        ) => {
            let gray = matches!(
                fmt.layout.order,
                ChannelOrder::Gray | ChannelOrder::GrayAlpha
            );
            let rd = |p: &[u8]| u16::from_ne_bytes([p[0], p[1]]) as f32 * (1.0 / 65535.0);
            rgb_planes(&data, bpp, 2, &rd, gray, alpha_present, bpp - 2, false)
        }
        (
            SampleType::F32,
            ChannelOrder::Rgb | ChannelOrder::Rgba | ChannelOrder::Gray | ChannelOrder::GrayAlpha,
        ) => {
            let gray = matches!(
                fmt.layout.order,
                ChannelOrder::Gray | ChannelOrder::GrayAlpha
            );
            let rd = |p: &[u8]| f32::from_ne_bytes([p[0], p[1], p[2], p[3]]);
            rgb_planes(&data, bpp, 4, &rd, gray, alpha_present, bpp - 4, false)
        }
        _ => {
            return Err(Ssimulacra2Error::UnsupportedInput(
                "PU path supports RGB/RGBA/Gray/GrayA layouts",
            ));
        }
    };

    Ok(PreparedInput::Linear {
        planes,
        alpha,
        width: w,
        height: h,
    })
}

#[cfg(test)]
mod tests;
