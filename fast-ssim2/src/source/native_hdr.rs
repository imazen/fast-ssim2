// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Native HDR transfer/color port: BSD-3-Clause, see ../../LICENSE.libjxl.

//! Native Cloudinary SSIMULACRA2 2.1 color frontend. See
//! `docs/native-hdr-parity.md` for the independent oracle and provenance.
//! PQ coefficients and HLG operation order follow bundled libjxl (BSD-3-Clause).

use super::*;
use zenpixels::ColorPrimaries;

pub(crate) struct NativeInput {
    encoded: [Vec<f32>; 3],
    pub(crate) alpha: Option<Vec<f32>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) intensity_target: f32,
    transfer: TransferFunction,
    primaries: ColorPrimaries,
    gray: bool,
}

impl NativeInput {
    pub(crate) fn new(slice: &PixelSlice<'_>) -> Result<Self, Ssimulacra2Error> {
        validate_alpha_layout(slice)?;
        let d = slice.descriptor();
        if d.signal_range != SignalRange::Full {
            return Err(Ssimulacra2Error::UnsupportedInput(
                "native HDR requires full signal range",
            ));
        }
        let (bytes, gray, bgr) = match d.format {
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgbx8 => (1, false, false),
            PixelFormat::Bgra8 | PixelFormat::Bgrx8 => (1, false, true),
            PixelFormat::Gray8 => (1, true, false),
            PixelFormat::Rgb16 | PixelFormat::Rgba16 => (2, false, false),
            PixelFormat::Gray16 | PixelFormat::GrayA16 => (2, true, false),
            PixelFormat::RgbF32 | PixelFormat::RgbaF32 => (4, false, false),
            PixelFormat::GrayF32 | PixelFormat::GrayAF32 => (4, true, false),
            _ => {
                return Err(Ssimulacra2Error::UnsupportedInput(
                    "unsupported native HDR pixel layout",
                ));
            }
        };
        if d.transfer == TransferFunction::Linear && bytes != 4 {
            return Err(Ssimulacra2Error::UnsupportedInput(
                "linear transfer requires an f32 format",
            ));
        }
        if !gray
            && !matches!(
                d.primaries,
                ColorPrimaries::Bt709 | ColorPrimaries::Bt2020 | ColorPrimaries::DisplayP3
            )
        {
            return Err(Ssimulacra2Error::UnsupportedInput(
                "native HDR supports BT.709, BT.2020 and Display P3 primaries",
            ));
        }
        let default_target = match d.transfer {
            TransferFunction::Pq => 10000.0,
            TransferFunction::Hlg => 1000.0,
            _ => 255.0,
        };
        let intensity_target = slice
            .color_context()
            .and_then(|c| c.diffuse_white)
            .map_or(default_target, |w| w.nits());
        if !intensity_target.is_finite() || intensity_target <= 0.0 {
            return Err(Ssimulacra2Error::UnsupportedInput(
                "luminance anchor must be finite and positive",
            ));
        }
        let (width, height) = (slice.width() as usize, slice.rows() as usize);
        let n = checked_dimensions(width, height)?;
        let data = if d.alpha == Some(AlphaMode::Premultiplied) {
            Cow::Owned(unpremultiply(slice))
        } else {
            Cow::Borrowed(slice.as_strided_bytes())
        };
        let bpp = d.format.bytes_per_pixel();
        let read = |p: &[u8]| match bytes {
            1 => p[0] as f32 * (1.0 / 255.0),
            2 => u16::from_ne_bytes([p[0], p[1]]) as f32 * (1.0 / 65535.0),
            _ => f32::from_ne_bytes([p[0], p[1], p[2], p[3]]),
        };
        let mut encoded = core::array::from_fn(|_| Vec::with_capacity(n));
        let mut alpha = matches!(
            d.alpha,
            Some(AlphaMode::Straight | AlphaMode::Premultiplied)
        )
        .then(|| Vec::with_capacity(n));
        for y in 0..height {
            for px in data[y * slice.stride()..][..width * bpp].chunks_exact(bpp) {
                for (c, plane) in encoded.iter_mut().enumerate() {
                    let c = if gray {
                        0
                    } else if bgr {
                        2 - c
                    } else {
                        c
                    };
                    plane.push(read(&px[c * bytes..]));
                }
                if let Some(a) = &mut alpha {
                    a.push(read(&px[(if gray { 1 } else { 3 }) * bytes..]));
                }
            }
        }
        Ok(Self {
            encoded,
            alpha,
            width,
            height,
            intensity_target,
            transfer: d.transfer,
            primaries: d.primaries,
            gray,
        })
    }

    pub(crate) fn linearize(
        &self,
        bg: f32,
        stop: &dyn enough::Stop,
    ) -> Result<[Vec<f32>; 3], enough::StopReason> {
        let mut planes = self.encoded.clone();
        for y in 0..self.height {
            if y % 64 == 0 {
                stop.check()?;
            }
            for i in y * self.width..(y + 1) * self.width {
                let mut rgb = core::array::from_fn(|c| {
                    let v = self.encoded[c][i];
                    match &self.alpha {
                        Some(a) => a[i] * v + (1.0 - a[i]) * bg,
                        None => v,
                    }
                });
                rgb = match self.transfer {
                    TransferFunction::Pq => {
                        let scale = if self.intensity_target == 10000.0 {
                            1.0
                        } else {
                            10000.0 / self.intensity_target
                        };
                        rgb.map(|v| scale * pq_decode(v))
                    }
                    TransferFunction::Hlg => {
                        hlg_decode(rgb, self.primaries, self.intensity_target, self.gray)
                    }
                    _ => rgb,
                };
                if !self.gray {
                    rgb = gamut_to_srgb(rgb, self.primaries);
                }
                for c in 0..3 {
                    planes[c][i] = rgb[c];
                }
            }
        }
        Ok(planes)
    }
}

fn pq_decode(v: f32) -> f32 {
    let x = v.abs();
    let x = x.mul_add(x, x);
    // libjxl's fused 4/4 rational polynomial on x+x*x, not pow(ST 2084).
    let p = 0.550_034_9_f32
        .mul_add(x, 2.645_531_7)
        .mul_add(x, 0.738_602_3)
        .mul_add(x, -0.006_235_531)
        .mul_add(x, 0.000_262_975_66);
    let q = 2.677_187_7_f32
        .mul_add(x, -33.907_887)
        .mul_add(x, 174.364_67)
        .mul_add(x, -428.736_82)
        .mul_add(x, 421.350_1);
    (p / q).abs().copysign(v)
}

#[inline]
fn hlg_inverse_oetf(v: f32) -> f32 {
    // TF_HLG::InvOETF returns +0.0 for either signed zero before restoring
    // the sign of nonzero inputs.
    if v == 0.0 {
        return 0.0;
    }
    let e = f64::from(v.abs());
    let scene = if e <= 0.5 {
        e * e * (1.0 / 3.0)
    } else {
        (((e - 0.559_910_729_5) * (1.0 / 0.178_832_77)).exp() + (1.0 - 4.0 * 0.178_832_77))
            * (1.0 / 12.0)
    };
    (scene as f32).copysign(v)
}

fn hlg_decode(rgb: [f32; 3], primaries: ColorPrimaries, target: f32, gray: bool) -> [f32; 3] {
    // Ingest expands grayscale into identical channels. Decode its single
    // stored channel once, as the reference CMS does, then expand the result.
    let mut rgb = if gray {
        [hlg_inverse_oetf(rgb[0]); 3]
    } else {
        rgb.map(hlg_inverse_oetf)
    };
    if (295.0..=305.0).contains(&target) {
        return rgb;
    }
    let gamma = 1.2 * 1.111_f32.powf((target * 1e-3).log2());
    if gray {
        // Keep the oracle's powf polynomial and rounding across platforms.
        return [super::native_hdr_pow::powf(rgb[0], gamma); 3];
    }
    let lum = match primaries {
        ColorPrimaries::Bt2020 => [0x3e8680a4, 0x3f2d9148, 0x3d72e65a],
        ColorPrimaries::DisplayP3 => [0x3e6a784e, 0x3f3115c7, 0x3da2612d],
        _ => [0x3e59bee3, 0x3f37153b, 0x3d93d85f],
    }
    .map(f32::from_bits);
    let y = (rgb[0] * lum[0] + rgb[1] * lum[1]) + rgb[2] * lum[2];
    let ratio = y.powf(gamma - 1.0);
    if ratio.is_finite() {
        rgb = rgb.map(|v| v * ratio);
        if gamma < 1.0 {
            let max = rgb[0].max(rgb[1].max(rgb[2]));
            if max > 1.0 {
                rgb = rgb.map(|v| v * (1.0 / max));
            }
        }
    }
    rgb
}

#[allow(clippy::excessive_precision)] // Exact oracle-captured f64 coefficients.
fn gamut_to_srgb(rgb: [f32; 3], primaries: ColorPrimaries) -> [f32; 3] {
    // Inverse destination ICC colorants times source ICC colorants, computed
    // in f64. These include the reference profiles' 15.16 quantization.
    let m = match primaries {
        ColorPrimaries::Bt2020 => [
            [
                1.6605489717172115,
                -0.58764255219249795,
                -0.07284140369155373,
            ],
            [
                -0.12458788429670342,
                1.1328994991842685,
                -0.008355278081633815,
            ],
            [
                -0.018110241302956312,
                -0.10057918868756022,
                1.1187154637140566,
            ],
        ],
        ColorPrimaries::DisplayP3 => [
            [
                1.2249939720802692,
                -0.22493976928479925,
                -3.7013567941057701e-05,
            ],
            [
                -0.04209546229039899,
                1.0420416884814589,
                2.5046319613523937e-05,
            ],
            [
                -0.019598792376690974,
                -0.078630222497538468,
                1.0982539500628608,
            ],
        ],
        _ => return rgb,
    };
    m.map(|r| {
        // The reference CMS starts its sum at +0.0. Keep that first add:
        // omitting it can retain -0.0 when all three products are negative zero.
        (((0.0 + r[0] * f64::from(rgb[0])) + r[1] * f64::from(rgb[1])) + r[2] * f64::from(rgb[2]))
            as f32
    })
}
