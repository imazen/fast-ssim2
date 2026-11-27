//! Deterministic inputs shared by native-HDR regression tests and oracle capture.
use fast_ssim2::{PixelFormat, PixelView, Transfer};
use zenpixels::{AlphaMode, DiffuseWhite, FormatCow, HlgMode, Primaries};

pub struct Case {
    pub transfer: u8,
    pub primaries: u8,
    pub target: f32,
    pub alpha: u8,
    pub width: usize,
    pub height: usize,
}

impl Case {
    pub fn samples(&self, distorted: bool) -> Vec<f32> {
        let mut state = 0x93a5_c82du32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        (0..self.width * self.height)
            .flat_map(|i| {
                let rgb: [f32; 3] = core::array::from_fn(|c| {
                    let code = (next() & 65535) as f32 * (1.0 / 65535.0);
                    let code = if distorted && (i / 7 + c) % 3 == 0 {
                        (code * 16.0).round() * (1.0 / 16.0)
                    } else {
                        code
                    };
                    if self.transfer == 8 {
                        code * if i % 5 == 0 { 8.0 } else { 0.25 }
                    } else {
                        code
                    }
                });
                let a = (next() & 255) as f32 * (1.0 / 255.0);
                let has_alpha = if distorted {
                    self.alpha & 2 != 0
                } else {
                    self.alpha & 1 != 0
                };
                [rgb[0], rgb[1], rgb[2], if has_alpha { a } else { 1.0 }]
            })
            .collect()
    }

    pub fn image<'a>(&self, data: &'a [f32], distorted: bool) -> PixelView<'a> {
        let has_alpha = if distorted {
            self.alpha & 2 != 0
        } else {
            self.alpha & 1 != 0
        };
        let fmt = PixelFormat::RGBAF32_LINEAR
            .with_transfer(match self.transfer {
                8 => Transfer::Linear,
                16 => Transfer::Pq,
                18 => Transfer::Hlg(HlgMode::Scene),
                _ => panic!("transfer"),
            })
            .with_primaries(match self.primaries {
                1 => Primaries::Bt709,
                9 => Primaries::Bt2020,
                11 => Primaries::DisplayP3,
                _ => panic!("primaries"),
            })
            .with_alpha(if has_alpha {
                AlphaMode::Straight
            } else {
                AlphaMode::Opaque
            })
            .with_white(DiffuseWhite::new(self.target).unwrap());
        PixelView::new_with_format_cow(
            bytemuck::cast_slice(data),
            self.width as u32,
            self.height as u32,
            self.width * 16,
            FormatCow::from_owned(fmt),
        )
        .unwrap()
    }
}
