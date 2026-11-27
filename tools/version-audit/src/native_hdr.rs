//! In-memory, interleaved native-HDR regression audit. No decoding is timed.
use std::{sync::Arc, time::Duration};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Rgb8,
    Rgba8,
    Linear,
    Pq,
    Hlg,
    HlgGray,
}

impl Kind {
    fn native(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg | Self::HlgGray)
    }

    fn descriptor_legacy(self) -> zenpixels::PixelDescriptor {
        match self {
            Self::Rgb8 => zenpixels::PixelDescriptor::RGB8_SRGB,
            Self::Rgba8 => zenpixels::PixelDescriptor::RGBA8_SRGB,
            Self::Linear => zenpixels::PixelDescriptor::RGBF32_LINEAR,
            Self::Pq | Self::Hlg => zenpixels::PixelDescriptor::RGB16_SRGB
                .with_primaries(zenpixels::ColorPrimaries::Bt2020)
                .with_transfer(if matches!(self, Self::Pq) {
                    zenpixels::TransferFunction::Pq
                } else {
                    zenpixels::TransferFunction::Hlg
                }),
            Self::HlgGray => {
                zenpixels::PixelDescriptor::GRAY16_SRGB.with_transfer(zenpixels::TransferFunction::Hlg)
            }
        }
    }

    fn format_current(self) -> current::PixelFormat {
        match self {
            Self::Rgb8 => current::PixelFormat::RGB8_SRGB,
            Self::Rgba8 => current::PixelFormat::RGBA8_SRGB,
            Self::Linear => current::PixelFormat::RGBF32_LINEAR,
            Self::Pq => current::PixelFormat::RGB16_BT2100_PQ,
            Self::Hlg => current::PixelFormat::RGB16_BT2100_HLG_SCENE,
            Self::HlgGray => current::PixelFormat::GRAY16_SRGB
                .with_transfer(current::Transfer::Hlg(zenpixels_05::HlgMode::Scene)),
        }
    }
}

struct Pair {
    source: Vec<u8>,
    distorted: Vec<u8>,
    width: u32,
    height: u32,
    kind: Kind,
}

impl Pair {
    fn new(width: u32, height: u32, kind: Kind) -> Self {
        let mut source = Vec::new();
        let mut distorted = Vec::new();
        let mut state = 0x9151_cafe_dead_beef_u64;
        for i in 0..width * height {
            for c in 0..match kind {
                Kind::Rgba8 => 4,
                Kind::HlgGray => 1,
                _ => 3,
            } {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Noise plus abrupt constant patches; no smooth gradients.
                let sample = if (i / width / 16 + i % width / 16).is_multiple_of(3) {
                    [8031, 27613, 50123, 38001][c]
                } else {
                    (state >> 32) as u16
                };
                let other = if c == 3 {
                    sample
                } else {
                    sample.saturating_add(511)
                };
                match kind {
                    Kind::Rgb8 | Kind::Rgba8 => {
                        source.push((sample >> 8) as u8);
                        distorted.push((other >> 8) as u8);
                    }
                    Kind::Linear => {
                        source.extend_from_slice(
                            &(f32::from(sample) * (1.0 / 65535.0)).to_ne_bytes(),
                        );
                        distorted
                            .extend_from_slice(&(f32::from(other) * (1.0 / 65535.0)).to_ne_bytes());
                    }
                    _ => {
                        source.extend_from_slice(&sample.to_ne_bytes());
                        distorted.extend_from_slice(&other.to_ne_bytes());
                    }
                }
            }
        }
        Self {
            source,
            distorted,
            width,
            height,
            kind,
        }
    }

    fn image_legacy<'a>(&self, bytes: &'a [u8]) -> zenpixels::PixelSlice<'a> {
        let descriptor = self.kind.descriptor_legacy();
        zenpixels::PixelSlice::new(
            bytes,
            self.width,
            self.height,
            self.width as usize * descriptor.format.bytes_per_pixel(),
            descriptor,
        )
        .unwrap()
    }

    fn image_current<'a>(&self, bytes: &'a [u8]) -> current::PixelView<'a> {
        let format = self.kind.format_current();
        current::PixelView::new_with_format_cow(
            bytes,
            self.width,
            self.height,
            self.width as usize * format.layout.bytes_per_pixel(),
            zenpixels_05::FormatCow::from_owned(format),
        )
        .unwrap()
    }
}

macro_rules! score_legacy {
    ($module:ident, $pair:expr) => {{
        let pair = $pair;
        $module::compute_ssimulacra2_with_config(
            &pair.image_legacy(&pair.source),
            &pair.image_legacy(&pair.distorted),
            &$module::Ssimulacra2Config::simd(),
        )
        .unwrap()
    }};
}

macro_rules! score_current {
    ($pair:expr) => {{
        let pair = $pair;
        current::compute_ssimulacra2_with_config(
            &pair.image_current(&pair.source),
            &pair.image_current(&pair.distorted),
            &current::Ssimulacra2Config::simd(),
        )
        .unwrap()
    }};
}

pub(super) fn run() {
    if let Ok(kind) = std::env::var("AUDIT_PROFILE") {
        let kind = match kind.as_str() {
            "Hlg" => Kind::Hlg,
            "HlgGray" => Kind::HlgGray,
            "Pq" => Kind::Pq,
            "Rgb8" => Kind::Rgb8,
            _ => panic!("AUDIT_PROFILE must be Hlg, HlgGray, Pq or Rgb8"),
        };
        let pair = Pair::new(1920, 1080, kind);
        for _ in 0..32 {
            std::hint::black_box(score_current!(std::hint::black_box(&pair)));
        }
        return;
    }
    let rounds: usize = std::env::var("AUDIT_ROUNDS")
        .unwrap_or_else(|_| "25".into())
        .parse()
        .unwrap();
    assert!(rounds > 0);
    let result = zenbench::run_gated(zenbench::GateConfig::default(), |suite| {
        if let Ok(filter) = std::env::var("AUDIT_GROUP") {
            suite.set_group_filter(filter);
        }
        for (w, h) in [(32, 32), (320, 240), (1920, 1080)] {
            for kind in [
                Kind::Rgb8,
                Kind::Rgba8,
                Kind::Linear,
                Kind::Pq,
                Kind::Hlg,
                Kind::HlgGray,
            ] {
                let pair = Arc::new(Pair::new(w, h, kind));
                let before = if kind.native() {
                    score_legacy!(initial_hdr, &pair)
                } else {
                    score_legacy!(pre_hdr, &pair)
                };
                let after = score_current!(&pair);
                // Changed SSIM rounding can change score bits; timing must
                // still compare the same input and a successful nonidentity score.
                assert!(before.is_finite() && after.is_finite());
                println!("score\t{kind:?}\t{w}x{h}\tbefore={before:.12}\tafter={after:.12}");
                suite.compare(format!("{kind:?}_{w}x{h}"), |g| {
                    let label = if kind.native() {
                        "initial_hdr"
                    } else {
                        "pre_hdr"
                    };
                    g.baseline(label);
                    g.config()
                        .min_rounds(rounds)
                        .max_rounds(rounds)
                        .warmup_time(Duration::from_millis(100));
                    g.config().auto_rounds = false;
                    g.config().max_time = Duration::from_secs(90);
                    g.config().max_wall_time = Duration::from_secs(120);
                    let old = Arc::clone(&pair);
                    if kind.native() {
                        g.bench(label, move |b| b.iter(|| score_legacy!(initial_hdr, &old)));
                    } else {
                        g.bench(label, move |b| b.iter(|| score_legacy!(pre_hdr, &old)));
                    }
                    let new = Arc::clone(&pair);
                    g.bench("current", move |b| b.iter(|| score_current!(&new)));
                });
                if w == 1920 && matches!(kind, Kind::Rgb8 | Kind::Hlg) {
                    suite.compare(format!("cached_{kind:?}_{w}x{h}"), |g| {
                        g.config().min_rounds(rounds).max_rounds(rounds);
                        g.config().auto_rounds = false;
                        g.config().max_time = Duration::from_secs(90);
                        g.config().max_wall_time = Duration::from_secs(120);
                        if kind.native() {
                            g.baseline("initial_hdr");
                            let old = Arc::clone(&pair);
                            let reference = initial_hdr::Ssimulacra2Reference::new_with_config(
                                &old.image_legacy(&old.source),
                                &initial_hdr::Ssimulacra2Config::simd(),
                            )
                            .unwrap();
                            g.bench("initial_hdr", move |b| {
                                b.iter(|| {
                                    reference
                                        .compare_with_config(
                                            &old.image_legacy(&old.distorted),
                                            &initial_hdr::Ssimulacra2Config::simd(),
                                        )
                                        .unwrap()
                                })
                            });
                        } else {
                            g.baseline("pre_hdr");
                            let old = Arc::clone(&pair);
                            let reference = pre_hdr::Ssimulacra2Reference::new_with_config(
                                &old.image_legacy(&old.source),
                                &pre_hdr::Ssimulacra2Config::simd(),
                            )
                            .unwrap();
                            g.bench("pre_hdr", move |b| {
                                b.iter(|| {
                                    reference
                                        .compare_with_config(
                                            &old.image_legacy(&old.distorted),
                                            &pre_hdr::Ssimulacra2Config::simd(),
                                        )
                                        .unwrap()
                                })
                            });
                        }
                        let new = Arc::clone(&pair);
                        let reference = current::Ssimulacra2Reference::new_with_config(
                            &new.image_current(&new.source),
                            &current::Ssimulacra2Config::simd(),
                        )
                        .unwrap();
                        g.bench("current", move |b| {
                            b.iter(|| {
                                reference
                                    .compare_with_config(
                                        &new.image_current(&new.distorted),
                                        &current::Ssimulacra2Config::simd(),
                                    )
                                    .unwrap()
                            })
                        });
                    });
                }
            }
        }
    });
    zenbench::postprocess_result(&result);
}
