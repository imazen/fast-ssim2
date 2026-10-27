//! Compare identical PNG inputs and in-memory pixels against both C++ tools.
use fast_ssim2::{
    PixelDescriptor, PixelSlice, Ssimulacra2Config, Ssimulacra2Reference, StripConfig,
    TransferFunction, compute_ssimulacra2, compute_ssimulacra2_with_config,
};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    fmt::Write,
    fs,
    io::Write as IoWrite,
    path::Path,
    process::{Command, Stdio},
};
use zenpixels::ColorPrimaries;

#[derive(Clone, Copy)]
struct Format {
    name: &'static str,
    transfer: u8,
    primaries: u8,
    depth: u8,
    alpha: bool,
    gray: bool,
    floats: bool,
}

impl Format {
    fn descriptor(self) -> PixelDescriptor {
        let base = match (self.gray, self.alpha, self.depth, self.floats) {
            (true, false, _, false) => PixelDescriptor::GRAY16_SRGB,
            (false, false, 8, false) => PixelDescriptor::RGB8_SRGB,
            (false, true, 8, false) => PixelDescriptor::RGBA8_SRGB,
            (false, false, 16, false) => PixelDescriptor::RGB16_SRGB,
            (false, true, 16, false) => PixelDescriptor::RGBA16_SRGB,
            (false, false, _, true) => PixelDescriptor::RGBF32_LINEAR,
            _ => panic!("unsupported spot format"),
        };
        base.with_transfer(match self.transfer {
            8 => TransferFunction::Linear,
            13 => TransferFunction::Srgb,
            16 => TransferFunction::Pq,
            18 => TransferFunction::Hlg,
            _ => panic!("transfer"),
        })
        .with_primaries(match self.primaries {
            1 => ColorPrimaries::Bt709,
            9 => ColorPrimaries::Bt2020,
            _ => panic!("primaries"),
        })
    }
}

fn pipe(oracle: &Path, request: &str) -> String {
    let mut child = Command::new(oracle)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start C++ oracle");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap()
}

fn cli(oracle: &Path, source: &Path, distorted: &Path) -> f64 {
    let output = Command::new(oracle)
        .args([source, distorted])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}: {}",
        oracle.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

fn samples(format: Format, width: usize, height: usize, distorted: bool) -> Vec<u16> {
    let mut state = 0x53ab_19d1_u32;
    (0..width * height)
        .flat_map(|i| {
            let channels = if format.gray {
                1
            } else if format.alpha {
                4
            } else {
                3
            };
            (0..channels).map(move |c| (i, c))
        })
        .map(|(i, c)| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = if (i / width / 8 + i % width / 8).is_multiple_of(3) {
                [9013, 32019, 54031, 43021][c]
            } else {
                state as u16
            };
            let v = if distorted && c != 3 {
                v.saturating_add(753)
            } else {
                v
            };
            if format.depth == 8 { v >> 8 } else { v }
        })
        .collect()
}

fn write_png(
    path: &Path,
    format: Format,
    width: usize,
    height: usize,
    data: &[u16],
    profile: Vec<u8>,
) {
    let mut info = png::Info::with_size(width as u32, height as u32);
    info.bit_depth = if format.depth == 8 {
        png::BitDepth::Eight
    } else {
        png::BitDepth::Sixteen
    };
    info.color_type = if format.gray {
        png::ColorType::Grayscale
    } else if format.alpha {
        png::ColorType::Rgba
    } else {
        png::ColorType::Rgb
    };
    info.icc_profile = Some(Cow::Owned(profile));
    let encoder = png::Encoder::with_info(fs::File::create_new(path).unwrap(), info).unwrap();
    let mut writer = encoder.write_header().unwrap();
    let bytes: Vec<u8> = if format.depth == 8 {
        data.iter().map(|&v| v as u8).collect()
    } else {
        data.iter().flat_map(|v| v.to_be_bytes()).collect()
    };
    writer.write_image_data(&bytes).unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(
        args.len(),
        4,
        "usage: reference_spot RAW_CLOUD_ORACLE CLOUD_CLI_17G LIBJXL_CLI_17G NEW_OUTPUT_DIR"
    );
    let (raw, cloud, libjxl, output) = (
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    );
    fs::create_dir(output).expect("output directory must be new");
    let mut rows = String::new();
    for (label, oracle) in [
        ("raw_cloudinary", raw),
        ("cloudinary_cli", cloud),
        ("libjxl_cli", libjxl),
    ] {
        writeln!(
            rows,
            "# {label}_sha256 {}",
            Sha256::digest(fs::read(oracle).unwrap())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )
        .unwrap();
    }
    rows.push_str("# Scores are f64 hexadecimal bits; default/scalar/SIMD/cache must agree.\n");
    rows.push_str("format\twidth\theight\tdefault\tscalar\tsimd\traw_cloudinary\tcloudinary_cli\tlibjxl_cli\tstrips\n");
    let formats = [
        Format {
            name: "rgb8-srgb",
            transfer: 13,
            primaries: 1,
            depth: 8,
            alpha: false,
            gray: false,
            floats: false,
        },
        Format {
            name: "rgba8-srgb",
            transfer: 13,
            primaries: 1,
            depth: 8,
            alpha: true,
            gray: false,
            floats: false,
        },
        Format {
            name: "rgb16-srgb",
            transfer: 13,
            primaries: 1,
            depth: 16,
            alpha: false,
            gray: false,
            floats: false,
        },
        Format {
            name: "rgbf32-srgb",
            transfer: 13,
            primaries: 1,
            depth: 16,
            alpha: false,
            gray: false,
            floats: true,
        },
        Format {
            name: "rgbf32-linear709",
            transfer: 8,
            primaries: 1,
            depth: 16,
            alpha: false,
            gray: false,
            floats: true,
        },
        Format {
            name: "rgbf32-linear2020",
            transfer: 8,
            primaries: 9,
            depth: 16,
            alpha: false,
            gray: false,
            floats: true,
        },
        Format {
            name: "rgb16-pq2020",
            transfer: 16,
            primaries: 9,
            depth: 16,
            alpha: false,
            gray: false,
            floats: false,
        },
        Format {
            name: "rgba16-pq2020",
            transfer: 16,
            primaries: 9,
            depth: 16,
            alpha: true,
            gray: false,
            floats: false,
        },
        Format {
            name: "rgb16-hlg2020",
            transfer: 18,
            primaries: 9,
            depth: 16,
            alpha: false,
            gray: false,
            floats: false,
        },
        Format {
            name: "gray16-hlg",
            transfer: 18,
            primaries: 1,
            depth: 16,
            alpha: false,
            gray: true,
            floats: false,
        },
    ];
    for (width, height) in [(17, 13), (513, 257)] {
        for format in formats {
            let target = match format.transfer {
                16 => 10000.0,
                18 => 1000.0,
                _ => 255.0,
            };
            let profile_hex = pipe(
                raw,
                &format!(
                    "profile\n{} {} {}\n",
                    format.transfer,
                    format.primaries,
                    u8::from(format.gray)
                ),
            );
            let profile_hex = profile_hex.trim();
            let profile: Vec<u8> = (0..profile_hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&profile_hex[i..i + 2], 16).unwrap())
                .collect();
            let data = [
                samples(format, width, height, false),
                samples(format, width, height, true),
            ];
            let float_data: [Vec<f32>; 2] = data.each_ref().map(|v| {
                v.iter()
                    .map(|&v| {
                        f32::from(v)
                            * if format.depth == 8 {
                                1.0 / 255.0
                            } else {
                                1.0 / 65535.0
                            }
                    })
                    .collect()
            });
            let bytes: [Vec<u8>; 2] = core::array::from_fn(|i| {
                if format.floats {
                    bytemuck::cast_slice(&float_data[i]).to_vec()
                } else if format.depth == 8 {
                    data[i].iter().map(|&v| v as u8).collect()
                } else {
                    bytemuck::cast_slice(&data[i]).to_vec()
                }
            });
            let descriptor = format.descriptor();
            let images = bytes.each_ref().map(|b| {
                PixelSlice::new(
                    b,
                    width as u32,
                    height as u32,
                    width * descriptor.format.bytes_per_pixel(),
                    descriptor,
                )
                .unwrap()
            });
            let default = compute_ssimulacra2(&images[0], &images[1]).unwrap();
            let scalar = compute_ssimulacra2_with_config(
                &images[0],
                &images[1],
                &Ssimulacra2Config::scalar(),
            )
            .unwrap();
            let simd =
                compute_ssimulacra2_with_config(&images[0], &images[1], &Ssimulacra2Config::simd())
                    .unwrap();
            let cache = Ssimulacra2Reference::new(&images[0])
                .unwrap()
                .compare(&images[1])
                .unwrap();
            for value in [scalar, simd, cache] {
                assert_eq!(value.to_bits(), default.to_bits());
            }
            let mut raw_score = f64::INFINITY;
            for &bg in if format.alpha {
                &[0.1, 0.9][..]
            } else {
                &[0.5][..]
            } {
                let mut request = String::from(if format.gray {
                    "score-gray\n"
                } else {
                    "score\n"
                });
                for image in &float_data {
                    writeln!(
                        request,
                        "{} {} {target} {bg} {} {width} {height}",
                        format.transfer,
                        format.primaries,
                        u8::from(format.alpha)
                    )
                    .unwrap();
                    let channels = if format.gray {
                        1
                    } else if format.alpha {
                        4
                    } else {
                        3
                    };
                    for px in image.chunks_exact(channels) {
                        let rgb = if format.gray {
                            [px[0]; 3]
                        } else {
                            [px[0], px[1], px[2]]
                        };
                        let a = if format.alpha { px[3] } else { 1.0 };
                        writeln!(
                            request,
                            "{:08x} {:08x} {:08x} {:08x}",
                            rgb[0].to_bits(),
                            rgb[1].to_bits(),
                            rgb[2].to_bits(),
                            a.to_bits()
                        )
                        .unwrap();
                    }
                }
                let score = pipe(raw, &request);
                raw_score = raw_score.min(f64::from_bits(
                    u64::from_str_radix(score.split_whitespace().next().unwrap(), 16).unwrap(),
                ));
            }
            let paths = [
                output.join(format!("{}-{width}x{height}-source.png", format.name)),
                output.join(format!("{}-{width}x{height}-distorted.png", format.name)),
            ];
            for i in 0..2 {
                write_png(&paths[i], format, width, height, &data[i], profile.clone());
            }
            let cpp_cloud = cli(cloud, &paths[0], &paths[1]);
            let cpp_libjxl = cli(libjxl, &paths[0], &paths[1]);
            let strips = match compute_ssimulacra2_with_config(
                &images[0],
                &images[1],
                &Ssimulacra2Config::default().with_strip(StripConfig::new(64)),
            ) {
                Ok(value) => {
                    assert!(matches!(format.transfer, 8 | 13) && format.primaries == 1);
                    format!("{:016x}", value.to_bits())
                }
                Err(fast_ssim2::Ssimulacra2Error::InvalidConfiguration(_)) => {
                    assert!(matches!(format.transfer, 16 | 18) || format.primaries != 1);
                    "unsupported".into()
                }
                Err(error) => panic!("strip scoring: {error}"),
            };
            let row = format!(
                "{}\t{width}\t{height}\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{strips}\n",
                format.name,
                default.to_bits(),
                scalar.to_bits(),
                simd.to_bits(),
                raw_score.to_bits(),
                cpp_cloud.to_bits(),
                cpp_libjxl.to_bits()
            );
            print!("{row}");
            rows.push_str(&row);
            fs::write(output.join("scores.tsv"), &rows).unwrap();
        }
    }
}
