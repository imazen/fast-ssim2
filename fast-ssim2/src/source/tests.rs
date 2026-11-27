use super::*;
use zenpixels::{FormatCow, HlgMode, PixelFormat, SampleLayout};

fn graya16_srgb() -> PixelFormat {
    PixelFormat::new(
        SampleLayout::GRAYA16,
        ColorSpace::GRAY_SRGB.with_alpha(AlphaMode::Straight),
    )
}

fn grayaf32_linear() -> PixelFormat {
    PixelFormat::new(
        SampleLayout::GRAYAF32,
        ColorSpace::GRAY_LINEAR.with_alpha(AlphaMode::Straight),
    )
}

fn floats<'a>(values: &'a [f32], fmt: PixelFormat, stride: usize) -> PixelView<'a> {
    PixelView::new_with_format_cow(
        bytemuck::cast_slice(values),
        1,
        2,
        stride,
        FormatCow::from_owned(fmt),
    )
    .unwrap()
}

#[test]
fn linear_alpha_and_gray_respect_layout_and_stride() {
    let rgba = [
        0.2, 0.4, 0.6, 0.5, 99.0, 99.0, 99.0, 99.0, 0.8, 0.6, 0.4, 0.25,
    ];
    let p = funnel(&floats(&rgba, PixelFormat::RGBAF32_LINEAR, 32)).unwrap();
    let PreparedInput::Linear { planes, alpha, .. } = p else {
        panic!("linear expected")
    };
    assert_eq!(planes, [vec![0.2, 0.8], vec![0.4, 0.6], vec![0.6, 0.4]]);
    assert_eq!(alpha.unwrap(), [0.5, 0.25]);
    for premult in [false, true] {
        let gray = if premult {
            [0.125, 0.5, 99.0, 99.0, 0.1875, 0.25]
        } else {
            [0.25, 0.5, 99.0, 99.0, 0.75, 0.25]
        };
        let fmt = grayaf32_linear().with_alpha(if premult {
            AlphaMode::Premultiplied
        } else {
            AlphaMode::Straight
        });
        let p = funnel(&floats(&gray, fmt, 16)).unwrap();
        let PreparedInput::Linear { planes, alpha, .. } = p else {
            panic!("linear expected")
        };
        assert_eq!(
            planes,
            [vec![0.25, 0.75], vec![0.25, 0.75], vec![0.25, 0.75]]
        );
        assert_eq!(alpha.unwrap(), [0.5, 0.25]);
    }
    let gray = [0.25, 99.0, 0.75];
    let p = funnel(&floats(&gray, PixelFormat::GRAYF32_LINEAR, 8)).unwrap();
    let PreparedInput::Linear { planes, .. } = p else {
        panic!("linear expected")
    };
    assert_eq!(planes[0], [0.25, 0.75]);
}

#[cfg(feature = "hdr-pu")]
#[test]
fn hdr_layouts_decode_the_same_colors() {
    use crate::pipeline::pu21::{hlg_triple_to_nits, pq_channel_to_nits};
    for transfer in [
        Transfer::Pq,
        Transfer::Hlg(HlgMode::Scene),
        Transfer::Linear,
    ] {
        for bits in [8, 16, 32] {
            if transfer == Transfer::Linear && bits != 32 {
                continue;
            }
            for gray in [false, true] {
                for alpha in [false, true] {
                    if gray && alpha && bits == 8 {
                        continue;
                    }
                    let fmt = match (bits, gray, alpha) {
                        (8, false, false) => PixelFormat::RGB8_SRGB,
                        (8, false, true) => PixelFormat::RGBA8_SRGB,
                        (8, true, false) => PixelFormat::GRAY8_SRGB,
                        (16, false, false) => PixelFormat::RGB16_SRGB,
                        (16, false, true) => PixelFormat::RGBA16_SRGB,
                        (16, true, false) => PixelFormat::GRAY16_SRGB,
                        (16, true, true) => graya16_srgb(),
                        (32, false, false) => PixelFormat::RGBF32_LINEAR,
                        (32, false, true) => PixelFormat::RGBAF32_LINEAR,
                        (32, true, false) => PixelFormat::GRAYF32_LINEAR,
                        (32, true, true) => grayaf32_linear(),
                        _ => unreachable!(),
                    }
                    .with_transfer(transfer);
                    let values: &[u8] = if gray {
                        &[51, 128]
                    } else {
                        &[51, 102, 153, 128]
                    };
                    let count = (if gray { 1 } else { 3 }) + usize::from(alpha);
                    // Build aligned backing storage per sample width.
                    let mut buf_u8 = Vec::new();
                    let mut buf_u16 = Vec::new();
                    let mut buf_f32 = Vec::new();
                    let stride = count * (bits / 8) * 2;
                    let bytes: &[u8] = match bits {
                        8 => {
                            buf_u8.extend_from_slice(&values[..count]);
                            buf_u8.resize(stride, 0xff);
                            buf_u8.extend_from_slice(&values[..count]);
                            &buf_u8
                        }
                        16 => {
                            let px: Vec<u16> =
                                values[..count].iter().map(|&v| v as u16 * 257).collect();
                            buf_u16.extend_from_slice(&px);
                            buf_u16.resize(count * 2, u16::MAX);
                            buf_u16.extend_from_slice(&px);
                            bytemuck::cast_slice(&buf_u16)
                        }
                        _ => {
                            let px: Vec<f32> = values[..count]
                                .iter()
                                .map(|&v| v as f32 * (1.0 / 255.0))
                                .collect();
                            buf_f32.extend_from_slice(&px);
                            buf_f32.resize(count * 2, f32::from_bits(u32::MAX));
                            buf_f32.extend_from_slice(&px);
                            bytemuck::cast_slice(&buf_f32)
                        }
                    };
                    let view = PixelView::new(bytes, 1, 2, stride, &fmt).unwrap();
                    let PreparedInput::Linear {
                        planes, alpha: a, ..
                    } = funnel_nits(&view).unwrap()
                    else {
                        panic!("linear expected")
                    };
                    let rgb = if gray { [0.2; 3] } else { [0.2, 0.4, 0.6] };
                    let expected = match transfer {
                        Transfer::Pq => rgb.map(pq_channel_to_nits),
                        Transfer::Hlg(_) => hlg_triple_to_nits(rgb),
                        _ => rgb,
                    };
                    for c in 0..3 {
                        for &actual in &planes[c] {
                            assert!(
                                (actual - expected[c]).abs() <= 0.001 * expected[c].max(1.0),
                                "{fmt:?}: {actual} != {}",
                                expected[c]
                            );
                        }
                    }
                    assert_eq!(a.is_some(), alpha);
                    if bits == 8 && !gray && alpha {
                        let mut bgr = buf_u8.clone();
                        for y in 0..2 {
                            bgr.swap(y * stride, y * stride + 2);
                        }
                        let bgr_fmt = PixelFormat::BGRA8_SRGB.with_transfer(transfer);
                        let view = PixelView::new(&bgr, 1, 2, stride, &bgr_fmt).unwrap();
                        let PreparedInput::Linear {
                            planes: bgr_planes,
                            alpha: bgr_alpha,
                            ..
                        } = funnel_nits(&view).unwrap()
                        else {
                            panic!("linear expected")
                        };
                        assert_eq!(planes, bgr_planes);
                        assert_eq!(a, bgr_alpha);
                    }
                }
            }
        }
    }
}

#[test]
fn dimensions_validate_before_allocation() {
    assert_eq!(checked_dimensions(1, 1).unwrap(), 1);
    assert_eq!(
        checked_dimensions(16384, 16384).unwrap(),
        crate::MAX_IMAGE_PIXELS
    );
    for (w, h) in [(0, 8), (8, 0)] {
        assert_eq!(
            checked_dimensions(w, h),
            Err(Ssimulacra2Error::InvalidImageSize)
        );
    }
    // Original fits, but its padded shape does not.
    assert!(matches!(
        checked_dimensions(crate::MAX_IMAGE_PIXELS, 1),
        Err(Ssimulacra2Error::ImageTooLarge { .. })
    ));
    assert!(matches!(
        checked_dimensions(usize::MAX, 2),
        Err(Ssimulacra2Error::ImageTooLarge { .. })
    ));
    assert!(matches!(
        checked_dimensions(16385, 16384),
        Err(Ssimulacra2Error::ImageTooLarge { .. })
    ));
}

#[test]
fn native_hdr_gray_power_keeps_cpp_rounding() {
    // Independent C++ frontend-gray capture, HLG at 1000 nits, code 271/65535.
    // f64::powf followed by an f32 cast differs by one bit from this oracle.
    let samples = [f32::from_bits(0x3b878088)];
    let fmt = PixelFormat::GRAYF32_LINEAR.with_transfer(Transfer::Hlg(HlgMode::Scene));
    let image = PixelView::new(bytemuck::cast_slice(&samples), 1, 1, 4, &fmt).unwrap();
    let prepared = funnel(&image).unwrap();
    let linear = crate::linearize_prepared(&prepared, 1, 0.5);
    for plane in &linear {
        assert_eq!(plane[0].to_bits(), 0x3508bc8e);
    }
}

#[test]
fn native_hdr_frontend_matches_cpp() {
    let mut format = PixelFormat::RGBF32_LINEAR;
    let mut target = 255.0;
    let mut bg = 0.5;
    for line in include_str!("native_hdr_cpp.tsv")
        .lines()
        .chain(include_str!("native_hdr_gray_cpp.tsv").lines())
        .chain(include_str!("native_hdr_signed_cpp.tsv").lines())
        .chain(include_str!("native_hdr_low_peak_cpp.tsv").lines())
        .chain(include_str!("native_hdr_low_peak_gray_cpp.tsv").lines())
    {
        if line.starts_with('#') {
            continue;
        }
        if let Some(case) = line
            .strip_prefix("case ")
            .or_else(|| line.strip_prefix("case-gray "))
        {
            let fields: Vec<_> = case.split_whitespace().collect();
            target = fields[2].parse::<f32>().unwrap();
            bg = fields[3].parse::<f32>().unwrap();
            format = if line.starts_with("case-gray") {
                grayaf32_linear()
            } else {
                PixelFormat::RGBAF32_LINEAR
            }
            .with_transfer(match fields[0] {
                "8" => Transfer::Linear,
                "16" => Transfer::Pq,
                "18" => Transfer::Hlg(HlgMode::Scene),
                _ => panic!("transfer"),
            })
            .with_primaries(match fields[1] {
                "1" => Primaries::Bt709,
                "9" => Primaries::Bt2020,
                "11" => Primaries::DisplayP3,
                _ => panic!("primaries"),
            })
            .with_alpha(if fields[4] == "1" {
                AlphaMode::Straight
            } else {
                AlphaMode::Opaque
            })
            .with_white(DiffuseWhite::new(target).unwrap());
            continue;
        }
        assert_eq!(line.len(), 80);
        let values: Vec<f32> = (0..10)
            .map(|i| f32::from_bits(u32::from_str_radix(&line[i * 8..i * 8 + 8], 16).unwrap()))
            .collect();
        let gray = [values[0], values[3]];
        let samples = if format.layout == SampleLayout::GRAYAF32 {
            &gray[..]
        } else {
            &values[..4]
        };
        let image = PixelView::new(
            bytemuck::cast_slice(samples),
            1,
            1,
            format.layout.bytes_per_pixel(),
            &format,
        )
        .unwrap();
        let prepared = funnel(&image).unwrap();
        let linear = crate::linearize_prepared(&prepared, 1, bg);
        for c in 0..3 {
            assert_eq!(
                linear[c][0].to_bits(),
                values[4 + c].to_bits(),
                "{format:?}, target={target}, bg={bg}, pixel={:?}, channel={c}",
                &values[..4]
            );
        }
        let matrix = crate::pipeline::xyb::intensity_matrix(target);
        let xyb = crate::pipeline::xyb::linear_rgb_to_xyb_pixel_matrix(
            [linear[0][0], linear[1][0], linear[2][0]],
            matrix,
        );
        let mut simd = core::array::from_fn(|c| vec![linear[c][0]; 9]);
        crate::pipeline::simd::planes_to_positive_xyb_matrix_simd(&mut simd, matrix);
        for c in 0..3 {
            assert_eq!(
                xyb[c].to_bits(),
                values[7 + c].to_bits(),
                "scalar XYB {format:?}, target={target}, channel={c}"
            );
            for &value in &simd[c] {
                assert_eq!(
                    value.to_bits(),
                    values[7 + c].to_bits(),
                    "SIMD XYB {format:?}, target={target}, channel={c}"
                );
            }
        }
    }
}

/// The caller opts into this external-oracle test through `just native-hdr-dense`.
/// Missing tools/configuration are errors when it is selected.
#[test]
#[ignore = "requires SSIM2_HDR_ORACLE; run just native-hdr-dense ORACLE"]
#[cfg(not(target_arch = "wasm32"))]
fn native_hdr_dense_frontend_matches_cpp() {
    use std::{
        fmt::Write,
        io::Write as IoWrite,
        process::{Command, Stdio},
    };
    let oracle =
        std::env::var_os("SSIM2_HDR_ORACLE").expect("SSIM2_HDR_ORACLE must name the C++ oracle");
    let mut checked = 0usize;
    for (tf, transfer, targets) in [
        (8, Transfer::Linear, &[255.0, 203.0][..]),
        (16, Transfer::Pq, &[10000.0, 255.0][..]),
        (
            18,
            Transfer::Hlg(HlgMode::Scene),
            &[1000.0, 300.0, 203.0][..],
        ),
    ] {
        for (p, primaries, gray) in [
            (1, Primaries::Bt709, false),
            (9, Primaries::Bt2020, false),
            (11, Primaries::DisplayP3, false),
            (1, Primaries::Bt709, true),
        ] {
            for &target in targets {
                for alpha in [false, true] {
                    for bg in if alpha { &[0.1, 0.9][..] } else { &[0.5][..] } {
                        let data: Vec<u16> = (0..=u16::MAX)
                            .flat_map(|v| {
                                [
                                    v,
                                    v.wrapping_mul(37).wrapping_add(13),
                                    v.wrapping_mul(691).wrapping_add(11),
                                    v.wrapping_mul(11),
                                ]
                            })
                            .collect();
                        let mode = if gray { "frontend-gray" } else { "frontend" };
                        let mut request = format!(
                            "{mode}\n{tf} {p} {target} {bg} {} 65536 1\n",
                            u8::from(alpha)
                        );
                        for px in data.as_chunks::<4>().0 {
                            let bits = px.map(|v| (v as f32 * (1.0 / 65535.0)).to_bits());
                            writeln!(
                                request,
                                "{:08x} {:08x} {:08x} {:08x}",
                                bits[0],
                                bits[if gray { 0 } else { 1 }],
                                bits[if gray { 0 } else { 2 }],
                                bits[3]
                            )
                            .unwrap();
                        }
                        // Linear u16 is intentionally unsupported. Test the same decoded
                        // f32 samples for linear transfer; PQ/HLG exercise the u16 reader.
                        let samples: Vec<u16> = if gray {
                            data.as_chunks::<4>()
                                .0
                                .iter()
                                .flat_map(|px| [px[0], px[3]])
                                .collect()
                        } else {
                            data
                        };
                        let floats: Vec<f32> = samples
                            .iter()
                            .map(|&v| v as f32 * (1.0 / 65535.0))
                            .collect();
                        let fmt = match (transfer == Transfer::Linear, gray) {
                            (true, false) => PixelFormat::RGBAF32_LINEAR,
                            (true, true) => grayaf32_linear(),
                            (false, false) => PixelFormat::RGBA16_SRGB,
                            (false, true) => graya16_srgb(),
                        };
                        let fmt = fmt
                            .with_transfer(transfer)
                            .with_primaries(primaries)
                            .with_alpha(if alpha {
                                AlphaMode::Straight
                            } else {
                                AlphaMode::Opaque
                            })
                            .with_white(DiffuseWhite::new(target).unwrap());
                        let bytes = if transfer == Transfer::Linear {
                            bytemuck::cast_slice(&floats)
                        } else {
                            bytemuck::cast_slice(&samples)
                        };
                        let image = PixelView::new(
                            bytes,
                            65536,
                            1,
                            65536 * fmt.layout.bytes_per_pixel(),
                            &fmt,
                        )
                        .unwrap();
                        let prepared = funnel(&image).unwrap();
                        let linear = crate::linearize_prepared(&prepared, 65536, *bg);
                        let mut xyb = linear.clone();
                        crate::pipeline::simd::planes_to_positive_xyb_matrix_simd(
                            &mut xyb,
                            crate::pipeline::xyb::intensity_matrix(target),
                        );
                        let mut child = Command::new(&oracle)
                            .stdin(Stdio::piped())
                            .stdout(Stdio::piped())
                            .spawn()
                            .expect("start oracle");
                        child
                            .stdin
                            .take()
                            .unwrap()
                            .write_all(request.as_bytes())
                            .unwrap();
                        let output = child.wait_with_output().unwrap();
                        assert!(output.status.success());
                        let text = String::from_utf8(output.stdout).unwrap();
                        let mut lines = text.lines();
                        for i in 0..65536 {
                            let line = lines.next().expect("missing oracle pixel");
                            let expected: Vec<_> = line
                                .split_whitespace()
                                .map(|s| u32::from_str_radix(s, 16).unwrap())
                                .collect();
                            assert_eq!(expected.len(), 6);
                            for c in 0..3 {
                                assert_eq!(
                                    linear[c][i].to_bits(),
                                    expected[c],
                                    "linear: {fmt:?} target={target} bg={bg} code={i} channel={c}"
                                );
                                assert_eq!(
                                    xyb[c][i].to_bits(),
                                    expected[c + 3],
                                    "XYB: {fmt:?} target={target} bg={bg} code={i} channel={c}"
                                );
                            }
                            checked += 1;
                        }
                        assert!(lines.next().is_none());
                    }
                }
            }
        }
        println!("native frontend tf={tf}: {checked} cumulative pixels exactly match C++");
    }
}
