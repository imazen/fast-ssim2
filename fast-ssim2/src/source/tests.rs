use super::*;
use zenpixels::PixelDescriptor;

fn floats(values: &[f32], desc: PixelDescriptor, stride: usize) -> PixelSlice<'_> {
    PixelSlice::new(bytemuck::cast_slice(values), 1, 2, stride, desc).unwrap()
}

#[test]
fn linear_alpha_and_gray_respect_layout_and_stride() {
    let rgba = [
        0.2, 0.4, 0.6, 0.5, 99.0, 99.0, 99.0, 99.0, 0.8, 0.6, 0.4, 0.25,
    ];
    let p = funnel(&floats(&rgba, PixelDescriptor::RGBAF32_LINEAR, 32)).unwrap();
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
        let desc = PixelDescriptor::GRAYAF32_LINEAR.with_alpha(Some(if premult {
            AlphaMode::Premultiplied
        } else {
            AlphaMode::Straight
        }));
        let p = funnel(&floats(&gray, desc, 16)).unwrap();
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
    let p = funnel(&floats(&gray, PixelDescriptor::GRAYF32_LINEAR, 8)).unwrap();
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
        TransferFunction::Pq,
        TransferFunction::Hlg,
        TransferFunction::Linear,
    ] {
        for bits in [8, 16, 32] {
            if transfer == TransferFunction::Linear && bits != 32 {
                continue;
            }
            for gray in [false, true] {
                for alpha in [false, true] {
                    if gray && alpha && bits == 8 {
                        continue;
                    }
                    let desc = match (bits, gray, alpha) {
                        (8, false, false) => PixelDescriptor::RGB8_SRGB,
                        (8, false, true) => PixelDescriptor::RGBA8_SRGB,
                        (8, true, false) => PixelDescriptor::GRAY8_SRGB,
                        (16, false, false) => PixelDescriptor::RGB16_SRGB,
                        (16, false, true) => PixelDescriptor::RGBA16_SRGB,
                        (16, true, false) => PixelDescriptor::GRAY16_SRGB,
                        (16, true, true) => PixelDescriptor::GRAYA16_SRGB,
                        (32, false, false) => PixelDescriptor::RGBF32_LINEAR,
                        (32, false, true) => PixelDescriptor::RGBAF32_LINEAR,
                        (32, true, false) => PixelDescriptor::GRAYF32_LINEAR,
                        (32, true, true) => PixelDescriptor::GRAYAF32_LINEAR,
                        _ => unreachable!(),
                    }
                    .with_transfer(transfer);
                    let values: &[u8] = if gray {
                        &[51, 128]
                    } else {
                        &[51, 102, 153, 128]
                    };
                    let count = (if gray { 1 } else { 3 }) + usize::from(alpha);
                    let mut pixel = Vec::new();
                    for &v in &values[..count] {
                        match bits {
                            8 => pixel.push(v),
                            16 => pixel.extend_from_slice(&(v as u16 * 257).to_ne_bytes()),
                            _ => pixel.extend_from_slice(&(v as f32 * (1.0 / 255.0)).to_ne_bytes()),
                        }
                    }
                    let stride = pixel.len() * 2;
                    let mut bytes = pixel.clone();
                    bytes.resize(stride, 0xff);
                    bytes.extend_from_slice(&pixel);
                    let slice = PixelSlice::new(&bytes, 1, 2, stride, desc).unwrap();
                    let PreparedInput::Linear {
                        planes, alpha: a, ..
                    } = funnel_nits(&slice).unwrap()
                    else {
                        panic!("linear expected")
                    };
                    let rgb = if gray { [0.2; 3] } else { [0.2, 0.4, 0.6] };
                    let expected = match transfer {
                        TransferFunction::Pq => rgb.map(pq_channel_to_nits),
                        TransferFunction::Hlg => hlg_triple_to_nits(rgb),
                        _ => rgb,
                    };
                    for c in 0..3 {
                        for &actual in &planes[c] {
                            assert!(
                                (actual - expected[c]).abs() <= 0.001 * expected[c].max(1.0),
                                "{desc:?}: {actual} != {}",
                                expected[c]
                            );
                        }
                    }
                    assert_eq!(a.is_some(), alpha);
                    if bits == 8 && !gray && alpha {
                        let mut bgr = bytes.clone();
                        for y in 0..2 {
                            bgr.swap(y * stride, y * stride + 2);
                        }
                        let desc = PixelDescriptor::BGRA8_SRGB.with_transfer(transfer);
                        let slice = PixelSlice::new(&bgr, 1, 2, stride, desc).unwrap();
                        let PreparedInput::Linear {
                            planes: bgr_planes,
                            alpha: bgr_alpha,
                            ..
                        } = funnel_nits(&slice).unwrap()
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
fn native_hdr_frontend_matches_cpp() {
    use std::sync::Arc;
    let mut descriptor = PixelDescriptor::RGBF32_LINEAR;
    let mut target = 255.0;
    let mut bg = 0.5;
    for line in include_str!("native_hdr_cpp.tsv")
        .lines()
        .chain(include_str!("native_hdr_gray_cpp.tsv").lines())
        .chain(include_str!("native_hdr_signed_cpp.tsv").lines())
    {
        if line.starts_with('#') {
            continue;
        }
        if let Some(case) = line
            .strip_prefix("case ")
            .or_else(|| line.strip_prefix("case-gray "))
        {
            let fields: Vec<_> = case.split_whitespace().collect();
            descriptor = if line.starts_with("case-gray") {
                PixelDescriptor::GRAYAF32_LINEAR
            } else {
                PixelDescriptor::RGBAF32_LINEAR
            }
            .with_transfer(match fields[0] {
                "8" => TransferFunction::Linear,
                "16" => TransferFunction::Pq,
                "18" => TransferFunction::Hlg,
                _ => panic!("transfer"),
            })
            .with_primaries(match fields[1] {
                "1" => zenpixels::ColorPrimaries::Bt709,
                "9" => zenpixels::ColorPrimaries::Bt2020,
                "11" => zenpixels::ColorPrimaries::DisplayP3,
                _ => panic!("primaries"),
            })
            .with_alpha(if fields[4] == "1" {
                Some(AlphaMode::Straight)
            } else {
                None
            });
            target = fields[2].parse::<f32>().unwrap();
            bg = fields[3].parse::<f32>().unwrap();
            continue;
        }
        assert_eq!(line.len(), 80);
        let values: Vec<f32> = (0..10)
            .map(|i| f32::from_bits(u32::from_str_radix(&line[i * 8..i * 8 + 8], 16).unwrap()))
            .collect();
        let gray = [values[0], values[3]];
        let samples = if descriptor.format == PixelFormat::GrayAF32 {
            &gray[..]
        } else {
            &values[..4]
        };
        let image = PixelSlice::new(
            bytemuck::cast_slice(samples),
            1,
            1,
            descriptor.format.bytes_per_pixel(),
            descriptor,
        )
        .unwrap()
        .with_color_context(Arc::new(
            zenpixels::ColorContext::default()
                .with_diffuse_white(zenpixels::DiffuseWhite::new(target)),
        ));
        let prepared = funnel(&image).unwrap();
        let linear = crate::linearize_prepared(&prepared, 1, bg);
        for c in 0..3 {
            assert_eq!(
                linear[c][0].to_bits(),
                values[4 + c].to_bits(),
                "{descriptor:?}, target={target}, bg={bg}, pixel={:?}, channel={c}",
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
                "scalar XYB {descriptor:?}, target={target}, channel={c}"
            );
            for &value in &simd[c] {
                assert_eq!(
                    value.to_bits(),
                    values[7 + c].to_bits(),
                    "SIMD XYB {descriptor:?}, target={target}, channel={c}"
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
        sync::Arc,
    };
    let oracle =
        std::env::var_os("SSIM2_HDR_ORACLE").expect("SSIM2_HDR_ORACLE must name the C++ oracle");
    let mut checked = 0usize;
    for (tf, transfer, targets) in [
        (8, TransferFunction::Linear, [255.0, 203.0]),
        (16, TransferFunction::Pq, [10000.0, 255.0]),
        (18, TransferFunction::Hlg, [1000.0, 300.0]),
    ] {
        for (p, primaries) in [
            (1, zenpixels::ColorPrimaries::Bt709),
            (9, zenpixels::ColorPrimaries::Bt2020),
            (11, zenpixels::ColorPrimaries::DisplayP3),
        ] {
            for target in targets {
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
                        let mut request = format!(
                            "frontend\n{tf} {p} {target} {bg} {} 65536 1\n",
                            u8::from(alpha)
                        );
                        for px in data.as_chunks::<4>().0 {
                            let bits = px.map(|v| (v as f32 * (1.0 / 65535.0)).to_bits());
                            writeln!(
                                request,
                                "{:08x} {:08x} {:08x} {:08x}",
                                bits[0], bits[1], bits[2], bits[3]
                            )
                            .unwrap();
                        }
                        // Linear u16 is intentionally unsupported. Test the same decoded
                        // f32 samples for linear transfer; PQ/HLG exercise the u16 reader.
                        let floats: Vec<f32> =
                            data.iter().map(|&v| v as f32 * (1.0 / 65535.0)).collect();
                        let desc = if transfer == TransferFunction::Linear {
                            PixelDescriptor::RGBAF32_LINEAR
                        } else {
                            PixelDescriptor::RGBA16_SRGB
                        };
                        let desc = desc
                            .with_transfer(transfer)
                            .with_primaries(primaries)
                            .with_alpha(alpha.then_some(AlphaMode::Straight));
                        let bytes = if transfer == TransferFunction::Linear {
                            bytemuck::cast_slice(&floats)
                        } else {
                            bytemuck::cast_slice(&data)
                        };
                        let image = PixelSlice::new(
                            bytes,
                            65536,
                            1,
                            65536 * desc.format.bytes_per_pixel(),
                            desc,
                        )
                        .unwrap()
                        .with_color_context(Arc::new(
                            zenpixels::ColorContext::default()
                                .with_diffuse_white(zenpixels::DiffuseWhite::new(target)),
                        ));
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
                                    "linear: {desc:?} target={target} bg={bg} code={i} channel={c}"
                                );
                                assert_eq!(
                                    xyb[c][i].to_bits(),
                                    expected[c + 3],
                                    "XYB: {desc:?} target={target} bg={bg} code={i} channel={c}"
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
