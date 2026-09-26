//! PU21-integrated HDR scoring (`hdr-pu` feature).
#![cfg(feature = "hdr-pu")]

use fast_ssim2::{PixelDescriptor, PixelSlice, compute_ssimulacra2_pu};

fn nits_slice(data: &[f32], w: usize, h: usize) -> PixelSlice<'_> {
    PixelSlice::new(
        bytemuck::cast_slice(data),
        w as u32,
        h as u32,
        w * 12,
        PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap()
}

#[test]
fn identical_nits_scores_100() {
    let data: Vec<f32> = (0..64 * 64 * 3)
        .map(|i| ((i * 37) % 977) as f32 + 0.05)
        .collect();
    let s = nits_slice(&data, 64, 64);
    let score = compute_ssimulacra2_pu(&s, &s).unwrap();
    assert!(
        (score - 100.0).abs() < 0.01,
        "identical nits should score ~100, got {score}"
    );
}

#[test]
fn darker_distortion_scores_lower() {
    let src: Vec<f32> = (0..64 * 64 * 3).map(|i| ((i * 37) % 977) as f32 + 5.0).collect();
    // 3× dimmer distorted copy — a luminance-heavy corruption.
    let mut dst = src.clone();
    for v in dst.iter_mut().step_by(4) {
        *v *= 0.3;
    }
    let (s, d) = (nits_slice(&src, 64, 64), nits_slice(&dst, 64, 64));
    let score = compute_ssimulacra2_pu(&s, &d).unwrap();
    assert!(score.is_finite() && score < 90.0, "dimmed copy: {score}");
}

#[test]
fn pq_encoded_input_works() {
    // PQ u16 buffer: encode a mid-nits ramp, feed as PQ — decodes to nits internally.
    let nits = 200.0f32;
    let m1 = 0.159_301_75_f32;
    let m2 = 78.843_75_f32;
    let c1 = 0.835_937_5_f32;
    let c2 = 18.851_562_f32;
    let c3 = 18.687_5_f32;
    // PQ inverse EOTF: nits → code
    let y = (nits / 10000.0).powf(m1);
    let code = ((c1 + c2 * y) / (1.0 + c3 * y)).powf(m2);
    let cv = (code.clamp(0.0, 1.0) * 65535.0).round() as u16;
    let data: Vec<u16> = vec![cv; 64 * 64 * 3];
    let slice = PixelSlice::new(
        bytemuck::cast_slice(&data),
        64,
        64,
        64 * 6,
        PixelDescriptor::RGB16_BT2100_PQ,
    )
    .unwrap();
    let score = compute_ssimulacra2_pu(&slice, &slice).unwrap();
    assert!(
        (score - 100.0).abs() < 0.01,
        "identical PQ input should score ~100, got {score}"
    );
}

#[test]
fn srgb_descriptor_rejected_in_pu_mode() {
    let data = vec![128u8; 64 * 64 * 3];
    let s = PixelSlice::new(&data, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    assert!(compute_ssimulacra2_pu(&s, &s).is_err());
}

#[test]
fn linear_nits_with_alpha_composites() {
    // RGBAF32_LINEAR + alpha: transparent corners fade to the dark/light
    // backgrounds — a differing opaque-vs-transparent pair should drop.
    let mut data = vec![100.0f32; 64 * 64 * 4];
    for i in 0..64 * 64 {
        data[i * 4 + 3] = if i % 7 == 0 { 0.0 } else { 1.0 };
    }
    let pa = PixelSlice::new(
        bytemuck::cast_slice(&data), 64, 64, 64 * 16,
        PixelDescriptor::RGBAF32_LINEAR,
    ).unwrap();
    let opaque: Vec<f32> = vec![100.0; 64 * 64 * 3];
    let po = nits_slice(&opaque, 64, 64);
    let score = compute_ssimulacra2_pu(&pa, &po).unwrap();
    assert!(score.is_finite() && score < 100.0, "alpha-aware: {score}");
}

#[test]
fn bt2020_primaries_accepted_in_pu() {
    // Linear-f32 with BT.2020 primaries — no gamut conversion, opsin
    // consumes primaries as-is (zensim convention).
    let data = vec![150.0f32; 64 * 64 * 3];
    let s = PixelSlice::new(
        bytemuck::cast_slice(&data), 64, 64, 64 * 12,
        PixelDescriptor::RGBF32_LINEAR.with_primaries(zenpixels::ColorPrimaries::Bt2020),
    )
    .unwrap();
    let score = compute_ssimulacra2_pu(&s, &s).unwrap();
    assert!((score - 100.0).abs() < 0.01, "identical BT.2020: {score}");
}
