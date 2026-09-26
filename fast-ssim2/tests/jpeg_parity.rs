#![allow(clippy::chunks_exact_to_as_chunks)]
//! JPEG-input parity for `MatchOfficial`.
//!
//! The reference `ssimulacra2` binary decodes JPEGs with libjpeg(-turbo).
//! Our library takes already-decoded pixels, so conformance on JPEG
//! inputs requires a libjpeg-exact decoder — zenjpeg with its default
//! `IdctMethod::Libjpeg` is byte-for-byte identical to turbo/djpeg
//! (verified upstream by `__ffi-tests`). This test pins the whole chain:
//! zenjpeg encode → zenjpeg decode → `EncodedSrgb` → official score.
//!
//! Expected values were captured against the same chain; a zenjpeg
//! decode-output change will show up here as a score change.

#![cfg(not(target_arch = "wasm32"))]

use enough::Unstoppable;
use fast_ssim2::pipeline::{self, EncodedData, EncodedSrgb};
use zenjpeg::decoder::{Decoder, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn gen_px(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 3];
    let mut s = seed;
    for (i, b) in px.iter_mut().enumerate() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let x = (i / 3) % w;
        let y = (i / 3) / w;
        *b = (((x / 16 + y / 16) * 23) as u8).wrapping_add((s >> 27) as u8);
    }
    px
}

fn encode(px: &[u8], w: u32, h: u32, q: f32) -> Vec<u8> {
    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_bytes(px, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode")
}

fn decode_zenjpeg(j: &[u8]) -> EncodedSrgb {
    let r = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(j, Unstoppable)
        .expect("zenjpeg decode");
    let px = r.pixels_u8().expect("u8 pixels");
    EncodedSrgb {
        width: r.width() as usize,
        height: r.height() as usize,
        data: EncodedData::U8(px.to_vec()),
        alpha: None,
    }
}

/// The official score through the zenjpeg libjpeg-exact decode chain.
/// Bit-exact expectation: both the IDCT and the metric are deterministic.
#[test]
fn jpeg_chain_official_score_pinned() {
    let (w, h) = (320u32, 240u32);
    let src = gen_px(w as usize, h as usize, 0x1234);
    let mut dst = src.clone();
    for b in dst.iter_mut().step_by(97) {
        *b = b.wrapping_add(9);
    }
    let e1 = decode_zenjpeg(&encode(&src, w, h, 90.0));
    let e2 = decode_zenjpeg(&encode(&dst, w, h, 90.0));
    let s = pipeline::compute_encoded(&e1, &e2, pipeline::Kernel::Simd).unwrap();
    assert!(
        (s - 74.1141044413).abs() < 1e-9,
        "official score via zenjpeg decode chain = {s:.10}, expected 74.1141044413"
    );
}
