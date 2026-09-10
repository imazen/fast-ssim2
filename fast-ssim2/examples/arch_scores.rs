//! Cross-architecture score consistency: print `compute_ssimulacra2` for a
//! deterministic, self-contained set of image pairs, at full precision.
//!
//! Run it on two machines at the same commit and diff the output. Every score
//! should match bit-for-bit on any two targets whose `magetypes` `mul_add` is a
//! real fused multiply-add (NEON, AVX2, AVX-512); the known exception is the
//! blur, which stays fused deliberately, so a target without hardware FMA
//! (wasm128, the scalar polyfill) lands in a different equivalence class — see
//! `CLAUDE.md` and `benchmarks/cpp_parity_2026-08-31.md` §4b.
//!
//! Needs no corpus and no C++ binary: the sources are synthesised here, so the
//! two hosts are guaranteed to be scoring identical inputs.
//!
//!   cargo run --release --example arch_scores > scores-$(uname -m).tsv
//!   diff scores-aarch64.tsv scores-x86_64.tsv
//!
//! The content is deliberately *structured* — gradients, rings, edges, texture —
//! rather than flat patches: SSIMULACRA2 divides a near-zero blur residual by
//! `kC2 = 9e-4`, so flat fields amplify rounding by ~1e6 and would report
//! disagreement that says nothing about the arch (same reason the compiled-in
//! reference table's `uniform_shift` family is not a usable gate).

use fast_ssim2::{LinearRgbImage, compute_ssimulacra2, srgb_u8_to_linear};

struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) & 0xFF) as u8
    }
}

/// Structured 8-bit source: a diagonal gradient, concentric rings, a hard edge
/// and a textured quadrant, so every pyramid level has real content.
fn source(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let ring = {
                let dx = fx - 0.5;
                let dy = fy - 0.5;
                let r = (dx * dx + dy * dy).sqrt();
                (r * 40.0).sin() * 0.5 + 0.5
            };
            let edge = if fx > 0.6 && fy < 0.4 { 0.85 } else { 0.15 };
            let texture = if fx < 0.35 && fy > 0.65 {
                f32::from(lcg.next_u8()) / 255.0
            } else {
                0.0
            };
            let r = (fx * 0.5 + ring * 0.5).clamp(0.0, 1.0);
            let g = (fy * 0.4 + edge * 0.6).clamp(0.0, 1.0);
            let b = (ring * 0.3 + texture * 0.7).clamp(0.0, 1.0);
            let i = (y * w + x) * 3;
            px[i] = (r * 255.0).round() as u8;
            px[i + 1] = (g * 255.0).round() as u8;
            px[i + 2] = (b * 255.0).round() as u8;
        }
    }
    px
}

fn to_linear(px: &[u8], w: usize, h: usize) -> LinearRgbImage {
    let data: Vec<[f32; 3]> = px
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            [
                srgb_u8_to_linear(c[0]),
                srgb_u8_to_linear(c[1]),
                srgb_u8_to_linear(c[2]),
            ]
        })
        .collect();
    LinearRgbImage::new(data, w, h)
}

fn noise(px: &[u8], amp: i32, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    px.iter()
        .map(|&v| {
            let n = (i32::from(lcg.next_u8()) % (2 * amp + 1)) - amp;
            (i32::from(v) + n).clamp(0, 255) as u8
        })
        .collect()
}

fn box_blur(px: &[u8], w: usize, h: usize, r: i32) -> Vec<u8> {
    let mut out = vec![0u8; px.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            for c in 0..3 {
                let mut sum = 0u32;
                let mut n = 0u32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = (x + dx).clamp(0, w as i32 - 1) as usize;
                        let sy = (y + dy).clamp(0, h as i32 - 1) as usize;
                        sum += u32::from(px[(sy * w + sx) * 3 + c]);
                        n += 1;
                    }
                }
                out[((y as usize) * w + x as usize) * 3 + c] = (sum / n) as u8;
            }
        }
    }
    out
}

/// Quantise each channel to `levels` steps — a banding-style distortion that
/// keeps structure while changing every pixel a little.
fn quantise(px: &[u8], levels: u32) -> Vec<u8> {
    let step = 255.0 / (levels - 1) as f32;
    px.iter()
        .map(|&v| ((f32::from(v) / step).round() * step).clamp(0.0, 255.0) as u8)
        .collect()
}

fn main() {
    println!("# fast-ssim2 arch score consistency");
    println!("# target_arch = {}", std::env::consts::ARCH);
    println!("size\tdistortion\tscore");

    for &(w, h) in &[(64usize, 64usize), (177, 129), (256, 256), (640, 480)] {
        let src = source(w, h, 0x0bad_c0de_dead_beef);
        let a = to_linear(&src, w, h);
        for (name, dist) in [
            ("self", src.clone()),
            ("noise_a2", noise(&src, 2, 0x1234_5678)),
            ("noise_a12", noise(&src, 12, 0x8765_4321)),
            ("boxblur_r1", box_blur(&src, w, h, 1)),
            ("boxblur_r3", box_blur(&src, w, h, 3)),
            ("quantise_8", quantise(&src, 8)),
            ("quantise_32", quantise(&src, 32)),
        ] {
            let b = to_linear(&dist, w, h);
            let score = compute_ssimulacra2(a.clone(), b).expect("score");
            // Full f64 precision: the point is bit-for-bit comparison.
            println!("{w}x{h}\t{name}\t{score:.17e}");
        }
    }
}
