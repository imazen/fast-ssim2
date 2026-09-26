//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! `benches.rs` measures absolute throughput at several sizes, which tells you
//! how fast the metric is but not whether the SIMD paths are earning their
//! keep. A kernel slower than its own scalar fallback is invisible there. This
//! bench runs the identical pipeline with the native SIMD token disabled — the
//! comparison that can actually expose a bad kernel. (The equivalent gap in
//! linear-srgb was hiding a real regression.)
//!
//! Run: `cargo bench --bench tier_isolation`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use fast_ssim2::compute_ssimulacra2;
use num_traits::clamp;
use rand::RngExt;
use std::hint::black_box;

/// Owned encoded-sRGB `f32` pixels (k/255 grid stays LUT-exact).
fn srgb_f32_owned(data: Vec<[f32; 3]>, w: usize, h: usize) -> zenpixels::PixelBuffer {
    zenpixels::PixelBuffer::from_vec(
        bytemuck::cast_slice::<f32, u8>(data.as_flattened()).to_vec(),
        w as u32,
        h as u32,
        zenpixels::PixelDescriptor::RGBF32.with_transfer(zenpixels::TransferFunction::Srgb),
    )
    .unwrap()
}
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

fn make_rgb_pair(width: usize, height: usize) -> (zenpixels::PixelBuffer, zenpixels::PixelBuffer) {
    let mut rng = rand::rng();
    let source_data: Vec<[f32; 3]> = (0..width * height)
        .map(|_| {
            [
                rng.random_range(0.0f32..=1.0),
                rng.random_range(0.0f32..=1.0),
                rng.random_range(0.0f32..=1.0),
            ]
        })
        .collect();
    let distorted_data: Vec<[f32; 3]> = source_data
        .iter()
        .map(|&[r, g, b]| {
            [
                clamp(r + rng.random_range(-0.05f32..=0.05), 0.0, 1.0),
                clamp(g + rng.random_range(-0.05f32..=0.05), 0.0, 1.0),
                clamp(b + rng.random_range(-0.05f32..=0.05), 0.0, 1.0),
            ]
        })
        .collect();
    let source = srgb_f32_owned(source_data, width, height);
    let distorted = srgb_f32_owned(distorted_data, width, height);
    (source, distorted)
}

/// SSIMULACRA2 builds a multi-scale pyramid, so the ratio of blur/XYB work to
/// per-scale overhead shifts with size. A single size cannot tell you whether
/// the SIMD kernels hold up across the range.
const SIZES: &[(&str, usize, usize)] = &[
    ("320x240", 320, 240),
    ("1920x1080", 1920, 1080),
    ("3840x2160", 3840, 2160),
];

fn bench_tiers(c: &mut Criterion) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, ensure \
             archmage/testable_dispatch). Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    for &(label, w, h) in SIZES {
        let (source, distorted) = make_rgb_pair(w, h);
        let mut group = c.benchmark_group(format!("ssimulacra2/{label}"));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            group.bench_function(arm, |b| {
                set_simd(simd);
                b.iter(|| {
                    compute_ssimulacra2(
                        black_box(&source.as_slice()),
                        black_box(&distorted.as_slice()),
                    )
                    .unwrap()
                })
            });
        }
        set_simd(true);
        group.finish();
    }
    set_simd(true);
}

criterion_group!(benches, bench_tiers);
criterion_main!(benches);
