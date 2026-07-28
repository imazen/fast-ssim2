//! Per-kernel NEON-vs-scalar for fast-ssim2's SIMD ops.
//!
//! The crate was only ever measured end-to-end (1.7-1.8x on aarch64). An
//! end-to-end number cannot show a single kernel losing to its own scalar
//! fallback — faster kernels hide it. That failure mode turned up in garb,
//! zensim, zentone, zenpng and zenresize in the same sweep.
//!
//! Run: `cargo bench --bench kernel_tiers`
//! Do NOT pass `-C target-cpu=native` (the tier then cannot be disabled).

use fast_ssim2::__bench_kernels as k;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") { "neon" } else { "v3(avx2)" };

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(on: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!on).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_on: bool) -> bool { false }

fn planes(n: usize, seed: u32) -> [Vec<f32>; 3] {
    let mut s = seed | 1;
    let mut g = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as f32 / 16_777_216.0
    };
    [
        (0..n).map(|_| g()).collect(),
        (0..n).map(|_| g()).collect(),
        (0..n).map(|_| g()).collect(),
    ]
}

fn bench_kernels(c: &mut Criterion) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    let n = 1920 * 1080;
    let a = planes(n, 1);
    let b = planes(n, 7);

    let mut group = c.benchmark_group("image_multiply");
    for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
        group.bench_function(arm, |bch| {
            set_simd(simd);
            let mut out = planes(n, 3);
            bch.iter(|| k::image_multiply_simd(&a, &b, &mut out));
        });
    }
    set_simd(true);
    group.finish();

    // ssim_map + edge_diff_map at a cache-resident size, so the measurement is
    // compute-bound rather than pinned at the memory-bandwidth ceiling the way
    // image_multiply above is.
    let (w, h) = (512usize, 512usize);
    let sn = w * h;
    let (m1, m2) = (planes(sn, 11), planes(sn, 13));
    let (s11, s22, s12) = (planes(sn, 17), planes(sn, 19), planes(sn, 23));

    let mut group = c.benchmark_group("ssim_map/512x512");
    for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
        group.bench_function(arm, |bch| {
            set_simd(simd);
            bch.iter(|| k::ssim_map_simd(1, 0, w, h, &m1, &m2, &s11, &s22, &s12));
        });
    }
    set_simd(true);
    group.finish();

    let mut group = c.benchmark_group("edge_diff_map/512x512");
    for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
        group.bench_function(arm, |bch| {
            set_simd(simd);
            bch.iter(|| k::edge_diff_map_simd(1, 0, w, h, &m1, &m2, &s11, &s22));
        });
    }
    set_simd(true);
    group.finish();

    // XYB is the per-pixel color transform on the front of the pipeline.
    let mut xyb = vec![[0.0f32; 3]; sn];
    let mut st = 1u32;
    for p in xyb.iter_mut() {
        for c in p.iter_mut() {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *c = (st >> 8) as f32 / 16_777_216.0;
        }
    }
    let mut group = c.benchmark_group("linear_rgb_to_xyb/512x512");
    for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
        group.bench_function(arm, |bch| {
            set_simd(simd);
            let mut buf = xyb.clone();
            bch.iter(|| k::linear_rgb_to_xyb_simd(&mut buf));
        });
    }
    set_simd(true);
    group.finish();

    // The recursive-Gaussian blur: SSIMULACRA2's dominant stage. `Blur` is
    // public, so this needs no forwarder.
    for &(label, bw, bh) in &[("512x512", 512usize, 512usize), ("1920x1080", 1920, 1080)] {
        let plane = planes(bw * bh, 29);
        let mut group = c.benchmark_group(format!("blur/{label}"));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            group.bench_function(arm, |bch| {
                set_simd(simd);
                let mut blur = fast_ssim2::Blur::new(bw, bh);
                let mut out = planes(bw * bh, 31);
                bch.iter(|| blur.blur_into(&plane, &mut out));
            });
        }
        set_simd(true);
        group.finish();
    }

    // The crate's own hand-written scalar reference, for reference.
    let mut group = c.benchmark_group("image_multiply_handwritten_scalar");
    group.bench_function("handwritten", |bch| {
        let mut out = planes(n, 3);
        bch.iter(|| k::image_multiply_scalar(&a, &b, &mut out));
    });
    group.finish();
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
