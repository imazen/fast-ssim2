//! Flat-field lattice probe: how much does the score sawtooth across u8
//! quantization levels, and does a more accurate cube root smooth it?
//!
//! For each gray level `v`: score flat(v) vs flat(v) with ~1% of pixels
//! bumped +1. Plots `CubeRoot` (hwy reference recipe) vs `CubeRootHi`
//! (magetypes `cbrt_midp`, ~3 ulp) and reports step-roughness stats.
//!
//! Finding (2026-09): both curves are equally sawtoothy — the wobble is
//! structural (u8 LUT step spacing + `-ln` pooling amplification near
//! score≈99), not cbrt ulp error. Real-image divergence is ~0.005.

use fast_ssim2::pipeline::{
    EncodedData, EncodedSrgb, Kernel, Opts, XybFlavor, compute_planar_stop, linearize,
};

fn flat_u8(v: u8, w: usize, h: usize) -> EncodedSrgb {
    EncodedSrgb {
        width: w,
        height: h,
        data: EncodedData::U8(vec![v; w * h * 3]),
        alpha: None,
    }
}

fn perturbed(v: u8, w: usize, h: usize, delta: i8, frac: f32) -> EncodedSrgb {
    let mut d = vec![v; w * h * 3];
    let n = (w * h) as f32 * frac;
    for i in 0..(n as usize) {
        let idx = (i * 7919) % (w * h);
        let nv = (v as i16 + delta as i16).clamp(0, 255) as u8;
        for c in 0..3 {
            d[idx * 3 + c] = nv;
        }
    }
    EncodedSrgb {
        width: w,
        height: h,
        data: EncodedData::U8(d),
        alpha: None,
    }
}

fn score(src: &EncodedSrgb, dst: &EncodedSrgb, opts: Opts) -> f64 {
    let (w, h) = (src.width, src.height);
    let lin1 = linearize(src, 0.5);
    let lin2 = linearize(dst, 0.5);
    compute_planar_stop(lin1, lin2, w, h, opts, &enough::Unstoppable).unwrap()
}

fn main() {
    if std::env::args().any(|a| a == "--cancel-ab") {
        cancel_ab();
        return;
    }

    let (w, h) = (64, 64);
    let opts_ref = Opts { kernel: Kernel::Simd, flavor: XybFlavor::CubeRoot };
    let opts_hi = Opts { kernel: Kernel::Simd, flavor: XybFlavor::CubeRootHi };

    // Real-image divergence check.
    let load = |p: &str| {
        let img = image::ImageReader::open(p).unwrap().decode().unwrap().to_rgb8();
        let (iw, ih) = img.dimensions();
        EncodedSrgb {
            width: iw as usize,
            height: ih as usize,
            data: EncodedData::U8(img.into_raw()),
            alpha: None,
        }
    };
    let td = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/");
    let (a, b) = (
        load(&format!("{td}tank_source.png")),
        load(&format!("{td}tank_distorted.png")),
    );
    let sref = score(&a, &b, opts_ref);
    let shi = score(&a, &b, opts_hi);
    println!("real pair: ref {sref:.8}  hi {shi:.8}  diff {:.3e}", shi - sref);

    // Flat-field sweep.
    let mut ref_scores = Vec::new();
    let mut hi_scores = Vec::new();
    println!("v\tCubeRoot\t\tCubeRootHi\t\tdiff");
    for v in (16..=240).step_by(4) {
        let src = flat_u8(v, w, h);
        let dst = perturbed(v, w, h, 1, 0.01);
        let s1 = score(&src, &dst, opts_ref);
        let s2 = score(&src, &dst, opts_hi);
        ref_scores.push(s1);
        hi_scores.push(s2);
        println!("{v}\t{s1:.8}\t{s2:.8}\t{:.2e}", s2 - s1);
    }
    let stats = |xs: &[f64]| {
        let steps: Vec<f64> = xs
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        let flips = xs
            .windows(3)
            .filter(|w| (w[1] - w[0]) * (w[2] - w[1]) < 0.0)
            .count();
        (
            steps.iter().sum::<f64>() / steps.len() as f64,
            steps.iter().cloned().fold(0f64, f64::max),
            flips,
        )
    };
    let (m1, x1, f1) = stats(&ref_scores);
    let (m2, x2, f2) = stats(&hi_scores);
    println!(
        "roughness ref: mean|step| {m1:.4} max {x1:.4} flips {f1}/{}",
        ref_scores.len() - 2
    );
    println!(
        "roughness hi : mean|step| {m2:.4} max {x2:.4} flips {f2}/{}",
        hi_scores.len() - 2
    );
}

// ── addendum: divisor-cancellation A/B ─────────────────────────────────
// Reproduce the scale-0 map stage on a flat input and compare the f32 vs
// f64 σ/μ quotient paths (sigma_f64 toggle). If the wobble lived in the
// s12−μ1μ2 / s11−μ1² subtractions, f64 eval would visibly differ.
#[allow(dead_code)]
fn cancel_ab() {
    use fast_ssim2::pipeline::{simd, maps};
    let (w, h) = (64, 64);
    let rg = fast_ssim2::pipeline::gauss::create_recursive_gaussian(1.5);
    println!("v\td_f32\td_f64\tf32-f64");
    for v in (16..=240).step_by(16) {
        let src = flat_u8(v, w, h);
        let dst = perturbed(v, w, h, 1, 0.01);
        let mut x1 = linearize(&src, 0.5);
        let mut x2 = linearize(&dst, 0.5);
        let n = w * h;
        fast_ssim2::pipeline::planes_to_positive_xyb(&mut x1, n);
        fast_ssim2::pipeline::planes_to_positive_xyb(&mut x2, n);
        // mul planes → blur
        let mul = |a: &[Vec<f32>; 3], b: &[Vec<f32>; 3]| {
            let mut o = [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]];
            fast_ssim2::pipeline::gauss::multiply_planes(a, b, &mut o);
            o
        };
        let s1 = simd::blur_planes_simd(&rg, &mul(&x1, &x1), w, h);
        let s2 = simd::blur_planes_simd(&rg, &mul(&x2, &x2), w, h);
        let s12 = simd::blur_planes_simd(&rg, &mul(&x1, &x2), w, h);
        let m1 = simd::blur_planes_simd(&rg, &x1, w, h);
        let m2 = simd::blur_planes_simd(&rg, &x2, w, h);
        let a32 = maps::ssim_map_opts(&m1, &m2, &s1, &s2, &s12, w, h, false);
        let a64 = maps::ssim_map_opts(&m1, &m2, &s1, &s2, &s12, w, h, true);
        println!("{v}\t{:.8}\t{:.8}\t{:.3e}", a32[0], a64[0], a64[0] - a32[0]);
    }
}
