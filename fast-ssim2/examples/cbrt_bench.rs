//! cbrt flavor timing: isolated XYB stage + e2e planar, min/median of N
//! interleaved reps (min is contention-robust on a busy box).

use fast_ssim2::pipeline::{Kernel, Opts, XybFlavor, simd};
use std::time::Instant;

const N: usize = 30;
const W: usize = 512;
const H: usize = 512;

fn planes(seed: u32) -> [Vec<f32>; 3] {
    let n = W * H;
    [
        (0..n).map(|i| ((((i as u32 * 2654435761) ^ seed) & 0xffff) as f32) / 65536.0).collect(),
        (0..n).map(|i| ((((i as u32 * 2246822519) ^ seed) & 0xffff) as f32) / 65536.0).collect(),
        (0..n).map(|i| ((((i as u32 * 3266489917) ^ seed) & 0xffff) as f32) / 65536.0).collect(),
    ]
}

fn stats(v: &mut [std::time::Duration]) -> (f64, f64) {
    v.sort();
    (v[0].as_secs_f64() * 1e3, v[v.len() / 2].as_secs_f64() * 1e3)
}

fn main() {
    // ── isolated XYB stage ──
    let src = planes(7);
    let mut t_ref = Vec::new();
    let mut t_mid = Vec::new();
    let mut t_lo = Vec::new();
    for _ in 0..N {
        let mut p = src.clone();
        let t = Instant::now();
        simd::planes_to_positive_xyb_simd(&mut p);
        t_ref.push(t.elapsed());
        std::hint::black_box(&p);
        let mut p = src.clone();
        let t = Instant::now();
        simd::planes_to_positive_xyb_hi_simd(&mut p);
        t_mid.push(t.elapsed());
        std::hint::black_box(&p);
        let mut p = src.clone();
        let t = Instant::now();
        simd::planes_to_positive_xyb_lo_simd(&mut p);
        t_lo.push(t.elapsed());
        std::hint::black_box(&p);
    }
    let (r0, r5) = stats(&mut t_ref);
    let (m0, m5) = stats(&mut t_mid);
    let (l0, l5) = stats(&mut t_lo);
    println!("isolated XYB (512² planes):");
    println!("  ref   min {r0:.3}ms  med {r5:.3}ms");
    println!("  midp  min {m0:.3}ms  med {m5:.3}ms   ({:+.1}%)", (m0 / r0 - 1.0) * 100.0);
    println!("  lowp  min {l0:.3}ms  med {l5:.3}ms   ({:+.1}%)", (l0 / r0 - 1.0) * 100.0);

    // ── e2e planar ──
    let a = planes(0xaaa);
    let b = planes(0xbbb);
    let mut e_ref = Vec::new();
    let mut e_mid = Vec::new();
    let mut e_lo = Vec::new();
    let opts = |f| Opts { kernel: Kernel::Simd, flavor: f };
    for _ in 0..N {
        let t = Instant::now();
        std::hint::black_box(fast_ssim2::pipeline::compute_planar_stop(
            a.clone(), b.clone(), W, H, opts(XybFlavor::CubeRoot), &enough::Unstoppable).unwrap());
        e_ref.push(t.elapsed());
        let t = Instant::now();
        std::hint::black_box(fast_ssim2::pipeline::compute_planar_stop(
            a.clone(), b.clone(), W, H, opts(XybFlavor::CubeRootHi), &enough::Unstoppable).unwrap());
        e_mid.push(t.elapsed());
        let t = Instant::now();
        std::hint::black_box(fast_ssim2::pipeline::compute_planar_stop(
            a.clone(), b.clone(), W, H, opts(XybFlavor::CubeRootLo), &enough::Unstoppable).unwrap());
        e_lo.push(t.elapsed());
    }
    let (r0, r5) = stats(&mut e_ref);
    let (m0, m5) = stats(&mut e_mid);
    let (l0, l5) = stats(&mut e_lo);
    println!("e2e planar (512², 6 scales, SIMD):");
    println!("  ref   min {r0:.3}ms  med {r5:.3}ms");
    println!("  midp  min {m0:.3}ms  med {m5:.3}ms   ({:+.1}%)", (m0 / r0 - 1.0) * 100.0);
    println!("  lowp  min {l0:.3}ms  med {l5:.3}ms   ({:+.1}%)", (l0 / r0 - 1.0) * 100.0);
}
