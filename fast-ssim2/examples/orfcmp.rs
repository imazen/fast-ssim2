use fast_ssim2::pipeline::precompute::ReferenceCache;
use fast_ssim2::pipeline::{self, EncodedData, EncodedSrgb};
use std::time::Instant;
fn load(p: &str) -> EncodedSrgb {
    let i = image::open(p).unwrap().into_rgb8();
    let (w, h) = i.dimensions();
    EncodedSrgb {
        width: w as usize,
        height: h as usize,
        data: EncodedData::U8(i.into_raw()),
        alpha: None,
    }
}
fn main() {
    let a = std::env::args().nth(1).unwrap();
    let b = std::env::args().nth(2).unwrap();
    let ea = load(&a);
    let eb = load(&b);
    let t = Instant::now();
    let r = ReferenceCache::new(&ea).unwrap();
    println!("new: {:.1}ms", t.elapsed().as_secs_f64() * 1e3);
    for _ in 0..2 {
        let t = Instant::now();
        let s = r.compare(&eb).unwrap();
        println!(
            "compare: {:.8} ({:.1}ms)",
            s,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t = Instant::now();
    let s2 = pipeline::compute_encoded(&ea, &eb, pipeline::Kernel::Scalar).unwrap();
    println!(
        "one-shot(scalar): {:.8} ({:.1}ms)",
        s2,
        t.elapsed().as_secs_f64() * 1e3
    );
    for par in [false, true] {
        let t = Instant::now();
        let s3 = r.compare_strip(&eb, 64, 96, par).unwrap();
        println!(
            "compare_strip(par={}): {:.8} ({:.1}ms)",
            par,
            s3,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}
