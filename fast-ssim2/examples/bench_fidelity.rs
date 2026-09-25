use fast_ssim2::ToLinearRgb;
use imgref::ImgVec;
use std::time::Instant;

fn load_png(path: &str) -> ImgVec<[u8; 3]> {
    let img = image::open(path).unwrap().into_rgb8();
    let (w, h) = img.dimensions();
    ImgVec::new(img.pixels().map(|p|[p[0],p[1],p[2]]).collect(), w as usize, h as usize)
}
fn main() {
    let a = load_png(&std::env::args().nth(1).unwrap());
    let b = load_png(&std::env::args().nth(2).unwrap_or_else(|| std::env::args().nth(1).unwrap()));
    let n: u32 = std::env::args().nth(3).map(|s|s.parse().unwrap()).unwrap_or(20);
    for (name, fid) in [("precise", fast_ssim2::Fidelity::Precise), ("official", fast_ssim2::Fidelity::MatchOfficial)] {
        let cfg = fast_ssim2::Ssimulacra2Config::default().with_fidelity(fid);
        // warm
        let s = fast_ssim2::compute_ssimulacra2_with_config(a.as_ref(), b.as_ref(), cfg.clone()).unwrap();
        let t = Instant::now();
        for _ in 0..n { std::hint::black_box(fast_ssim2::compute_ssimulacra2_with_config(a.as_ref(), b.as_ref(), cfg.clone()).unwrap()); }
        let dt = t.elapsed().as_secs_f64() / n as f64;
        println!("{name}: score={s} {dt:.4} ms/img");
    }
}
