use std::env;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn load_rgb(path: &str) -> Rgb {
    let img = image::open(path).unwrap().into_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<[f32; 3]> = img
        .pixels()
        .map(|p| {
            [
                p.0[0] as f32 / 255.0,
                p.0[1] as f32 / 255.0,
                p.0[2] as f32 / 255.0,
            ]
        })
        .collect();
    Rgb::new(
        pixels,
        w,
        h,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: compare-ssim <source> <distorted>");
        std::process::exit(1);
    }

    let src = load_rgb(&args[1]);
    let dst = load_rgb(&args[2]);

    // rust-av ssimulacra2 v0.5.1 (uses its own scalar code path)
    let rustav = ssimulacra2::compute_frame_ssimulacra2(src.clone(), dst.clone()).unwrap();

    // fast-ssim2 default (SIMD)
    let fast_simd = fast_ssim2::compute_frame_ssimulacra2(src.clone(), dst.clone()).unwrap();

    // fast-ssim2 scalar path
    let fast_scalar = fast_ssim2::compute_frame_ssimulacra2_with_config(
        src,
        dst,
        fast_ssim2::Ssimulacra2Config::scalar(),
    )
    .unwrap();

    let d_simd = fast_simd - rustav;
    let d_scalar = fast_scalar - rustav;

    println!("rust-av v0.5.1:     {rustav:.10}");
    println!("fast-ssim2 (SIMD):  {fast_simd:.10}  Δ={d_simd:+.10}");
    println!("fast-ssim2 (scalar):{fast_scalar:.10}  Δ={d_scalar:+.10}");
    println!("SIMD vs scalar:     {:+.10}", fast_simd - fast_scalar);
}
