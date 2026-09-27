use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic};
use std::env;

fn load_rgb(path: &str) -> (Rgb, fast_ssim2::PixelBuffer) {
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
    let rgb = Rgb::new(
        pixels,
        w,
        h,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();
    let buffer = fast_ssim2::PixelBuffer::from_vec(
        img.into_raw(),
        w as u32,
        h as u32,
        fast_ssim2::PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();
    (rgb, buffer)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: compare-ssim <source> <distorted>");
        std::process::exit(1);
    }

    let (src, src_buf) = load_rgb(&args[1]);
    let (dst, dst_buf) = load_rgb(&args[2]);

    // rust-av ssimulacra2 v0.5.1 (uses its own scalar code path)
    let rustav = ssimulacra2::compute_frame_ssimulacra2(src.clone(), dst.clone()).unwrap();

    // fast-ssim2 default (SIMD kernels)
    let fast_simd =
        fast_ssim2::compute_ssimulacra2(&src_buf.as_slice(), &dst_buf.as_slice()).unwrap();

    // fast-ssim2 scalar-kernel path (bit-identical output)
    let fast_scalar = fast_ssim2::compute_ssimulacra2_with_config(
        &src_buf.as_slice(),
        &dst_buf.as_slice(),
        &fast_ssim2::Ssimulacra2Config::scalar(),
    )
    .unwrap();

    let d_simd = fast_simd - rustav;
    let d_scalar = fast_scalar - rustav;

    println!("rust-av v0.5.1:     {rustav:.10}");
    println!("fast-ssim2 (SIMD):  {fast_simd:.10}  Δ={d_simd:+.10}");
    println!("fast-ssim2 (scalar):{fast_scalar:.10}  Δ={d_scalar:+.10}");
    println!("SIMD vs scalar:     {:+.10}", fast_simd - fast_scalar);
}
