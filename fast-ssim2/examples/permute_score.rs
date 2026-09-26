// Batch scorer over a manifest of image pairs — corpus-parity tool.
// Usage: permute_score <manifest.csv> [simd|scalar]
//   manifest lines: source.png,distorted.png[,tag]
// Output CSV: tag,score
use enough::Unstoppable;
use fast_ssim2::pipeline::{self, EncodedData, EncodedSrgb, Kernel};
use std::fmt::Write as _;
use zenjpeg::decoder::Decoder;
use zenjpeg::decoder::PixelFormat;

fn load_jpeg(path: &str) -> EncodedSrgb {
    // zenjpeg's default IdctMethod::Libjpeg is byte-exact vs
    // libjpeg-turbo/djpeg — matches the reference ssimulacra2 binary's
    // libjpeg decode, unlike image-rs's own IDCT.
    let data = std::fs::read(path).unwrap();
    let r = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&data, Unstoppable)
        .unwrap();
    let (w, h) = (r.width() as usize, r.height() as usize);
    let px: Vec<u8> = r.pixels_u8().unwrap().to_vec();
    EncodedSrgb {
        width: w,
        height: h,
        data: EncodedData::U8(px),
        alpha: None,
    }
}

fn load(path: &str) -> EncodedSrgb {
    if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        return load_jpeg(path);
    }
    let img = image::open(path).unwrap();
    if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
        let mut alpha = Vec::with_capacity(rgba.len() / 4);
        for px in rgba.as_raw().as_chunks::<4>().0 {
            rgb.extend_from_slice(&px[..3]);
            alpha.push(px[3] as f32 * (1.0 / 255.0));
        }
        return EncodedSrgb {
            width: w as usize,
            height: h as usize,
            data: EncodedData::U8(rgb),
            alpha: Some(alpha),
        };
    }
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    EncodedSrgb {
        width: w as usize,
        height: h as usize,
        data: EncodedData::U8(rgb.into_raw()),
        alpha: None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let manifest = &args[1];
    let kernel = match args.get(2).map(|s| s.as_str()) {
        Some("scalar") => Kernel::Scalar,
        _ => Kernel::Simd,
    };
    let mut out = String::new();
    for line in std::fs::read_to_string(manifest).unwrap().lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let f: Vec<&str> = line.split(',').collect();
        let e1 = load(f[0]);
        let e2 = load(f[1]);
        let s = pipeline::compute_encoded(&e1, &e2, kernel).unwrap_or(f64::NAN);
        let tag = if f.len() > 2 { f[2] } else { f[1] };
        let _ = writeln!(out, "{tag},{s}");
        eprintln!("{tag} done");
    }
    println!("{out}");
}
