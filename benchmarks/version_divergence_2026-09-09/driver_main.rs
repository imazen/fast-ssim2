//! Attribute the fast-ssim2 0.7.1 -> 0.8.2 score divergence, and rank every
//! candidate against the C++ SSIMULACRA2 binary.
//!
//! Seven scorers, all fed the *same* linear-RGB buffers so only the metric
//! differs:
//!   v071      crates.io 0.7.1        fused opsin matmul, f64-Newton cbrt
//!   v082      crates.io 0.8.2        fused opsin matmul, f32-Halley cbrt
//!   head      local HEAD (0.9.0)     unfused matmul,     f32-Halley cbrt
//!   repro     HEAD + fused mm + f64 cbrt          (does that reproduce 0.7.1?)
//!   cbrt64    HEAD + f64 cbrt                     (cbrt change alone)
//!   cppcbrt   HEAD + fused mm + jpegli cbrt       (maximum C++ fidelity)
//!   cppcbrtu  HEAD + jpegli cbrt                  (jpegli cbrt, unfused mm)

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use image::{ImageReader, RgbImage};

const N_IMPL: usize = 8;
const NAMES: [&str; N_IMPL] = [
    "v071", "v082", "head", "repro", "cbrt64", "cppcbrt", "cppcbrtu", "cppmax",
];

fn call_cpp(bin: &Path, s: &Path, d: &Path) -> f64 {
    let out = Command::new(bin).arg(s).arg(d).output().expect("exec");
    assert!(out.status.success(), "ssimulacra2 failed on {}", s.display());
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_whitespace().last()?.parse::<f64>().ok())
        .next()
        .expect("parse score")
}

struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) & 0xFF) as u8
    }
}

fn jpeg_roundtrip(img: &RgbImage, q: u8) -> RgbImage {
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut std::io::Cursor::new(&mut buf), q)
        .encode_image(img)
        .expect("encode");
    image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg)
        .expect("decode")
        .to_rgb8()
}

fn add_noise(img: &RgbImage, amp: i32, seed: u64) -> RgbImage {
    let mut lcg = Lcg(seed);
    let mut out = img.clone();
    for p in out.pixels_mut() {
        for c in 0..3 {
            let n = (lcg.next_u8() as i32 % (2 * amp + 1)) - amp;
            p.0[c] = (p.0[c] as i32 + n).clamp(0, 255) as u8;
        }
    }
    out
}

fn box_blur(img: &RgbImage, r: i32) -> RgbImage {
    let (w, h) = img.dimensions();
    let mut out = RgbImage::new(w, h);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let (mut s, mut n) = ([0u32; 3], 0u32);
            for dy in -r..=r {
                for dx in -r..=r {
                    let p = img
                        .get_pixel(
                            (x + dx).clamp(0, w as i32 - 1) as u32,
                            (y + dy).clamp(0, h as i32 - 1) as u32,
                        )
                        .0;
                    for c in 0..3 {
                        s[c] += p[c] as u32;
                    }
                    n += 1;
                }
            }
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([(s[0] / n) as u8, (s[1] / n) as u8, (s[2] / n) as u8]),
            );
        }
    }
    out
}

/// Push each channel toward mid-grey: makes the image locally flat, which is
/// where SSIMULACRA2's 1/kC2 amplification of tiny XYB differences bites.
fn flatten(img: &RgbImage, keep: f32) -> RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        for c in 0..3 {
            let v = p.0[c] as f32;
            p.0[c] = (128.0 + (v - 128.0) * keep).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn lin(img: &RgbImage) -> Vec<[f32; 3]> {
    img.pixels()
        .map(|p| {
            [
                head::srgb_u8_to_linear(p.0[0]),
                head::srgb_u8_to_linear(p.0[1]),
                head::srgb_u8_to_linear(p.0[2]),
            ]
        })
        .collect()
}

fn score_all(a: &[[f32; 3]], b: &[[f32; 3]], w: usize, h: usize) -> [f64; N_IMPL] {
    [
        v071::compute_ssimulacra2(
            v071::LinearRgbImage::new(a.to_vec(), w, h),
            v071::LinearRgbImage::new(b.to_vec(), w, h),
        )
        .unwrap(),
        v082::compute_ssimulacra2(
            v082::LinearRgbImage::new(a.to_vec(), w, h),
            v082::LinearRgbImage::new(b.to_vec(), w, h),
        )
        .unwrap(),
        head::compute_ssimulacra2(
            head::LinearRgbImage::new(a.to_vec(), w, h),
            head::LinearRgbImage::new(b.to_vec(), w, h),
        )
        .unwrap(),
        repro::compute_ssimulacra2(
            repro::LinearRgbImage::new(a.to_vec(), w, h),
            repro::LinearRgbImage::new(b.to_vec(), w, h),
        )
        .unwrap(),
        cbrt64::compute_ssimulacra2(
            cbrt64::LinearRgbImage::new(a.to_vec(), w, h),
            cbrt64::LinearRgbImage::new(b.to_vec(), w, h),
        )
        .unwrap(),
        cppcbrt::compute_ssimulacra2(
            cppcbrt::LinearRgbImage::new(a.to_vec(), w, h),
            cppcbrt::LinearRgbImage::new(b.to_vec(), w, h),
        )
        .unwrap(),
        cppcbrtu::compute_ssimulacra2(
            cppcbrtu::LinearRgbImage::new(a.to_vec(), w, h),
            cppcbrtu::LinearRgbImage::new(b.to_vec(), w, h),
        )
        .unwrap(),
        cppmax::compute_ssimulacra2(
            cppmax::LinearRgbImage::new(a.to_vec(), w, h),
            cppmax::LinearRgbImage::new(b.to_vec(), w, h),
        )
        .unwrap(),
    ]
}

fn stat(label: &str, v: &[f64]) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let meanabs = v.iter().map(|x| x.abs()).sum::<f64>() / n;
    let max = v.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let pos = v.iter().filter(|&&x| x > 0.0).count();
    let exact = v.iter().filter(|&&x| x == 0.0).count();
    println!(
        "{label:<26} mean {mean:>+12.7}  mean|.| {meanabs:>11.7}  max|.| {max:>10.6}  n>0 {pos:>4}/{}  identical {exact:>4}",
        v.len()
    );
}

fn main() {
    // The sRGB LUT must be identical across versions, or the comparison is
    // measuring linearisation instead of the metric.
    for v in 0..=255u8 {
        assert_eq!(
            head::srgb_u8_to_linear(v).to_bits(),
            v071::srgb_u8_to_linear(v).to_bits(),
            "sRGB LUT differs at {v} between HEAD and 0.7.1"
        );
        assert_eq!(
            head::srgb_u8_to_linear(v).to_bits(),
            v082::srgb_u8_to_linear(v).to_bits(),
            "sRGB LUT differs at {v} between HEAD and 0.8.2"
        );
    }
    eprintln!("sRGB LUT identical across 0.7.1 / 0.8.2 / HEAD");

    let bin = PathBuf::from(
        env::var("SSIMULACRA2_BIN").unwrap_or_else(|_| "/opt/homebrew/bin/ssimulacra2".into()),
    );
    let corpus = PathBuf::from(
        env::var("PHOTO_CORPUS").unwrap_or_else(|_| "/Users/lilith/work/zen/codec-corpus".into()),
    );
    let tmp = PathBuf::from(env::var("HOME").unwrap()).join("tmp/ssim2-parity-study/scratch");
    fs::create_dir_all(&tmp).unwrap();
    let limit: usize = env::var("PHOTO_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);

    let mut refs: Vec<(String, PathBuf)> = Vec::new();
    for (tag, rel) in [
        ("cid22", "CID22/CID22-512/training"),
        ("kadid", "kadid10k"),
        ("gb82", "gb82"),
        ("clic", "clic2025/training"),
    ] {
        let dir = corpus.join(rel);
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        let mut f: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "png").unwrap_or(false))
            .collect();
        f.sort();
        for p in f.into_iter().take(limit) {
            refs.push((
                format!("{tag}_{}", p.file_stem().unwrap().to_string_lossy()),
                p,
            ));
        }
    }
    eprintln!("{} references", refs.len());

    let tsv_path = PathBuf::from(env::var("HOME").unwrap())
        .join("tmp/ssim2-parity-study/cbrt_attribution.tsv");
    let mut tsv = File::create(&tsv_path).unwrap();
    write!(tsv, "reference\tsize\tdistortion\tcpp").unwrap();
    for n in NAMES {
        write!(tsv, "\t{n}").unwrap();
    }
    writeln!(tsv).unwrap();

    let mut rows: Vec<(String, f64, [f64; N_IMPL])> = Vec::new();

    for (name, path) in &refs {
        let Ok(rdr) = ImageReader::open(path) else { continue };
        let Ok(full) = rdr.decode() else { continue };
        let full = full.to_rgb8();
        for (size_tag, size) in [("64", Some(64u32)), ("256", Some(256)), ("full", None)] {
            let src = match size {
                Some(s) => {
                    let (w, h) = full.dimensions();
                    if w < s || h < s {
                        continue;
                    }
                    image::imageops::crop_imm(&full, (w - s) / 2, (h - s) / 2, s, s).to_image()
                }
                None => {
                    let (w, h) = full.dimensions();
                    if w > 768 || h > 768 {
                        image::imageops::resize(
                            &full,
                            w.min(768),
                            h.min(768),
                            image::imageops::FilterType::CatmullRom,
                        )
                    } else {
                        full.clone()
                    }
                }
            };
            let flat = flatten(&src, 0.12);
            for (dname, base, dimg) in [
                ("jpeg_q90".to_string(), &src, jpeg_roundtrip(&src, 90)),
                ("jpeg_q50".to_string(), &src, jpeg_roundtrip(&src, 50)),
                ("jpeg_q10".to_string(), &src, jpeg_roundtrip(&src, 10)),
                ("boxblur_r2".to_string(), &src, box_blur(&src, 2)),
                ("noise_a8".to_string(), &src, add_noise(&src, 8, 0x1234_5678)),
                ("flat_jpeg_q90".to_string(), &flat, jpeg_roundtrip(&flat, 90)),
                ("flat_noise_a2".to_string(), &flat, add_noise(&flat, 2, 0xabcd)),
            ] {
                let sp = tmp.join("s.png");
                let dp = tmp.join("d.png");
                base.save(&sp).unwrap();
                dimg.save(&dp).unwrap();
                let cpp = call_cpp(&bin, &sp, &dp);

                let (w, h) = base.dimensions();
                let (w, h) = (w as usize, h as usize);
                let a = lin(base);
                let b = lin(&dimg);
                let s = score_all(&a, &b, w, h);

                write!(tsv, "{name}\t{size_tag}\t{dname}\t{cpp:.12}").unwrap();
                for v in s {
                    write!(tsv, "\t{v:.12}").unwrap();
                }
                writeln!(tsv).unwrap();
                rows.push((format!("{name}/{size_tag}/{dname}"), cpp, s));
            }
        }
        eprint!(".");
    }
    eprintln!();
    println!("\n{} cells, {} references\n", rows.len(), refs.len());

    println!("=== vs the C++ SSIMULACRA2 binary ===");
    for (i, n) in NAMES.iter().enumerate() {
        let v: Vec<f64> = rows.iter().map(|r| r.2[i] - r.1).collect();
        stat(&format!("{n} - C++"), &v);
    }

    println!("\n=== attribution: is the 0.7.1 -> 0.8.2 move the cube root? ===");
    let pair = |a: usize, b: usize| {
        let v: Vec<f64> = rows.iter().map(|r| r.2[a] - r.2[b]).collect();
        stat(&format!("{} - {}", NAMES[a], NAMES[b]), &v);
    };
    pair(1, 0); // v082 - v071  : the historical divergence
    pair(3, 0); // repro - v071 : HEAD numerics + fused mm + f64 cbrt vs 0.7.1
    pair(4, 2); // cbrt64 - head: cube root alone, at HEAD
    pair(3, 4); // repro - cbrt64 : opsin matmul fusion alone
    pair(5, 2); // cppcbrt - head
    pair(6, 2); // cppcbrtu - head : jpegli cbrt alone
    pair(7, 5); // cppmax - cppcbrt : jpegli horizontal blur alone

    println!("\n=== closer-to-C++ head-to-head (of {} cells) ===", rows.len());
    for (i, n) in NAMES.iter().enumerate() {
        for (j, m) in NAMES.iter().enumerate() {
            if i >= j {
                continue;
            }
            let wins = rows
                .iter()
                .filter(|r| (r.2[i] - r.1).abs() < (r.2[j] - r.1).abs())
                .count();
            let ties = rows
                .iter()
                .filter(|r| (r.2[i] - r.1).abs() == (r.2[j] - r.1).abs())
                .count();
            println!("{n:<9} closer than {m:<9}: {wins:>4}   ties {ties:>4}");
        }
    }

    let mut worst: Vec<&(String, f64, [f64; N_IMPL])> = rows.iter().collect();
    worst.sort_by(|a, b| {
        (b.2[1] - b.2[0])
            .abs()
            .partial_cmp(&(a.2[1] - a.2[0]).abs())
            .unwrap()
    });
    println!("\n10 largest |0.8.2 - 0.7.1|:");
    print!("{:<50}{:>10}", "cell", "C++");
    for n in NAMES {
        print!("{n:>11}");
    }
    println!();
    for r in worst.iter().take(10) {
        print!("{:<50}{:>10.5}", r.0, r.1);
        for v in r.2 {
            print!("{v:>11.5}");
        }
        println!();
    }
    println!("\nTSV: {}", tsv_path.display());
}
