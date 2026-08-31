//! C++ parity on **real photographic content**, at four sizes, over five
//! distortion families.
//!
//! The compiled-in reference table is entirely synthetic — flat patches,
//! gradients, checkerboards, LCG noise — and 40 of its 66 cases compare an
//! image against *itself*, which every implementation scores exactly 100.0.
//! That leaves the parity gate resting almost entirely on `uniform_shift`,
//! which is numerically degenerate (see `src/cpp_parity_diag.rs`). This
//! example measures the thing that actually matters: agreement with the C++
//! binary on photographs that have real structure at every pyramid level.
//!
//! Usage:
//!   SSIMULACRA2_BIN=/opt/homebrew/bin/ssimulacra2 \
//!     cargo run --release --example photo_parity
//!
//!   PHOTO_CORPUS=/path/to/codec-corpus   (default: ../codec-corpus)
//!   PHOTO_TSV=/path/out.tsv              machine-readable table
//!   PHOTO_LIMIT=N                        references per corpus set (default 6)
//!   PARITY_TMP=~/tmp/...                 scratch PNGs

use std::env;
use std::fs::{self, File};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use fast_ssim2::{Ssimulacra2Config, compute_ssimulacra2_with_config};
use image::{ImageReader, RgbImage};
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn to_rgb(img: &RgbImage) -> Rgb {
    let (w, h) = img.dimensions();
    let px: Vec<[f32; 3]> = img
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
        px,
        std::num::NonZeroUsize::new(w as usize).unwrap(),
        std::num::NonZeroUsize::new(h as usize).unwrap(),
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap()
}

fn call_cpp(bin: &Path, source: &Path, distorted: &Path) -> Result<f64, String> {
    let output = Command::new(bin)
        .arg(source)
        .arg(distorted)
        .output()
        .map_err(|e| format!("exec {}: {e}", bin.display()))?;
    if !output.status.success() {
        return Err(format!(
            "ssimulacra2 failed on {}: {}",
            source.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(tok) = line.split_whitespace().last()
            && let Ok(score) = tok.parse::<f64>()
        {
            return Ok(score);
        }
    }
    Err(format!("unparseable output: {stdout}"))
}

/// Deterministic LCG, so a run is reproducible from the TSV alone.
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

fn jpeg_roundtrip(img: &RgbImage, quality: u8) -> RgbImage {
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut Cursor::new(&mut buf), quality)
        .encode_image(img)
        .expect("jpeg encode");
    image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg)
        .expect("jpeg decode")
        .to_rgb8()
}

fn box_blur(img: &RgbImage, radius: i32) -> RgbImage {
    let (w, h) = img.dimensions();
    let mut out = RgbImage::new(w, h);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = [0u32; 3];
            let mut n = 0u32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let sx = (x + dx).clamp(0, w as i32 - 1) as u32;
                    let sy = (y + dy).clamp(0, h as i32 - 1) as u32;
                    let p = img.get_pixel(sx, sy).0;
                    for c in 0..3 {
                        sum[c] += p[c] as u32;
                    }
                    n += 1;
                }
            }
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8]),
            );
        }
    }
    out
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

/// Chroma-ish shift: push one channel, leave the others. Exercises the X and
/// B planes, which the grey `uniform_shift` cases never touch.
fn channel_shift(img: &RgbImage, delta: i32) -> RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        p.0[0] = (p.0[0] as i32 + delta).clamp(0, 255) as u8;
        p.0[2] = (p.0[2] as i32 - delta).clamp(0, 255) as u8;
    }
    out
}

fn crop(img: &RgbImage, size: u32) -> Option<RgbImage> {
    let (w, h) = img.dimensions();
    if w < size || h < size {
        return None;
    }
    let (x0, y0) = ((w - size) / 2, (h - size) / 2);
    Some(image::imageops::crop_imm(img, x0, y0, size, size).to_image())
}

fn save_png(path: &Path, img: &RgbImage) -> Result<(), String> {
    img.save(path)
        .map_err(|e| format!("save {}: {e}", path.display()))
}

fn collect_refs(corpus: &Path, limit: usize) -> Vec<(String, PathBuf)> {
    // Named sets, chosen for content diversity: CID22 (compression-study
    // photos), KADID-10k (photographic IQA references), gb82 (screen content
    // and line art), CLIC2025 (large modern photographs).
    let sets: [(&str, &str); 4] = [
        ("cid22", "CID22/CID22-512/training"),
        ("kadid", "kadid10k"),
        ("gb82", "gb82"),
        ("clic", "clic2025/training"),
    ];
    let mut out = Vec::new();
    for (tag, rel) in sets {
        let dir = corpus.join(rel);
        let Ok(rd) = fs::read_dir(&dir) else {
            eprintln!("WARN: {} missing, set '{tag}' skipped", dir.display());
            continue;
        };
        let mut files: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "png").unwrap_or(false))
            .collect();
        files.sort(); // deterministic pick
        for p in files.into_iter().take(limit) {
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            out.push((format!("{tag}_{stem}"), p));
        }
    }
    out
}

/// `(reference, size tag, distortion, C++ score, simd score, scalar score)`
type Row = (String, String, String, f64, f64, f64);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bin = env::var("SSIMULACRA2_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/homebrew/bin/ssimulacra2"));
    if !bin.exists() && which::which(&bin).is_err() {
        return Err("set SSIMULACRA2_BIN to the C++ ssimulacra2 binary".into());
    }
    let corpus = env::var("PHOTO_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/lilith/work/zen/codec-corpus"));
    let limit: usize = env::var("PHOTO_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let tmp = env::var("PARITY_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env::var("HOME").expect("HOME")).join("tmp/fast-ssim2-parity/photo")
        });
    fs::create_dir_all(&tmp)?;

    let refs = collect_refs(&corpus, limit);
    println!("C++ binary: {}", bin.display());
    println!("corpus:     {}", corpus.display());
    println!("references: {}", refs.len());

    let mut tsv = env::var("PHOTO_TSV")
        .ok()
        .map(|p| File::create(p).expect("create PHOTO_TSV"));
    if let Some(f) = tsv.as_mut() {
        writeln!(f, "reference\tsize\tdistortion\tcpp\tsimd\tscalar")?;
    }

    // (name, size) sweep covers per-call fixed cost and the pyramid depth:
    // 32px reaches 3 scales, 64px 4, 256px 6, and "full" whatever the source
    // allows. Deep scales are where flat-field degeneracy creeps back in.
    let sizes: [(&str, Option<u32>); 4] = [
        ("32", Some(32)),
        ("64", Some(64)),
        ("256", Some(256)),
        ("full", None),
    ];

    let mut rows: Vec<Row> = Vec::new();

    for (name, path) in &refs {
        let src_full = match ImageReader::open(path)?.decode() {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("WARN: {} failed to decode: {e}", path.display());
                continue;
            }
        };
        for (size_tag, size) in sizes {
            let src = match size {
                Some(s) => match crop(&src_full, s) {
                    Some(c) => c,
                    None => continue,
                },
                // Cap "full" so a 2048px CLIC reference does not dominate wall time.
                None => {
                    let (w, h) = src_full.dimensions();
                    if w > 1024 || h > 1024 {
                        image::imageops::resize(
                            &src_full,
                            w.min(1024),
                            h.min(1024),
                            image::imageops::FilterType::CatmullRom,
                        )
                    } else {
                        src_full.clone()
                    }
                }
            };

            let variants: Vec<(String, RgbImage)> = vec![
                ("jpeg_q90".into(), jpeg_roundtrip(&src, 90)),
                ("jpeg_q50".into(), jpeg_roundtrip(&src, 50)),
                ("jpeg_q10".into(), jpeg_roundtrip(&src, 10)),
                ("boxblur_r2".into(), box_blur(&src, 2)),
                ("noise_a8".into(), add_noise(&src, 8, 0x1234_5678)),
                ("chroma_shift_6".into(), channel_shift(&src, 6)),
            ];

            let sp = tmp.join(format!("{name}_{size_tag}_src.png"));
            save_png(&sp, &src)?;
            for (dname, dimg) in variants {
                let dp = tmp.join(format!("{name}_{size_tag}_{dname}.png"));
                save_png(&dp, &dimg)?;
                let cpp = call_cpp(&bin, &sp, &dp)?;
                let simd = compute_ssimulacra2_with_config(
                    to_rgb(&src),
                    to_rgb(&dimg),
                    Ssimulacra2Config::simd(),
                )?;
                let scalar = compute_ssimulacra2_with_config(
                    to_rgb(&src),
                    to_rgb(&dimg),
                    Ssimulacra2Config::scalar(),
                )?;
                if let Some(f) = tsv.as_mut() {
                    writeln!(
                        f,
                        "{name}\t{size_tag}\t{dname}\t{cpp:.12}\t{simd:.12}\t{scalar:.12}"
                    )?;
                }
                rows.push((name.clone(), size_tag.into(), dname, cpp, simd, scalar));
                let _ = fs::remove_file(&dp);
            }
            let _ = fs::remove_file(&sp);
        }
        print!(".");
        std::io::stdout().flush()?;
    }
    println!("\n{} pairs scored", rows.len());

    let summarize = |label: &str, sel: &dyn Fn(&Row) -> bool| {
        let d: Vec<f64> = rows.iter().filter(|r| sel(r)).map(|r| r.4 - r.3).collect();
        let t: Vec<f64> = rows.iter().filter(|r| sel(r)).map(|r| r.4 - r.5).collect();
        if d.is_empty() {
            return;
        }
        let n = d.len();
        let mut sorted = d.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{label:<22} {n:>5} {:>+12.6} {:>+12.6} {:>+12.6} {:>12.6} {:>5} {:>13.9}",
            sorted[0],
            sorted[n - 1],
            d.iter().sum::<f64>() / n as f64,
            d.iter().map(|x| x.abs()).sum::<f64>() / n as f64,
            d.iter().filter(|&&x| x > 0.0).count(),
            t.iter().map(|x| x.abs()).fold(0.0f64, f64::max),
        );
    };

    println!(
        "\n{:<22} {:>5} {:>12} {:>12} {:>12} {:>12} {:>5} {:>13}",
        "group", "n", "min(s-c)", "max(s-c)", "mean(s-c)", "mean|s-c|", "n>0", "max|simd-scl|"
    );
    println!("{:-<105}", "");
    summarize("ALL", &|_| true);
    for s in ["32", "64", "256", "full"] {
        summarize(&format!("size={s}"), &|r| r.1 == s);
    }
    for dn in [
        "jpeg_q90",
        "jpeg_q50",
        "jpeg_q10",
        "boxblur_r2",
        "noise_a8",
        "chroma_shift_6",
    ] {
        summarize(&format!("dist={dn}"), &|r| r.2 == dn);
    }
    for tag in ["cid22", "kadid", "gb82", "clic"] {
        summarize(&format!("set={tag}"), &|r| r.0.starts_with(tag));
    }

    // Worst offenders, so a real divergence is not averaged away.
    let mut worst: Vec<&Row> = rows.iter().collect();
    worst.sort_by(|a, b| (b.4 - b.3).abs().partial_cmp(&(a.4 - a.3).abs()).unwrap());
    println!("\n{:-^105}", " 15 largest |simd - C++| ");
    println!(
        "{:<34} {:>5} {:<16} {:>12} {:>12} {:>12}",
        "reference", "size", "distortion", "C++", "simd", "simd-C++"
    );
    for r in worst.iter().take(15) {
        println!(
            "{:<34} {:>5} {:<16} {:>12.6} {:>12.6} {:>+12.6}",
            r.0,
            r.1,
            r.2,
            r.3,
            r.4,
            r.4 - r.3
        );
    }

    Ok(())
}
