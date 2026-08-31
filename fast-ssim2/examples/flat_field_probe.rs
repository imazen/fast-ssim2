//! Why the `uniform_shift_*` reference cases disagree with the C++ binary.
//!
//! On a perfectly flat image the SSIM' term is numerically degenerate:
//! `sigma11 - mu1^2`, `sigma22 - mu2^2` and `sigma12 - mu1*mu2` are all
//! *analytically* zero away from the border, so both `num_s` and `denom_s`
//! collapse to the `kC2 = 9e-4` regularisation term plus whatever rounding
//! the Gaussian left behind. `d = max(1 - num_m*num_s/denom_s, 0)` then
//! rectifies that rounding noise into a strictly positive error contribution,
//! so the implementation with the *noisier* blur reports the *lower* score.
//!
//! This probe breaks the degeneracy by adding a controlled amount of texture
//! and watches the ours-vs-C++ delta as a function of texture amplitude. If
//! the disagreement is degeneracy, it collapses as soon as there is real
//! signal; if it is an algorithmic divergence, it does not.
//!
//! Usage:
//!   SSIMULACRA2_BIN=/opt/homebrew/bin/ssimulacra2 \
//!     cargo run --release --example flat_field_probe
//!
//!   PROBE_TSV=/path/out.tsv   -> machine-readable table
//!   PARITY_TMP=~/tmp/...      -> where the scratch PNGs go

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use fast_ssim2::{Ssimulacra2Config, compute_ssimulacra2_with_config};
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/dev/testcases.rs"));

fn to_rgb(data: &[u8], width: usize, height: usize) -> Rgb {
    let px: Vec<[f32; 3]> = data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();
    Rgb::new(
        px,
        std::num::NonZeroUsize::new(width).unwrap(),
        std::num::NonZeroUsize::new(height).unwrap(),
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap()
}

fn save_png(path: &Path, data: &[u8], width: usize, height: usize) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(file, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| format!("header: {e}"))?;
    writer
        .write_image_data(data)
        .map_err(|e| format!("data: {e}"))
}

fn call_cpp(bin: &Path, source: &Path, distorted: &Path) -> Result<f64, String> {
    let output = Command::new(bin)
        .arg(source)
        .arg(distorted)
        .output()
        .map_err(|e| format!("exec {}: {e}", bin.display()))?;
    if !output.status.success() {
        return Err(format!(
            "ssimulacra2 failed: {}",
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

/// Mid-gray plus symmetric `+/-amp` LCG texture, and the same field shifted
/// by `shift`. Both images carry *identical* texture, so the only real
/// difference between them is the constant shift — exactly the
/// `uniform_shift` case, but with the flat-field degeneracy removed.
fn textured_pair(
    width: usize,
    height: usize,
    amp: i32,
    shift: i32,
    seed: u64,
) -> (Vec<u8>, Vec<u8>) {
    let mut lcg = Lcg::new(seed);
    let mut src = Vec::with_capacity(width * height * 3);
    let mut dst = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        for _ in 0..3 {
            let n = if amp == 0 {
                0
            } else {
                (lcg.next_u8() as i32 % (2 * amp + 1)) - amp
            };
            src.push((128 + n).clamp(0, 255) as u8);
            dst.push((128 + n + shift).clamp(0, 255) as u8);
        }
    }
    (src, dst)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bin = env::var("SSIMULACRA2_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/homebrew/bin/ssimulacra2"));
    if !bin.exists() && which::which(&bin).is_err() {
        return Err("set SSIMULACRA2_BIN to the C++ ssimulacra2 binary".into());
    }
    let tmp = env::var("PARITY_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env::var("HOME").expect("HOME")).join("tmp/fast-ssim2-parity")
        });
    fs::create_dir_all(&tmp)?;
    println!("C++ binary: {}", bin.display());

    let mut tsv = env::var("PROBE_TSV")
        .ok()
        .map(|p| File::create(p).expect("create PROBE_TSV"));
    if let Some(f) = tsv.as_mut() {
        writeln!(f, "size\ttexture_amp\tshift\tcpp\tsimd\tscalar")?;
    }

    println!(
        "\n{:>6} {:>5} {:>6} {:>14} {:>14} {:>14} {:>13} {:>13}",
        "size", "amp", "shift", "C++", "simd", "scalar", "simd-C++", "scalar-simd"
    );
    println!("{:-<106}", "");

    for size in [32usize, 64, 256] {
        for shift in [1i32, 5, 20] {
            for amp in [0i32, 1, 2, 4, 8, 16, 32] {
                let (src, dst) = textured_pair(size, size, amp, shift, 0x5EED_1234);
                let sp = tmp.join(format!("probe_{size}_{amp}_{shift}_s.png"));
                let dp = tmp.join(format!("probe_{size}_{amp}_{shift}_d.png"));
                save_png(&sp, &src, size, size)?;
                save_png(&dp, &dst, size, size)?;
                let cpp = call_cpp(&bin, &sp, &dp)?;
                let simd = compute_ssimulacra2_with_config(
                    to_rgb(&src, size, size),
                    to_rgb(&dst, size, size),
                    Ssimulacra2Config::simd(),
                )?;
                let scalar = compute_ssimulacra2_with_config(
                    to_rgb(&src, size, size),
                    to_rgb(&dst, size, size),
                    Ssimulacra2Config::scalar(),
                )?;
                println!(
                    "{:>6} {:>5} {:>6} {:>14.8} {:>14.8} {:>14.8} {:>+13.8} {:>+13.8}",
                    size,
                    amp,
                    shift,
                    cpp,
                    simd,
                    scalar,
                    simd - cpp,
                    scalar - simd
                );
                if let Some(f) = tsv.as_mut() {
                    writeln!(
                        f,
                        "{size}\t{amp}\t{shift}\t{cpp:.12}\t{simd:.12}\t{scalar:.12}"
                    )?;
                }
            }
            println!();
        }
    }

    Ok(())
}
