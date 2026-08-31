//! Signed-delta parity report against the C++ SSIMULACRA2 reference.
//!
//! `tests/reference_parity.rs` only asserts |error| <= a per-pattern tolerance,
//! so it cannot distinguish symmetric FP noise from a one-directional bias.
//! This tool prints the **signed** delta (ours - C++) for every case, for both
//! the SIMD and the scalar backend, and writes a TSV for `benchmarks/`.
//!
//! Usage:
//!   cargo run --release --example parity_report
//!     -> compares against the compiled-in table in src/reference_data.rs
//!
//!   SSIMULACRA2_BIN=/path/to/ssimulacra2 cargo run --release --example parity_report
//!     -> ALSO re-runs the C++ binary live, so the table itself is verified
//!
//!   PARITY_TSV=/path/out.tsv  -> write the machine-readable table there
//!   PARITY_TMP=~/tmp/...      -> where the live-mode PNGs are written

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use fast_ssim2::reference_data::REFERENCE_CASES;
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

fn pattern_of(name: &str) -> &'static str {
    if name.contains("uniform_shift") {
        "uniform_shift"
    } else if name.contains("boxblur8x8")
        || name.contains("sharpen")
        || name.contains("yuv_roundtrip")
    {
        "distortions"
    } else if name.contains("_vs_") {
        "synthetic_vs"
    } else if name.starts_with("perfect_match") {
        "perfect_match"
    } else if name.starts_with("gradient") {
        "gradients"
    } else if name.starts_with("checkerboard") {
        "checkerboard"
    } else if name.starts_with("noise_seed") {
        "noise"
    } else if name.starts_with("edge") {
        "edges"
    } else {
        "other"
    }
}

struct Row {
    name: String,
    pattern: &'static str,
    table: f64,
    live: Option<f64>,
    simd: f64,
    scalar: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cases = generate_test_cases();

    // Live C++ mode is opt-in: without the binary we still report ours-vs-table.
    let bin = env::var("SSIMULACRA2_BIN").ok().map(PathBuf::from);
    let live_bin = bin.filter(|b| b.exists() || which::which(b).is_ok());
    let tmp = env::var("PARITY_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env::var("HOME").expect("HOME")).join("tmp/fast-ssim2-parity")
        });
    if let Some(b) = live_bin.as_ref() {
        fs::create_dir_all(&tmp)?;
        println!("live C++ binary: {}", b.display());
        println!("scratch PNGs:    {}", tmp.display());
    } else {
        println!("no SSIMULACRA2_BIN -> comparing against the compiled-in table only");
    }

    let mut rows = Vec::new();
    for case in &cases {
        let Some(expected) = REFERENCE_CASES.iter().find(|r| r.name == case.name) else {
            eprintln!(
                "WARN: {} is not in the compiled-in table, skipped",
                case.name
            );
            continue;
        };
        assert_eq!(
            expected.source_hash, case.source_hash,
            "source image generation changed for {}",
            case.name
        );
        assert_eq!(
            expected.distorted_hash, case.distorted_hash,
            "distorted image generation changed for {}",
            case.name
        );

        let simd = compute_ssimulacra2_with_config(
            to_rgb(&case.source_data, case.width, case.height),
            to_rgb(&case.distorted_data, case.width, case.height),
            Ssimulacra2Config::simd(),
        )?;
        let scalar = compute_ssimulacra2_with_config(
            to_rgb(&case.source_data, case.width, case.height),
            to_rgb(&case.distorted_data, case.width, case.height),
            Ssimulacra2Config::scalar(),
        )?;

        let live = if let Some(bin) = &live_bin {
            let sp = tmp.join(format!("{}_source.png", case.name));
            let dp = tmp.join(format!("{}_distorted.png", case.name));
            save_png(&sp, &case.source_data, case.width, case.height)?;
            save_png(&dp, &case.distorted_data, case.width, case.height)?;
            Some(call_cpp(bin, &sp, &dp)?)
        } else {
            None
        };

        rows.push(Row {
            name: case.name.clone(),
            pattern: pattern_of(&case.name),
            table: expected.expected_score,
            live,
            simd,
            scalar,
        });
    }

    // Authority = the live binary when we have it, else the captured table.
    let authority = |r: &Row| r.live.unwrap_or(r.table);

    println!(
        "\n{:<38} {:>14} {:>14} {:>14} {:>13} {:>13}",
        "case", "C++", "simd", "scalar", "simd-C++", "scalar-simd"
    );
    println!("{:-<112}", "");
    let mut sorted: Vec<&Row> = rows.iter().collect();
    sorted.sort_by(|a, b| {
        (authority(b) - b.simd)
            .abs()
            .partial_cmp(&(authority(a) - a.simd).abs())
            .unwrap()
    });
    for r in &sorted {
        println!(
            "{:<38} {:>14.8} {:>14.8} {:>14.8} {:>+13.8} {:>+13.8}",
            r.name,
            authority(r),
            r.simd,
            r.scalar,
            r.simd - authority(r),
            r.scalar - r.simd
        );
    }

    // Per-pattern signed summary: a one-directional bias shows up as
    // mean(signed) == mean(|signed|), which FP noise never does.
    println!("\n{:-^112}", " signed delta (ours_simd - C++) by pattern ");
    println!(
        "{:<16} {:>5} {:>12} {:>12} {:>12} {:>12} {:>6} {:>6}",
        "pattern", "n", "min", "max", "mean", "mean|.|", "n>0", "n<0"
    );
    let mut pats: Vec<&'static str> = rows.iter().map(|r| r.pattern).collect();
    pats.sort_unstable();
    pats.dedup();
    for p in pats {
        let d: Vec<f64> = rows
            .iter()
            .filter(|r| r.pattern == p)
            .map(|r| r.simd - authority(r))
            .collect();
        let n = d.len();
        let pos = d.iter().filter(|&&x| x > 0.0).count();
        let neg = d.iter().filter(|&&x| x < 0.0).count();
        println!(
            "{:<16} {:>5} {:>+12.8} {:>+12.8} {:>+12.8} {:>12.8} {:>6} {:>6}",
            p,
            n,
            d.iter().copied().fold(f64::INFINITY, f64::min),
            d.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            d.iter().sum::<f64>() / n as f64,
            d.iter().map(|x| x.abs()).sum::<f64>() / n as f64,
            pos,
            neg
        );
    }

    // SIMD-vs-scalar tier agreement, measured rather than asserted.
    let tier_max = rows
        .iter()
        .map(|r| (r.simd - r.scalar).abs())
        .fold(0.0f64, f64::max);
    let tier_ne = rows.iter().filter(|r| r.simd != r.scalar).count();
    println!(
        "\nsimd vs scalar: max |delta| = {:.17}, cases differing at all = {}/{}",
        tier_max,
        tier_ne,
        rows.len()
    );

    if let Some(path) = env::var("PARITY_TSV").ok().map(PathBuf::from) {
        let mut f = File::create(&path)?;
        writeln!(f, "case\tpattern\tcpp_table\tcpp_live\tsimd\tscalar")?;
        for r in &rows {
            writeln!(
                f,
                "{}\t{}\t{:.12}\t{}\t{:.12}\t{:.12}",
                r.name,
                r.pattern,
                r.table,
                r.live.map(|v| format!("{v:.12}")).unwrap_or_default(),
                r.simd,
                r.scalar
            )?;
        }
        println!("wrote {}", path.display());
    }

    Ok(())
}
