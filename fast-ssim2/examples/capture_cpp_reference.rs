//! Capture reference scores from C++ ssimulacra2 implementation.
//!
//! This tool:
//! 1. Generates synthetic test images
//! 2. Calls the C++ ssimulacra2 binary to get reference scores
//! 3. Generates src/reference_data.rs with expected values
//!
//! Prerequisites:
//! - Build cloudinary/ssimulacra2 C++ binary
//! - Set SSIMULACRA2_BIN environment variable to point to it
//!
//! Usage:
//!   SSIMULACRA2_BIN=/path/to/ssimulacra2 cargo run --release --example capture_cpp_reference

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/dev/testcases.rs"));

/// Save RGB data as PNG
fn save_png(path: &Path, data: &[u8], width: usize, height: usize) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut encoder = png::Encoder::new(file, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("Failed to write PNG header: {}", e))?;
    writer
        .write_image_data(data)
        .map_err(|e| format!("Failed to write PNG data: {}", e))?;
    Ok(())
}

/// Call C++ ssimulacra2 binary
fn call_cpp_ssimulacra2(bin_path: &Path, source: &Path, distorted: &Path) -> Result<f64, String> {
    let output = Command::new(bin_path)
        .arg(source)
        .arg(distorted)
        .output()
        .map_err(|e| format!("Failed to execute ssimulacra2: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "ssimulacra2 failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Parse score from output (format: "score: 12.345" or just "12.345")
    for line in stdout.lines() {
        if let Some(score_str) = line.split_whitespace().last()
            && let Ok(score) = score_str.parse::<f64>()
        {
            return Ok(score);
        }
    }

    Err(format!("Could not parse score from output: {}", stdout))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get C++ binary path
    let bin_path = env::var("SSIMULACRA2_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("ssimulacra2"));

    if !bin_path.exists() && which::which(&bin_path).is_err() {
        eprintln!("ERROR: ssimulacra2 binary not found!");
        eprintln!("Set SSIMULACRA2_BIN=/path/to/ssimulacra2");
        eprintln!("Or ensure 'ssimulacra2' is in PATH");
        std::process::exit(1);
    }

    println!("Using C++ ssimulacra2 binary: {}", bin_path.display());

    // Create temp directory for test images
    let temp_dir = PathBuf::from("/tmp/ssimulacra2_reference");
    fs::create_dir_all(&temp_dir)?;
    println!("Temp directory: {}", temp_dir.display());

    // Generate test cases
    let test_cases = generate_test_cases();
    println!("Generated {} test cases", test_cases.len());

    // Capture reference scores
    let mut reference_cases = Vec::new();
    let mut failed = 0;

    for (i, case) in test_cases.iter().enumerate() {
        print!("[{:3}/{}] {:<50} ... ", i + 1, test_cases.len(), case.name);
        std::io::stdout().flush()?;

        // Save images
        let source_path = temp_dir.join(format!("{}_source.png", case.name));
        let distorted_path = temp_dir.join(format!("{}_distorted.png", case.name));

        save_png(&source_path, &case.source_data, case.width, case.height)?;
        save_png(
            &distorted_path,
            &case.distorted_data,
            case.width,
            case.height,
        )?;

        // Call C++ ssimulacra2
        match call_cpp_ssimulacra2(&bin_path, &source_path, &distorted_path) {
            Ok(score) => {
                println!("score = {:.15}", score);
                reference_cases.push((
                    case.name.clone(),
                    case.width,
                    case.height,
                    score,
                    case.source_hash.clone(),
                    case.distorted_hash.clone(),
                ));
            }
            Err(e) => {
                println!("FAILED: {}", e);
                failed += 1;
            }
        }
    }

    if failed > 0 {
        eprintln!("\nWARNING: {} test cases failed", failed);
    }

    // Generate reference_data.rs
    generate_reference_file(&reference_cases)?;

    println!(
        "\nDone! Generated {} reference cases",
        reference_cases.len()
    );
    println!("Output: ssimulacra2/src/reference_data.rs");

    Ok(())
}

fn generate_reference_file(
    cases: &[(String, usize, usize, f64, String, String)],
) -> std::io::Result<()> {
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/reference_data.rs");
    let mut f = File::create(&output_path)?;

    writeln!(f, "//! Auto-generated C++ ssimulacra2 reference data.")?;
    writeln!(f, "//!")?;
    writeln!(
        f,
        "//! Generated by: cargo run --example capture_cpp_reference"
    )?;
    writeln!(
        f,
        "//! Date: {}",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;
    writeln!(f, "//! Total test cases: {}", cases.len())?;
    writeln!(f, "//!")?;
    writeln!(
        f,
        "//! This file contains reference values captured from the C++ ssimulacra2"
    )?;
    writeln!(
        f,
        "//! implementation. These values are used for regression testing without"
    )?;
    writeln!(f, "//! requiring the C++ binary at test runtime.")?;
    writeln!(f)?;
    writeln!(f, "#![allow(clippy::excessive_precision)]")?;
    writeln!(f)?;
    writeln!(
        f,
        "/// A reference test case with expected C++ ssimulacra2 score."
    )?;
    writeln!(f, "#[derive(Debug, Clone)]")?;
    writeln!(f, "pub struct ReferenceCase {{")?;
    writeln!(f, "    pub name: &'static str,")?;
    writeln!(f, "    pub width: usize,")?;
    writeln!(f, "    pub height: usize,")?;
    writeln!(f, "    pub expected_score: f64,")?;
    writeln!(
        f,
        "    /// SHA256 hash of source image raw RGB data (for detecting generation changes)"
    )?;
    writeln!(f, "    pub source_hash: &'static str,")?;
    writeln!(
        f,
        "    /// SHA256 hash of distorted image raw RGB data (for detecting generation changes)"
    )?;
    writeln!(f, "    pub distorted_hash: &'static str,")?;
    writeln!(f, "}}")?;
    writeln!(f)?;
    writeln!(f, "/// All reference test cases.")?;
    writeln!(f, "pub const REFERENCE_CASES: &[ReferenceCase] = &[")?;

    for (name, width, height, score, source_hash, distorted_hash) in cases {
        writeln!(f, "    ReferenceCase {{")?;
        writeln!(f, "        name: \"{}\",", name)?;
        writeln!(f, "        width: {},", width)?;
        writeln!(f, "        height: {},", height)?;
        writeln!(f, "        expected_score: {:.15},", score)?;
        writeln!(f, "        source_hash: \"{}\",", source_hash)?;
        writeln!(f, "        distorted_hash: \"{}\",", distorted_hash)?;
        writeln!(f, "    }},")?;
    }

    writeln!(f, "];")?;

    println!("Wrote {} to {}", cases.len(), output_path.display());
    Ok(())
}
