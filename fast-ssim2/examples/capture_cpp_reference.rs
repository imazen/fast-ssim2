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

use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Test image generator
struct TestImageGenerator;

impl TestImageGenerator {
    /// Generate uniform color image
    fn uniform(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![r, g, b]
            .into_iter()
            .cycle()
            .take(width * height * 3)
            .collect()
    }

    /// Generate horizontal gradient
    fn gradient_h(width: usize, height: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        for _y in 0..height {
            for x in 0..width {
                let val = if width > 1 {
                    (x * 255 / (width - 1)) as u8
                } else {
                    128
                };
                data.extend_from_slice(&[val, val, val]);
            }
        }
        data
    }

    /// Generate vertical gradient
    fn gradient_v(width: usize, height: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            let val = if height > 1 {
                (y * 255 / (height - 1)) as u8
            } else {
                128
            };
            for _x in 0..width {
                data.extend_from_slice(&[val, val, val]);
            }
        }
        data
    }

    /// Generate diagonal gradient
    fn gradient_diag(width: usize, height: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        let max_dist = width + height - 2;
        for y in 0..height {
            for x in 0..width {
                let val = if max_dist > 0 {
                    ((x + y) * 255 / max_dist) as u8
                } else {
                    128
                };
                data.extend_from_slice(&[val, val, val]);
            }
        }
        data
    }

    /// Generate checkerboard pattern
    fn checkerboard(width: usize, height: usize, cell_size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let val = if ((x / cell_size) + (y / cell_size)).is_multiple_of(2) {
                    255
                } else {
                    0
                };
                data.extend_from_slice(&[val, val, val]);
            }
        }
        data
    }

    /// Generate random noise (deterministic LCG)
    fn noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
        let mut lcg = Lcg::new(seed);
        let mut data = Vec::with_capacity(width * height * 3);
        for _ in 0..width * height {
            data.push(lcg.next_u8());
            data.push(lcg.next_u8());
            data.push(lcg.next_u8());
        }
        data
    }

    /// Generate edge pattern (sharp transition)
    fn edge(width: usize, height: usize, vertical: bool) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let val = if vertical {
                    if x < width / 2 { 0 } else { 255 }
                } else if y < height / 2 {
                    0
                } else {
                    255
                };
                data.extend_from_slice(&[val, val, val]);
            }
        }
        data
    }

    /// Apply 8x8 box blur distortion
    fn box_blur_8x8(input: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut output = vec![0u8; width * height * 3];
        const KERNEL_SIZE: i32 = 8;
        const HALF_KERNEL: i32 = KERNEL_SIZE / 2;

        for y in 0..height {
            for x in 0..width {
                let mut sum = [0u32; 3];
                let mut count = 0u32;

                for ky in -HALF_KERNEL..HALF_KERNEL {
                    for kx in -HALF_KERNEL..HALF_KERNEL {
                        let ny = (y as i32 + ky).clamp(0, height as i32 - 1) as usize;
                        let nx = (x as i32 + kx).clamp(0, width as i32 - 1) as usize;
                        let idx = (ny * width + nx) * 3;
                        sum[0] += input[idx] as u32;
                        sum[1] += input[idx + 1] as u32;
                        sum[2] += input[idx + 2] as u32;
                        count += 1;
                    }
                }

                let out_idx = (y * width + x) * 3;
                output[out_idx] = (sum[0] / count) as u8;
                output[out_idx + 1] = (sum[1] / count) as u8;
                output[out_idx + 2] = (sum[2] / count) as u8;
            }
        }
        output
    }

    /// Apply simple sharpen filter
    fn sharpen(input: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut output = vec![0u8; width * height * 3];
        // Simple 3x3 sharpen kernel: [0 -1 0; -1 5 -1; 0 -1 0]
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let idx = (y * width + x) * 3 + c;
                    let center = input[idx] as i32;

                    let top = if y > 0 {
                        input[((y - 1) * width + x) * 3 + c] as i32
                    } else {
                        center
                    };
                    let bottom = if y < height - 1 {
                        input[((y + 1) * width + x) * 3 + c] as i32
                    } else {
                        center
                    };
                    let left = if x > 0 {
                        input[(y * width + (x - 1)) * 3 + c] as i32
                    } else {
                        center
                    };
                    let right = if x < width - 1 {
                        input[(y * width + (x + 1)) * 3 + c] as i32
                    } else {
                        center
                    };

                    let sharpened = 5 * center - top - bottom - left - right;
                    output[idx] = sharpened.clamp(0, 255) as u8;
                }
            }
        }
        output
    }

    /// Apply RGB → YUV → RGB roundtrip (using simple BT.601 matrix)
    fn yuv_roundtrip(input: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut output = vec![0u8; width * height * 3];

        for i in 0..width * height {
            let idx = i * 3;
            let r = input[idx] as f32;
            let g = input[idx + 1] as f32;
            let b = input[idx + 2] as f32;

            // RGB → YUV (BT.601)
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0;
            let v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0;

            // YUV → RGB
            let r_out = y + 1.13983 * (v - 128.0);
            let g_out = y - 0.39465 * (u - 128.0) - 0.58060 * (v - 128.0);
            let b_out = y + 2.03211 * (u - 128.0);

            output[idx] = r_out.clamp(0.0, 255.0) as u8;
            output[idx + 1] = g_out.clamp(0.0, 255.0) as u8;
            output[idx + 2] = b_out.clamp(0.0, 255.0) as u8;
        }
        output
    }
}

/// LCG pseudo-random number generator
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u8(&mut self) -> u8 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) & 0xFF) as u8
    }
}

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

/// Test case definition
#[derive(Debug)]
struct TestCase {
    name: String,
    width: usize,
    height: usize,
    source_data: Vec<u8>,
    distorted_data: Vec<u8>,
    source_hash: String,
    distorted_hash: String,
}

impl TestCase {
    fn new(
        name: String,
        width: usize,
        height: usize,
        source_data: Vec<u8>,
        distorted_data: Vec<u8>,
    ) -> Self {
        let source_hash = Sha256::digest(&source_data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let distorted_hash = Sha256::digest(&distorted_data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Self {
            name,
            width,
            height,
            source_data,
            distorted_data,
            source_hash,
            distorted_hash,
        }
    }
}

/// Generate all test cases
fn generate_test_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // Sizes to test
    let sizes = [(32, 32), (64, 64), (128, 128), (256, 256)];

    for (width, height) in sizes {
        // Perfect match (should score 100)
        let data = TestImageGenerator::uniform(width, height, 128, 128, 128);
        cases.push(TestCase::new(
            format!("perfect_match_{}x{}", width, height),
            width,
            height,
            data.clone(),
            data,
        ));

        // Uniform colors with slight shift
        for shift in [1, 5, 10, 20, 50] {
            let source = TestImageGenerator::uniform(width, height, 128, 128, 128);
            let distorted =
                TestImageGenerator::uniform(width, height, 128 + shift, 128 + shift, 128 + shift);
            cases.push(TestCase::new(
                format!("uniform_shift_{}_{}x{}", shift, width, height),
                width,
                height,
                source,
                distorted,
            ));
        }

        // Gradients (identical = should score high)
        let grad_h = TestImageGenerator::gradient_h(width, height);
        cases.push(TestCase::new(
            format!("gradient_h_{}x{}", width, height),
            width,
            height,
            grad_h.clone(),
            grad_h,
        ));

        let grad_v = TestImageGenerator::gradient_v(width, height);
        cases.push(TestCase::new(
            format!("gradient_v_{}x{}", width, height),
            width,
            height,
            grad_v.clone(),
            grad_v,
        ));

        // Checkerboard (identical)
        for cell_size in [4, 8, 16] {
            let checker = TestImageGenerator::checkerboard(width, height, cell_size);
            cases.push(TestCase::new(
                format!("checkerboard_{}_{}x{}", cell_size, width, height),
                width,
                height,
                checker.clone(),
                checker,
            ));
        }

        // Random noise (identical)
        for seed in [42, 123, 999] {
            let noise = TestImageGenerator::noise(width, height, seed);
            cases.push(TestCase::new(
                format!("noise_seed_{}_{}x{}", seed, width, height),
                width,
                height,
                noise.clone(),
                noise,
            ));
        }

        // Edges (identical)
        let edge_v = TestImageGenerator::edge(width, height, true);
        cases.push(TestCase::new(
            format!("edge_vertical_{}x{}", width, height),
            width,
            height,
            edge_v.clone(),
            edge_v,
        ));
    }

    // Only for smallest size: test distorted vs source
    let width = 64;
    let height = 64;

    // Gradient vs uniform
    let grad = TestImageGenerator::gradient_h(width, height);
    let uniform = TestImageGenerator::uniform(width, height, 128, 128, 128);
    cases.push(TestCase::new(
        format!("gradient_vs_uniform_{}x{}", width, height),
        width,
        height,
        grad,
        uniform,
    ));

    // Noise vs uniform
    let noise = TestImageGenerator::noise(width, height, 42);
    let uniform = TestImageGenerator::uniform(width, height, 128, 128, 128);
    cases.push(TestCase::new(
        format!("noise_vs_uniform_{}x{}", width, height),
        width,
        height,
        noise,
        uniform,
    ));

    // Distortion tests: apply realistic image degradations
    // Box blur 8x8
    let source = TestImageGenerator::gradient_h(width, height);
    let blurred = TestImageGenerator::box_blur_8x8(&source, width, height);
    cases.push(TestCase::new(
        format!("gradient_vs_boxblur8x8_{}x{}", width, height),
        width,
        height,
        source,
        blurred,
    ));

    // Sharpen filter
    let source = TestImageGenerator::noise(width, height, 999);
    let sharpened = TestImageGenerator::sharpen(&source, width, height);
    cases.push(TestCase::new(
        format!("noise_vs_sharpen_{}x{}", width, height),
        width,
        height,
        source,
        sharpened,
    ));

    // YUV roundtrip
    let source = TestImageGenerator::gradient_diag(width, height);
    let yuv_roundtrip = TestImageGenerator::yuv_roundtrip(&source, width, height);
    cases.push(TestCase::new(
        format!("gradient_vs_yuv_roundtrip_{}x{}", width, height),
        width,
        height,
        source,
        yuv_roundtrip,
    ));

    // Edge pattern with box blur
    let source = TestImageGenerator::edge(width, height, true);
    let blurred = TestImageGenerator::box_blur_8x8(&source, width, height);
    cases.push(TestCase::new(
        format!("edge_vs_boxblur8x8_{}x{}", width, height),
        width,
        height,
        source,
        blurred,
    ));

    cases
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
