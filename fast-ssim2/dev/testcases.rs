// Shared synthetic-image generators for the C++ parity harness.
//
// `include!`d (not `mod`-ed) by:
//   - examples/capture_cpp_reference.rs  (regenerates src/reference_data.rs)
//   - examples/parity_report.rs          (signed-delta report vs the C++ binary)
//   - tests/reference_parity.rs          (the compiled-in parity gate)
//
// These three used to carry three hand-synchronised copies of the generators,
// with a comment in the test reading "must match capture_cpp_reference.rs
// exactly". One file, no drift. Excluded from the published crate.

/// LCG pseudo-random number generator (deterministic across platforms).
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u8(&mut self) -> u8 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) & 0xFF) as u8
    }
}

/// Test image generator.
pub struct TestImageGenerator;

impl TestImageGenerator {
    /// Generate uniform color image
    pub fn uniform(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![r, g, b]
            .into_iter()
            .cycle()
            .take(width * height * 3)
            .collect()
    }

    /// Generate horizontal gradient
    pub fn gradient_h(width: usize, height: usize) -> Vec<u8> {
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
    pub fn gradient_v(width: usize, height: usize) -> Vec<u8> {
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
    pub fn gradient_diag(width: usize, height: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        let max_dist = width + height - 2;
        for y in 0..height {
            for x in 0..width {
                let val = match ((x + y) * 255).checked_div(max_dist) {
                    Some(v) => v as u8,
                    None => 128,
                };
                data.extend_from_slice(&[val, val, val]);
            }
        }
        data
    }

    /// Generate checkerboard pattern
    pub fn checkerboard(width: usize, height: usize, cell_size: usize) -> Vec<u8> {
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
    pub fn noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
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
    pub fn edge(width: usize, height: usize, vertical: bool) -> Vec<u8> {
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
    pub fn box_blur_8x8(input: &[u8], width: usize, height: usize) -> Vec<u8> {
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

    /// Apply simple 3x3 sharpen filter: [0 -1 0; -1 5 -1; 0 -1 0]
    pub fn sharpen(input: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut output = vec![0u8; width * height * 3];
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

    /// Apply RGB -> YUV -> RGB roundtrip (simple BT.601 matrix)
    pub fn yuv_roundtrip(input: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut output = vec![0u8; width * height * 3];

        for i in 0..width * height {
            let idx = i * 3;
            let r = input[idx] as f32;
            let g = input[idx + 1] as f32;
            let b = input[idx + 2] as f32;

            // RGB -> YUV (BT.601)
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0;
            let v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0;

            // YUV -> RGB
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

/// One (source, distorted) pair with the raw-RGB hashes the parity gate
/// uses to detect silent changes to image generation.
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub source_data: Vec<u8>,
    pub distorted_data: Vec<u8>,
    pub source_hash: String,
    pub distorted_hash: String,
}

impl TestCase {
    pub fn new(
        name: String,
        width: usize,
        height: usize,
        source_data: Vec<u8>,
        distorted_data: Vec<u8>,
    ) -> Self {
        let source_hash = sha256_hex(&source_data);
        let distorted_hash = sha256_hex(&distorted_data);
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

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// Every synthetic case in the compiled-in reference table, in table order.
pub fn generate_test_cases() -> Vec<TestCase> {
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

        // Gradients (identical = should score 100)
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

    // Only for one size: distorted vs source
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
