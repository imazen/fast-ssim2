//! Profile blur passes separately
//!
//! Run with:
//!   cargo run --release --example profile_blur

use std::time::Instant;

fn main() {
    let width = 1024;
    let height = 1024;
    let iterations = 100;

    // Create test data
    let input: Vec<f32> = (0..width * height)
        .map(|i| i as f32 / (width * height) as f32)
        .collect();

    let mut temp = vec![0.0f32; width * height];
    let mut output = vec![0.0f32; width * height];

    // Benchmark horizontal pass (scalar IIR)
    let start = Instant::now();
    for _ in 0..iterations {
        horizontal_pass(&input, &mut temp, width, height);
    }
    let h_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // Benchmark vertical pass (SIMD)
    let start = Instant::now();
    for _ in 0..iterations {
        vertical_pass(&temp, &mut output, width, height);
    }
    let v_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    println!("Blur Pass Breakdown ({}x{}):", width, height);
    println!(
        "  Horizontal: {:.3} ms ({:.1}%)",
        h_time,
        h_time / (h_time + v_time) * 100.0
    );
    println!(
        "  Vertical:   {:.3} ms ({:.1}%)",
        v_time,
        v_time / (h_time + v_time) * 100.0
    );
    println!("  Total:      {:.3} ms", h_time + v_time);
}

const RADIUS: isize = 5;
const MUL_IN_1: f32 = 0.055295236;
const MUL_IN_3: f32 = -0.058836687;
const MUL_IN_5: f32 = 0.012955819;
const MUL_PREV_1: f32 = 1.9021131;
const MUL_PREV_3: f32 = 1.1755705;
const MUL_PREV_5: f32 = 0.00000000000000012246469;
const MUL_PREV2_1: f32 = -1.0;
const MUL_PREV2_3: f32 = -1.0;
const MUL_PREV2_5: f32 = -1.0;

fn horizontal_pass(input: &[f32], output: &mut [f32], width: usize, height: usize) {
    for y in 0..height {
        horizontal_row(
            &input[y * width..(y + 1) * width],
            &mut output[y * width..(y + 1) * width],
        );
    }
}

#[inline(always)]
fn horizontal_row(input: &[f32], output: &mut [f32]) {
    let width = input.len();
    let big_n = RADIUS;

    let mut prev_1 = 0.0f32;
    let mut prev_3 = 0.0f32;
    let mut prev_5 = 0.0f32;
    let mut prev2_1 = 0.0f32;
    let mut prev2_3 = 0.0f32;
    let mut prev2_5 = 0.0f32;

    let mut n = (-big_n) + 1;
    let width_i = width as isize;

    while n < width_i {
        let left = n - big_n - 1;
        let right = n + big_n - 1;

        let left_val = if left >= 0 && left < width_i {
            input[left as usize]
        } else {
            0.0f32
        };
        let right_val = if right >= 0 && right < width_i {
            input[right as usize]
        } else {
            0.0f32
        };

        let sum = left_val + right_val;

        let mut out_1 = sum * MUL_IN_1;
        let mut out_3 = sum * MUL_IN_3;
        let mut out_5 = sum * MUL_IN_5;

        out_1 = prev2_1.mul_add(MUL_PREV2_1, out_1);
        out_3 = prev2_3.mul_add(MUL_PREV2_3, out_3);
        out_5 = prev2_5.mul_add(MUL_PREV2_5, out_5);

        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;

        out_1 = prev_1.mul_add(MUL_PREV_1, out_1);
        out_3 = prev_3.mul_add(MUL_PREV_3, out_3);
        out_5 = prev_5.mul_add(MUL_PREV_5, out_5);

        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if n >= 0 {
            output[n as usize] = out_1 + out_3 + out_5;
        }

        n += 1;
    }
}

const VERT_MUL_IN_1: f32 = 0.055295236;
const VERT_MUL_IN_3: f32 = -0.058836687;
const VERT_MUL_IN_5: f32 = 0.012955819;
const VERT_MUL_PREV_1: f32 = -1.9021131;
const VERT_MUL_PREV_3: f32 = -1.1755705;
const VERT_MUL_PREV_5: f32 = -0.00000000000000012246469;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn vertical_pass(input: &[f32], output: &mut [f32], width: usize, height: usize) {
    let mut x = 0;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            while x + 8 <= width {
                unsafe {
                    vertical_column_avx2(input, output, width, height, x);
                }
                x += 8;
            }
        }

        while x + 4 <= width {
            unsafe {
                vertical_column_sse2(input, output, width, height, x);
            }
            x += 4;
        }
    }

    while x < width {
        vertical_column_scalar(input, output, width, height, x);
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vertical_column_avx2(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
) {
    unsafe {
        let big_n = RADIUS;
        let height_i = height as isize;

        let mul_in_1 = _mm256_set1_ps(VERT_MUL_IN_1);
        let mul_in_3 = _mm256_set1_ps(VERT_MUL_IN_3);
        let mul_in_5 = _mm256_set1_ps(VERT_MUL_IN_5);
        let mul_prev_1 = _mm256_set1_ps(VERT_MUL_PREV_1);
        let mul_prev_3 = _mm256_set1_ps(VERT_MUL_PREV_3);
        let mul_prev_5 = _mm256_set1_ps(VERT_MUL_PREV_5);

        let zeroes = _mm256_setzero_ps();

        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = (-big_n) + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_vals = if top >= 0 && top < height_i {
                _mm256_loadu_ps(input.as_ptr().add(top as usize * width + x))
            } else {
                zeroes
            };

            let bottom_vals = if bottom >= 0 && bottom < height_i {
                _mm256_loadu_ps(input.as_ptr().add(bottom as usize * width + x))
            } else {
                zeroes
            };

            let sum = _mm256_add_ps(top_vals, bottom_vals);

            let out1 = _mm256_fmadd_ps(prev_1, mul_prev_1, prev2_1);
            let out3 = _mm256_fmadd_ps(prev_3, mul_prev_3, prev2_3);
            let out5 = _mm256_fmadd_ps(prev_5, mul_prev_5, prev2_5);

            let out1 = _mm256_fmsub_ps(sum, mul_in_1, out1);
            let out3 = _mm256_fmsub_ps(sum, mul_in_3, out3);
            let out5 = _mm256_fmsub_ps(sum, mul_in_5, out5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            if n >= 0 {
                let result = _mm256_add_ps(_mm256_add_ps(out1, out3), out5);
                _mm256_storeu_ps(output.as_mut_ptr().add(n as usize * width + x), result);
            }

            n += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vertical_column_sse2(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
) {
    unsafe {
        let big_n = RADIUS;
        let height_i = height as isize;

        let mul_in_1 = _mm_set1_ps(VERT_MUL_IN_1);
        let mul_in_3 = _mm_set1_ps(VERT_MUL_IN_3);
        let mul_in_5 = _mm_set1_ps(VERT_MUL_IN_5);
        let mul_prev_1 = _mm_set1_ps(VERT_MUL_PREV_1);
        let mul_prev_3 = _mm_set1_ps(VERT_MUL_PREV_3);
        let mul_prev_5 = _mm_set1_ps(VERT_MUL_PREV_5);

        let zeroes = _mm_setzero_ps();

        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = (-big_n) + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_vals = if top >= 0 && top < height_i {
                _mm_loadu_ps(input.as_ptr().add(top as usize * width + x))
            } else {
                zeroes
            };

            let bottom_vals = if bottom >= 0 && bottom < height_i {
                _mm_loadu_ps(input.as_ptr().add(bottom as usize * width + x))
            } else {
                zeroes
            };

            let sum = _mm_add_ps(top_vals, bottom_vals);

            let out1 = _mm_add_ps(_mm_mul_ps(prev_1, mul_prev_1), prev2_1);
            let out3 = _mm_add_ps(_mm_mul_ps(prev_3, mul_prev_3), prev2_3);
            let out5 = _mm_add_ps(_mm_mul_ps(prev_5, mul_prev_5), prev2_5);

            let out1 = _mm_sub_ps(_mm_mul_ps(sum, mul_in_1), out1);
            let out3 = _mm_sub_ps(_mm_mul_ps(sum, mul_in_3), out3);
            let out5 = _mm_sub_ps(_mm_mul_ps(sum, mul_in_5), out5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            if n >= 0 {
                let result = _mm_add_ps(_mm_add_ps(out1, out3), out5);
                _mm_storeu_ps(output.as_mut_ptr().add(n as usize * width + x), result);
            }

            n += 1;
        }
    }
}

fn vertical_column_scalar(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
) {
    let big_n = RADIUS;
    let height_i = height as isize;

    let mut prev_1 = 0.0f32;
    let mut prev_3 = 0.0f32;
    let mut prev_5 = 0.0f32;
    let mut prev2_1 = 0.0f32;
    let mut prev2_3 = 0.0f32;
    let mut prev2_5 = 0.0f32;

    let mut n = (-big_n) + 1;
    while n < height_i {
        let top = n - big_n - 1;
        let bottom = n + big_n - 1;

        let top_val = if top >= 0 && top < height_i {
            input[top as usize * width + x]
        } else {
            0.0f32
        };

        let bottom_val = if bottom >= 0 && bottom < height_i {
            input[bottom as usize * width + x]
        } else {
            0.0f32
        };

        let sum = top_val + bottom_val;

        let out1 = prev_1.mul_add(VERT_MUL_PREV_1, prev2_1);
        let out3 = prev_3.mul_add(VERT_MUL_PREV_3, prev2_3);
        let out5 = prev_5.mul_add(VERT_MUL_PREV_5, prev2_5);

        let out1 = sum.mul_add(VERT_MUL_IN_1, -out1);
        let out3 = sum.mul_add(VERT_MUL_IN_3, -out3);
        let out5 = sum.mul_add(VERT_MUL_IN_5, -out5);

        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;
        prev_1 = out1;
        prev_3 = out3;
        prev_5 = out5;

        if n >= 0 {
            output[n as usize * width + x] = out1 + out3 + out5;
        }

        n += 1;
    }
}
