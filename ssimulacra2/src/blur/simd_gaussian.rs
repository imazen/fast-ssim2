/// SIMD-optimized Recursive Gaussian blur
///
/// Uses archmage/magetypes for cross-platform SIMD in the vertical pass.
/// Horizontal pass is a scalar IIR filter (no SIMD benefit for sequential IIR).
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

#[cfg(target_arch = "x86_64")]
const LANES: usize = 8;

pub struct SimdGaussian {
    temp_buffer: Vec<f32>,
    max_size: usize,
}

impl SimdGaussian {
    pub fn new(max_width: usize) -> Self {
        const MAX_HEIGHT: usize = 4096;
        let max_size = max_width * MAX_HEIGHT;
        Self {
            temp_buffer: vec![0.0; max_size],
            max_size,
        }
    }

    pub fn shrink_to(&mut self, width: usize, height: usize) {
        let needed = width * height;
        if needed > self.max_size {
            self.temp_buffer.resize(needed, 0.0);
            self.max_size = needed;
        }
    }

    #[allow(dead_code)]
    pub fn blur_single_plane(&mut self, plane: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut out = vec![0.0; width * height];
        self.blur_single_plane_into(plane, &mut out, width, height);
        out
    }

    pub fn blur_single_plane_into(
        &mut self,
        plane: &[f32],
        out: &mut [f32],
        width: usize,
        height: usize,
    ) {
        let size = width * height;
        if size > self.max_size {
            self.temp_buffer.resize(size, 0.0);
            self.max_size = size;
        }

        // Horizontal pass: scalar IIR filter
        horizontal_pass(plane, &mut self.temp_buffer[..size], width);

        // Vertical pass: SIMD-dispatched
        vertical_pass(&self.temp_buffer[..size], out, width, height);
    }
}

// ---------------------------------------------------------------------------
// Horizontal pass — scalar IIR filter
// ---------------------------------------------------------------------------

fn horizontal_pass(input: &[f32], output: &mut [f32], width: usize) {
    assert_eq!(input.len(), output.len());

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        input
            .par_chunks_exact(width)
            .zip(output.par_chunks_exact_mut(width))
            .for_each(|(inp, out)| horizontal_row(inp, out, width));
    }

    #[cfg(not(feature = "rayon"))]
    {
        input
            .chunks_exact(width)
            .zip(output.chunks_exact_mut(width))
            .for_each(|(inp, out)| horizontal_row(inp, out, width));
    }
}

#[inline(always)]
fn horizontal_row(input: &[f32], output: &mut [f32], width: usize) {
    let big_n = consts::RADIUS as isize;

    let mut prev_1 = 0f32;
    let mut prev_3 = 0f32;
    let mut prev_5 = 0f32;
    let mut prev2_1 = 0f32;
    let mut prev2_3 = 0f32;
    let mut prev2_5 = 0f32;

    let mut n = (-big_n) + 1;
    while n < width as isize {
        let left = n - big_n - 1;
        let right = n + big_n - 1;
        let left_val = if left >= 0 && (left as usize) < input.len() {
            input[left as usize]
        } else {
            0f32
        };
        let right_val = if right >= 0 && (right as usize) < input.len() {
            input[right as usize]
        } else {
            0f32
        };
        let sum = left_val + right_val;

        let mut out_1 = sum * consts::MUL_IN_1;
        let mut out_3 = sum * consts::MUL_IN_3;
        let mut out_5 = sum * consts::MUL_IN_5;

        out_1 = consts::MUL_PREV2_1.mul_add(prev2_1, out_1);
        out_3 = consts::MUL_PREV2_3.mul_add(prev2_3, out_3);
        out_5 = consts::MUL_PREV2_5.mul_add(prev2_5, out_5);
        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;

        out_1 = consts::MUL_PREV_1.mul_add(prev_1, out_1);
        out_3 = consts::MUL_PREV_3.mul_add(prev_3, out_3);
        out_5 = consts::MUL_PREV_5.mul_add(prev_5, out_5);
        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if n >= 0 && (n as usize) < output.len() {
            output[n as usize] = out_1 + out_3 + out_5;
        }

        n += 1;
    }
}

// ---------------------------------------------------------------------------
// Vertical pass — SIMD IIR filter processing LANES columns in parallel
// ---------------------------------------------------------------------------

fn vertical_pass(input: &[f32], output: &mut [f32], width: usize, height: usize) {
    assert_eq!(input.len(), output.len());
    incant!(vertical_pass_inner(input, output, width, height))
}

/// AVX2 vertical pass — processes 8 columns at a time.
#[cfg(target_arch = "x86_64")]
fn vertical_pass_inner_v3(
    token: archmage::X64V3Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
) {
    let big_n = consts::RADIUS as isize;

    let mul_in_1 = f32x8::splat(token, consts::VERT_MUL_IN_1);
    let mul_in_3 = f32x8::splat(token, consts::VERT_MUL_IN_3);
    let mul_in_5 = f32x8::splat(token, consts::VERT_MUL_IN_5);
    let mul_prev_1 = f32x8::splat(token, consts::VERT_MUL_PREV_1);
    let mul_prev_3 = f32x8::splat(token, consts::VERT_MUL_PREV_3);
    let mul_prev_5 = f32x8::splat(token, consts::VERT_MUL_PREV_5);
    let zeroes = f32x8::zero(token);

    let mut x = 0;

    while x + LANES <= width {
        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = (-big_n) + 1;
        while n < height as isize {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_vals = if top >= 0 && (top as usize) < height {
                let idx = top as usize * width + x;
                f32x8::from_array(token, input[idx..][..LANES].try_into().unwrap())
            } else {
                zeroes
            };

            let bottom_vals = if bottom >= 0 && (bottom as usize) < height {
                let idx = bottom as usize * width + x;
                f32x8::from_array(token, input[idx..][..LANES].try_into().unwrap())
            } else {
                zeroes
            };

            let sum = top_vals + bottom_vals;

            let out1 = prev_1.mul_add(mul_prev_1, prev2_1);
            let out3 = prev_3.mul_add(mul_prev_3, prev2_3);
            let out5 = prev_5.mul_add(mul_prev_5, prev2_5);

            let out1 = sum.mul_add(mul_in_1, -out1);
            let out3 = sum.mul_add(mul_in_3, -out3);
            let out5 = sum.mul_add(mul_in_5, -out5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            if n >= 0 {
                let result = out1 + out3 + out5;
                let out_start = n as usize * width + x;
                output[out_start..out_start + LANES].copy_from_slice(&result.to_array());
            }

            n += 1;
        }

        x += LANES;
    }

    // Scalar remainder for leftover columns
    vertical_pass_scalar_columns(input, output, width, height, x);
}

/// Scalar fallback for vertical pass.
fn vertical_pass_inner_scalar(
    _token: archmage::ScalarToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
) {
    vertical_pass_scalar_columns(input, output, width, height, 0);
}

/// Process remaining columns one at a time (used by both v3 remainder and scalar fallback).
fn vertical_pass_scalar_columns(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    start_x: usize,
) {
    let big_n = consts::RADIUS as isize;
    let mut x = start_x;

    while x < width {
        let mut prev_1 = 0.0f32;
        let mut prev_3 = 0.0f32;
        let mut prev_5 = 0.0f32;
        let mut prev2_1 = 0.0f32;
        let mut prev2_3 = 0.0f32;
        let mut prev2_5 = 0.0f32;

        let mut n = (-big_n) + 1;
        while n < height as isize {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_val = if top >= 0 && (top as usize) < height {
                input[top as usize * width + x]
            } else {
                0.0f32
            };

            let bottom_val = if bottom >= 0 && (bottom as usize) < height {
                input[bottom as usize * width + x]
            } else {
                0.0f32
            };

            let sum = top_val + bottom_val;

            let out1 = prev_1.mul_add(consts::VERT_MUL_PREV_1, prev2_1);
            let out3 = prev_3.mul_add(consts::VERT_MUL_PREV_3, prev2_3);
            let out5 = prev_5.mul_add(consts::VERT_MUL_PREV_5, prev2_5);

            let out1 = sum.mul_add(consts::VERT_MUL_IN_1, -out1);
            let out3 = sum.mul_add(consts::VERT_MUL_IN_3, -out3);
            let out5 = sum.mul_add(consts::VERT_MUL_IN_5, -out5);

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

        x += 1;
    }
}
