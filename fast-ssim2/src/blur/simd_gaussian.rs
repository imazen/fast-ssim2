/// SIMD-optimized Recursive Gaussian blur
///
/// Uses archmage/magetypes for cross-platform SIMD via `#[magetypes]` with
/// `GenericF32x8<Token>`. Both passes vectorise the IIR recurrence across the
/// axis it is *not* recurring along: the horizontal pass runs 8 rows per lane
/// group, the vertical pass runs 8 columns per lane group, so the serial IIR
/// dependency lives within a lane while the 8 lanes proceed in parallel.
use archmage::incant;
use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;

mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

pub struct SimdGaussian {
    temp_buffer: Vec<f32>,
    max_size: usize,
    /// IIR state for vertical pass: 6 stacked sub-slices of `groups * LANES`
    /// floats each (prev_1, prev_3, prev_5, prev2_1, prev2_3, prev2_5).
    ///
    /// Hoisted out of the per-call SIMD inner function so the 6 allocations
    /// no longer happen on every plane blur. With ssim2's 5 blurs per scale
    /// across 6 scales, that's ~180 small allocations per frame eliminated.
    /// The state is zeroed at the start of every blur because the IIR
    /// initializes to zero — we don't preserve state between calls.
    vert_state: Vec<f32>,
    vert_state_size: usize,
}

const VERT_STATE_LANES: usize = 8;

impl SimdGaussian {
    /// Create a new SIMD Gaussian blur context.
    ///
    /// `max_width` is treated as a hint; the temporary buffer grows on demand
    /// in [`Self::shrink_to`] and [`Self::blur_single_plane_into`], so an
    /// underestimate only costs one reallocation. The hint is intentionally
    /// not multiplied by an assumed maximum height: previously this constructor
    /// pre-allocated `max_width * 4096` floats unconditionally, which both
    /// wasted memory for short strips (e.g. a 16384-wide image with a 64-row
    /// working buffer would allocate 256 MiB upfront for nothing) and would
    /// silently overflow `usize` on 32-bit targets when `max_width` exceeded
    /// `usize::MAX / 4096`.
    pub fn new(max_width: usize) -> Self {
        // Cap the hint at a sane value so callers passing absurd widths
        // don't trigger an immediate gigabyte-scale allocation. The buffer
        // still grows on demand if the actual image needs more.
        let initial_capacity = max_width.min(usize::MAX / 4);
        Self {
            temp_buffer: Vec::with_capacity(initial_capacity),
            max_size: 0,
            vert_state: Vec::new(),
            vert_state_size: 0,
        }
    }

    /// Ensure the temporary buffer is large enough for `width * height`.
    ///
    /// Returns silently without resizing if the dimensions overflow `usize` or
    /// fit in the existing capacity. The actual blur entry point
    /// ([`Self::blur_single_plane_into`]) re-checks and panics with a clearer
    /// message on overflow, matching the previous (implicit) behavior on
    /// 64-bit targets but making the failure mode explicit on 32-bit.
    pub fn shrink_to(&mut self, width: usize, height: usize) {
        let Some(needed) = width.checked_mul(height) else {
            return;
        };
        if needed > self.max_size {
            self.temp_buffer.resize(needed, 0.0);
            self.max_size = needed;
        }
        // 6 IIR state arrays of `(width / 8) * 8` floats each.
        let groups = width / VERT_STATE_LANES;
        let vert_state_needed = 6usize.checked_mul(groups.saturating_mul(VERT_STATE_LANES));
        if let Some(n) = vert_state_needed
            && n > self.vert_state_size
        {
            self.vert_state.resize(n, 0.0);
            self.vert_state_size = n;
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
        // checked_mul guards against silent wraparound on 32-bit targets where
        // a malicious caller could otherwise pass dims whose product overflows.
        let size = width
            .checked_mul(height)
            .expect("SimdGaussian: width * height overflows usize");
        if size > self.max_size {
            self.temp_buffer.resize(size, 0.0);
            self.max_size = size;
        }
        let groups = width / VERT_STATE_LANES;
        let vert_state_needed = 6 * groups * VERT_STATE_LANES;
        if vert_state_needed > self.vert_state_size {
            self.vert_state.resize(vert_state_needed, 0.0);
            self.vert_state_size = vert_state_needed;
        }
        // IIR initialises state to zero on every call.
        self.vert_state[..vert_state_needed].fill(0.0);

        // Horizontal pass: dispatched for FMA
        horizontal_pass(plane, &mut self.temp_buffer[..size], width);

        // Vertical pass: SIMD-dispatched, processes all columns per height traversal
        vertical_pass(
            &self.temp_buffer[..size],
            out,
            &mut self.vert_state[..vert_state_needed],
            width,
            height,
        );
    }
}

// ---------------------------------------------------------------------------
// Horizontal pass — recursive Gaussian IIR, SIMD-vectorised across rows
// (8 rows per lane group) with a scalar remainder for leftover rows.
// ---------------------------------------------------------------------------

fn horizontal_pass(input: &[f32], output: &mut [f32], width: usize) {
    assert_eq!(input.len(), output.len());
    let height = input.len() / width;
    // SIMD path processes 8 rows in parallel (one row per lane). The horizontal
    // IIR is a serial recurrence *within* a row, so it can't vectorise across a
    // row — but it's fully independent *across* rows, so we run 8 rows at once
    // with the IIR state held in 8-lane vectors. This is the across-the-other-
    // axis trick the vertical pass already uses for columns. On Neoverse-N1 the
    // scalar horizontal IIR was ~50% of the whole blur (and blur was ~40% of the
    // SSIMULACRA2 pipeline) because each row's recurrence serialised; lane-per-
    // row recovers the parallelism the vertical pass already had.
    let groups = height / VERT_STATE_LANES;
    if groups > 0 {
        horizontal_pass_simd(input, output, width, groups * VERT_STATE_LANES);
    }
    // Scalar remainder rows (height not a multiple of 8).
    horizontal_pass_rows(input, output, width, groups * VERT_STATE_LANES);
}

/// SIMD horizontal pass dispatcher — processes 8 rows per lane group.
fn horizontal_pass_simd(input: &[f32], output: &mut [f32], width: usize, row_limit: usize) {
    incant!(
        horizontal_pass_simd_inner(input, output, width, row_limit),
        [v3, neon, wasm128, scalar]
    )
}

/// Generic row-parallel horizontal pass — 8 rows at a time, one row per lane.
///
/// Each column position `n` is loaded from 8 consecutive rows into a vector
/// (a manual 8-wide gather: 8 scalar loads at stride `width`), the recursive
/// Gaussian IIR is advanced with vector state, and the result column is stored
/// back across the 8 rows. The bounds branches on `left`/`right`/`n` are the
/// same for every lane in a group (they depend only on the column index), so
/// the inner loop stays branch-light.
#[magetypes(v3, neon, wasm128, scalar)]
fn horizontal_pass_simd_inner(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    row_limit: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let big_n = consts::RADIUS as isize;
    let mul_in_1 = f32x8::splat(token, consts::MUL_IN_1);
    let mul_in_3 = f32x8::splat(token, consts::MUL_IN_3);
    let mul_in_5 = f32x8::splat(token, consts::MUL_IN_5);
    let mul_prev_1 = f32x8::splat(token, consts::MUL_PREV_1);
    let mul_prev_3 = f32x8::splat(token, consts::MUL_PREV_3);
    let mul_prev_5 = f32x8::splat(token, consts::MUL_PREV_5);
    let zero = f32x8::zero(token);

    let mut row_base = 0usize;
    while row_base < row_limit {
        // Manual gather/scatter helpers for the LANES rows starting at row_base.
        let gather = |col: usize| -> f32x8 {
            let mut a = [0.0f32; LANES];
            for (lane, slot) in a.iter_mut().enumerate() {
                *slot = input[(row_base + lane) * width + col];
            }
            f32x8::from_array(token, a)
        };

        let mut prev_1 = zero;
        let mut prev_3 = zero;
        let mut prev_5 = zero;
        let mut prev2_1 = zero;
        let mut prev2_3 = zero;
        let mut prev2_5 = zero;

        let mut n = (-big_n) + 1;
        while n < width as isize {
            let left = n - big_n - 1;
            let right = n + big_n - 1;
            let left_val = if left >= 0 && (left as usize) < width {
                gather(left as usize)
            } else {
                zero
            };
            let right_val = if right >= 0 && (right as usize) < width {
                gather(right as usize)
            } else {
                zero
            };
            let sum = left_val + right_val;

            // Mirror the scalar `horizontal_row` op order EXACTLY so the SIMD
            // path's f32 rounding matches it bit-for-bit (MUL_PREV2_k == -1):
            //   out_k  = sum * MUL_IN_k                 (rounded product)
            //   out_k  = -prev2_k + out_k               (the MUL_PREV2 step)
            //   out_k  = MUL_PREV_k * prev_k + out_k    (the MUL_PREV step)
            // Fusing the first two into one mul_add would change rounding.
            let p1 = sum * mul_in_1;
            let p3 = sum * mul_in_3;
            let p5 = sum * mul_in_5;
            let out_1 = mul_prev_1.mul_add(prev_1, p1 - prev2_1);
            let out_3 = mul_prev_3.mul_add(prev_3, p3 - prev2_3);
            let out_5 = mul_prev_5.mul_add(prev_5, p5 - prev2_5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if n >= 0 && (n as usize) < width {
                let result = (out_1 + out_3 + out_5).to_array();
                let col = n as usize;
                for (lane, &v) in result.iter().enumerate() {
                    output[(row_base + lane) * width + col] = v;
                }
            }

            n += 1;
        }

        row_base += LANES;
    }
}

/// Scalar horizontal pass over rows `[start_row, height)`.
#[inline(always)]
fn horizontal_pass_rows(input: &[f32], output: &mut [f32], width: usize, start_row: usize) {
    let start = start_row * width;
    if start >= input.len() {
        return;
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        input[start..]
            .par_chunks_exact(width)
            .zip(output[start..].par_chunks_exact_mut(width))
            .for_each(|(inp, out)| horizontal_row(inp, out, width));
    }

    #[cfg(not(feature = "rayon"))]
    {
        input[start..]
            .chunks_exact(width)
            .zip(output[start..].chunks_exact_mut(width))
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
// Vertical pass — SIMD IIR filter processing all columns per height traversal
// ---------------------------------------------------------------------------

fn vertical_pass(
    input: &[f32],
    output: &mut [f32],
    state: &mut [f32],
    width: usize,
    height: usize,
) {
    assert_eq!(input.len(), output.len());
    incant!(
        vertical_pass_inner(input, output, state, width, height),
        [v3, neon, wasm128, scalar]
    )
}

/// Generic vertical pass — processes 8 columns at a time on all platforms.
///
/// Uses flat f32 state arrays so all column groups are processed per row,
/// avoiding repeated height traversals (which kills cache performance).
///
/// `state` is a caller-supplied buffer of length `6 * (width / LANES) * LANES`,
/// zeroed before the call. We split it into six sub-slices to back the IIR
/// state vectors (prev_1, prev_3, prev_5, prev2_1, prev2_3, prev2_5) — owned
/// by `SimdGaussian` so we don't reallocate them on every blur call.
#[magetypes(v3, neon, wasm128, scalar)]
fn vertical_pass_inner(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    state: &mut [f32],
    width: usize,
    height: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let big_n = consts::RADIUS as isize;
    let groups = width / LANES;

    // SIMD constants
    let mul_in_1 = f32x8::splat(token, consts::VERT_MUL_IN_1);
    let mul_in_3 = f32x8::splat(token, consts::VERT_MUL_IN_3);
    let mul_in_5 = f32x8::splat(token, consts::VERT_MUL_IN_5);
    let mul_prev_1 = f32x8::splat(token, consts::VERT_MUL_PREV_1);
    let mul_prev_3 = f32x8::splat(token, consts::VERT_MUL_PREV_3);
    let mul_prev_5 = f32x8::splat(token, consts::VERT_MUL_PREV_5);
    let zeroes = f32x8::zero(token);

    // State arrays: 6 IIR state variables x (groups x LANES) floats each.
    // Caller pre-zeroed and pre-sized — split the flat buffer in place.
    let state_size = groups * LANES;
    let (prev_1, rest) = state.split_at_mut(state_size);
    let (prev_3, rest) = rest.split_at_mut(state_size);
    let (prev_5, rest) = rest.split_at_mut(state_size);
    let (prev2_1, rest) = rest.split_at_mut(state_size);
    let (prev2_3, rest) = rest.split_at_mut(state_size);
    let (prev2_5, _) = rest.split_at_mut(state_size);

    let mut n = (-big_n) + 1;
    while n < height as isize {
        let top = n - big_n - 1;
        let bottom = n + big_n - 1;

        let top_valid = top >= 0 && (top as usize) < height;
        let bottom_valid = bottom >= 0 && (bottom as usize) < height;
        let top_row_start = if top_valid { top as usize * width } else { 0 };
        let bottom_row_start = if bottom_valid {
            bottom as usize * width
        } else {
            0
        };

        for g in 0..groups {
            let col = g * LANES;

            let top_vals = if top_valid {
                let idx = top_row_start + col;
                f32x8::from_array(token, input[idx..][..LANES].try_into().unwrap())
            } else {
                zeroes
            };

            let bottom_vals = if bottom_valid {
                let idx = bottom_row_start + col;
                f32x8::from_array(token, input[idx..][..LANES].try_into().unwrap())
            } else {
                zeroes
            };

            let sum = top_vals + bottom_vals;

            let p1 = f32x8::from_array(token, prev_1[col..][..LANES].try_into().unwrap());
            let p3 = f32x8::from_array(token, prev_3[col..][..LANES].try_into().unwrap());
            let p5 = f32x8::from_array(token, prev_5[col..][..LANES].try_into().unwrap());
            let p21 = f32x8::from_array(token, prev2_1[col..][..LANES].try_into().unwrap());
            let p23 = f32x8::from_array(token, prev2_3[col..][..LANES].try_into().unwrap());
            let p25 = f32x8::from_array(token, prev2_5[col..][..LANES].try_into().unwrap());

            let out1 = p1.mul_add(mul_prev_1, p21);
            let out3 = p3.mul_add(mul_prev_3, p23);
            let out5 = p5.mul_add(mul_prev_5, p25);

            let out1 = sum.mul_add(mul_in_1, -out1);
            let out3 = sum.mul_add(mul_in_3, -out3);
            let out5 = sum.mul_add(mul_in_5, -out5);

            // Update state: prev2 = prev, prev = out
            prev2_1[col..col + LANES].copy_from_slice(&p1.to_array());
            prev2_3[col..col + LANES].copy_from_slice(&p3.to_array());
            prev2_5[col..col + LANES].copy_from_slice(&p5.to_array());
            prev_1[col..col + LANES].copy_from_slice(&out1.to_array());
            prev_3[col..col + LANES].copy_from_slice(&out3.to_array());
            prev_5[col..col + LANES].copy_from_slice(&out5.to_array());

            if n >= 0 {
                let result = out1 + out3 + out5;
                let out_start = n as usize * width + col;
                output[out_start..out_start + LANES].copy_from_slice(&result.to_array());
            }
        }

        n += 1;
    }

    // Scalar remainder for leftover columns
    vertical_pass_scalar_columns(input, output, width, height, groups * LANES);
}

/// Process remaining columns one at a time (used by both SIMD remainder and scalar fallback).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_does_not_eagerly_allocate_height_hint() {
        // Previously `SimdGaussian::new(max_width)` allocated
        // `max_width * 4096` floats unconditionally. A 1024-wide hint should
        // not commit 16 MiB upfront -- the buffer grows lazily when blur is
        // actually invoked.
        let g = SimdGaussian::new(1024);
        assert_eq!(g.max_size, 0);
        // Capacity may be reserved up to the hint, but len stays at 0 so we
        // pay no time touching uninitialised pages.
        assert_eq!(g.temp_buffer.len(), 0);
    }

    #[test]
    fn shrink_to_ignores_overflowing_dims() {
        // Hostile caller passes dims whose product overflows usize. We must
        // not panic in `shrink_to`; the actual blur path is the place to
        // refuse the work.
        let mut g = SimdGaussian::new(0);
        g.shrink_to(usize::MAX, 2);
        assert_eq!(g.max_size, 0);
    }

    #[test]
    fn shrink_to_grows_on_demand() {
        let mut g = SimdGaussian::new(0);
        g.shrink_to(64, 64);
        assert!(g.max_size >= 64 * 64);
        assert_eq!(g.temp_buffer.len(), 64 * 64);
    }

    #[test]
    fn blur_runs_after_lazy_construction() {
        // End-to-end: a context constructed with hint=0 must still service a
        // small blur call by growing its buffer in blur_single_plane_into.
        let mut g = SimdGaussian::new(0);
        let plane = vec![0.5f32; 16 * 16];
        let mut out = vec![0.0f32; 16 * 16];
        g.blur_single_plane_into(&plane, &mut out, 16, 16);
        // Output is finite (the recursive Gaussian preserves a constant
        // signal up to scaling at small sizes; we only assert non-NaN here).
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "width * height overflows usize")]
    fn blur_panics_on_overflowing_dims() {
        let mut g = SimdGaussian::new(0);
        let plane = [0.0f32; 0];
        let mut out = [0.0f32; 0];
        g.blur_single_plane_into(&plane, &mut out, usize::MAX, 2);
    }
}
