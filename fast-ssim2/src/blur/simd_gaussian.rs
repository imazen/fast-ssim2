/// SIMD-optimized Recursive Gaussian blur
///
/// Uses archmage/magetypes for cross-platform SIMD via `#[magetypes]` with
/// `GenericF32x8<Token>`. Both passes vectorise the IIR recurrence across the
/// axis it is *not* recurring along: the horizontal pass runs 8 rows per lane
/// group, the vertical pass runs 8 columns per lane group, so the serial IIR
/// dependency lives within a lane while the 8 lanes proceed in parallel.
use archmage::incant;
use archmage::magetypes;
use magetypes::simd::generic::f32x4 as GenericF32x4;
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

/// Extra floats kept in `temp_buffer` so the horizontal pass can write at a
/// deliberately chosen offset — see [`SimdGaussian::temp_offset`].
const TEMP_DEALIAS_SLACK: usize = 1024;

/// How far the temp plane is placed *past* the source plane's position within
/// a 4 KiB page. One cache line would be enough to break the congruence; 256 B
/// keeps 64-byte alignment for the SIMD stores and matches what was measured.
const TEMP_DEALIAS_BYTES: usize = 256;

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
            self.temp_buffer.resize(needed + TEMP_DEALIAS_SLACK, 0.0);
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

    /// Index into `temp_buffer` at which the horizontal pass should write, so
    /// that the temp plane and the source plane are never congruent modulo
    /// 4 KiB.
    ///
    /// Why this exists: the horizontal pass runs the IIR over eight rows at
    /// once, one row per SIMD lane, so each column access is eight loads at
    /// stride `width * 4` bytes. When `width` is a power of two those eight
    /// addresses are congruent mod 4096, and when the destination plane is
    /// congruent with the source as well — which it is whenever both are
    /// page-aligned mappings, i.e. deterministically for planes of a few MB —
    /// the loads, the stores and each other all collide in the same cache set.
    ///
    /// Measured on a Ryzen 9 7900X, horizontal pass alone: 0.70 ns/px at width
    /// 1000, 1032, 2040 and 2056, but **5.34 ns/px at 1024 and 5.40 at 2048** —
    /// a 7.6x cliff at exactly the widths real images use. End-to-end,
    /// `compute_ssimulacra2` on 2048x1024 ran 22.7% slower than on 2040x1024.
    /// Shifting the destination by one 256-byte step removed all of it
    /// (5.34 -> 0.74). On an Apple M4 Pro the same cliff appears at width 4096
    /// (3.89 vs 0.72 ns/px) and disappears the same way.
    ///
    /// This changes *where* the intermediate lives, never what it contains, so
    /// scores are bit-identical. See `benchmarks/blur_stride_2026-09-09.md`.
    fn temp_offset(temp: *const f32, plane: *const f32) -> usize {
        const PAGE: usize = 4096;
        let temp_addr = temp as usize;
        let plane_addr = plane as usize;
        // Target position within the page: the source plane's, plus a step.
        // Dodging the *destination* plane as well was tried twice — against the
        // 8-rows-per-lane-group horizontal pass and again against jpegli's
        // row-contiguous one — and measured no difference either time, so it is
        // not done. The residual at power-of-two widths (measured 2026-09-09
        // with the new kernels: +2.4% at 1024 vs 1032, +7.5% at 4096 vs 4104)
        // lives somewhere else, most likely among the ~24 plane buffers the
        // metric itself allocates in `lib.rs`.
        let want = (plane_addr + TEMP_DEALIAS_BYTES) % PAGE;
        let have = temp_addr % PAGE;
        // Distance forward from `temp` to the next address with that position.
        let delta = (want + PAGE - have) % PAGE;
        // f32 granularity: the buffer is a `Vec<f32>`, so both addresses are
        // 4-byte aligned and `delta` is a whole number of floats.
        debug_assert_eq!(delta % size_of::<f32>(), 0);
        let floats = delta / size_of::<f32>();
        debug_assert!(floats <= TEMP_DEALIAS_SLACK);
        floats.min(TEMP_DEALIAS_SLACK)
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
            self.temp_buffer.resize(size + TEMP_DEALIAS_SLACK, 0.0);
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

        // Horizontal pass: dispatched for FMA. `off` keeps the temp plane out
        // of 4 KiB congruence with `plane` — see `temp_offset`.
        let off = Self::temp_offset(self.temp_buffer.as_ptr(), plane.as_ptr());
        horizontal_pass(plane, &mut self.temp_buffer[off..off + size], width);

        // Vertical pass: SIMD-dispatched, processes all columns per height traversal
        vertical_pass(
            &self.temp_buffer[off..off + size],
            out,
            &mut self.vert_state[..vert_state_needed],
            width,
            height,
        );
    }
}

// ---------------------------------------------------------------------------
// Horizontal pass — jpegli's `FastGaussian1D`, four outputs per iteration
// ---------------------------------------------------------------------------

/// Horizontal recursive Gaussian, in the form the C++ SSIMULACRA2 evaluates.
///
/// The recurrence is serial along a row, so it has to be vectorised across
/// *something*. Through 0.9.0 this crate vectorised across **rows** — eight
/// rows per lane group, one row per lane — which works but makes every column
/// access an eight-wide gather at stride `width * 4` bytes, and that stride is
/// what produced the 4 KiB-aliasing cliff `SimdGaussian::temp_offset` now
/// defends against.
///
/// jpegli vectorises across **columns** instead: four outputs per iteration,
/// from the closed forms for two, three and four recurrence steps (`CPP_MUL_IN`
/// lanes 1..3, generated in `build.rs`). Loads and stores stay contiguous. It
/// is both faster — 12.3–13.7% on NEON, 4–7% on AVX2, measured in
/// `benchmarks/blur_stride_2026-09-09.md` — and bit-identical to the reference,
/// which the row-parallel form was not.
///
/// `HWY_CAPPED(float, 4)` in the C++ pins this to four lanes on every non-scalar
/// target, so widening it past `f32x4` would *break* the parity it buys.
fn horizontal_pass(input: &[f32], output: &mut [f32], width: usize) {
    assert_eq!(input.len(), output.len());

    #[cfg(feature = "rayon")]
    {
        // Rows are independent, so this parallelises cleanly — and unlike the
        // row-vectorised predecessor, every lane of every chunk still walks
        // contiguous memory.
        use rayon::prelude::*;
        input
            .par_chunks_exact(width)
            .zip(output.par_chunks_exact_mut(width))
            .for_each(|(inp, out)| {
                incant!(
                    horizontal_pass_inner(inp, out, width),
                    [v3, neon, wasm128, scalar]
                );
            });
        return;
    }

    #[cfg(not(feature = "rayon"))]
    incant!(
        horizontal_pass_inner(input, output, width),
        [v3, neon, wasm128, scalar]
    )
}

/// One or more whole rows of the horizontal pass. `input` and `output` are
/// `width`-sized row slices (or a whole plane, whose rows are then walked in
/// order).
#[magetypes(v3, neon, wasm128, scalar)]
fn horizontal_pass_inner(token: Token, input: &[f32], output: &mut [f32], width: usize) {
    #[allow(non_camel_case_types)]
    type f32x4 = GenericF32x4<Token>;
    const LANES: isize = 4;

    let big_n = consts::RADIUS as isize;

    // `ShiftLeftLanes<i>(mul_in_k)` for i in 0..4. In the C++ these are produced
    // per iteration by a lane shift; here they are loop-invariant constants, so
    // the shift costs nothing and `magetypes` needs no lane-shift primitive.
    // (`magetypes` 0.9.29 has neither `Broadcast<N>` nor `ShiftLeftLanes<N>` —
    // imazen/archmage#115. Raw `core::arch` lane ops are reachable through
    // `archmage::intrinsics` without `unsafe`, but measured 0.5–1.5% *slower*
    // than this, so the portable form is what ships.)
    let shifted = |base: usize, i: usize| -> f32x4 {
        let mut a = [0.0f32; 4];
        for (j, slot) in a.iter_mut().enumerate() {
            if j >= i {
                *slot = consts::CPP_MUL_IN[base + (j - i)];
            }
        }
        f32x4::from_array(token, a)
    };
    let mul_in: [[f32x4; 4]; 3] =
        core::array::from_fn(|band| core::array::from_fn(|i| shifted(band * 4, i)));
    let mul_prev: [f32x4; 3] = core::array::from_fn(|band| {
        f32x4::from_array(
            token,
            core::array::from_fn(|j| consts::CPP_MUL_PREV[band * 4 + j]),
        )
    });
    let mul_prev2: [f32x4; 3] = core::array::from_fn(|band| {
        f32x4::from_array(
            token,
            core::array::from_fn(|j| consts::CPP_MUL_PREV2[band * 4 + j]),
        )
    });

    for (inp, out) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let at = |i: isize| -> f32 {
            if i >= 0 && (i as usize) < width {
                inp[i as usize]
            } else {
                0.0
            }
        };

        // Scalar prologue and epilogue, exactly as the C++ runs them: scalar
        // until `RoundUpTo(N + 1, 4)`, unrolled through
        // `width - N + 1 - 3`, scalar to the end.
        let (mut p1, mut p3, mut p5) = (0f32, 0f32, 0f32);
        let (mut q1, mut q3, mut q5) = (0f32, 0f32, 0f32);
        let scalar_step = |sum: f32, p: &mut (f32, f32, f32), q: &mut (f32, f32, f32)| -> f32 {
            let mut o1 = sum * consts::MUL_IN_1;
            let mut o3 = sum * consts::MUL_IN_3;
            let mut o5 = sum * consts::MUL_IN_5;
            o1 = consts::MUL_PREV2_1.mul_add(q.0, o1);
            o3 = consts::MUL_PREV2_3.mul_add(q.1, o3);
            o5 = consts::MUL_PREV2_5.mul_add(q.2, o5);
            *q = *p;
            o1 = consts::MUL_PREV_1.mul_add(p.0, o1);
            o3 = consts::MUL_PREV_3.mul_add(p.1, o3);
            o5 = consts::MUL_PREV_5.mul_add(p.2, o5);
            *p = (o1, o3, o5);
            o1 + o3 + o5
        };

        let mut n = -big_n + 1;
        let first_aligned = ((big_n + 1) + LANES - 1) / LANES * LANES;
        while n < first_aligned.min(width as isize) {
            let sum = at(n - big_n - 1) + at(n + big_n - 1);
            let (mut pv, mut qv) = ((p1, p3, p5), (q1, q3, q5));
            let o = scalar_step(sum, &mut pv, &mut qv);
            (p1, p3, p5) = pv;
            (q1, q3, q5) = qv;
            if n >= 0 {
                out[n as usize] = o;
            }
            n += 1;
        }

        let mut prev = [
            f32x4::splat(token, p1),
            f32x4::splat(token, p3),
            f32x4::splat(token, p5),
        ];
        let mut prev2 = [
            f32x4::splat(token, q1),
            f32x4::splat(token, q3),
            f32x4::splat(token, q5),
        ];

        while n < width as isize - big_n + 1 - (LANES - 1) {
            let base_l = (n - big_n - 1) as usize;
            let base_r = (n + big_n - 1) as usize;
            // The C++ loads one vector of sums and broadcasts each lane; with
            // no `Broadcast<N>` available, splat each sum from a scalar load —
            // same values, and it measured no slower.
            let s: [f32x4; 4] =
                core::array::from_fn(|k| f32x4::splat(token, inp[base_l + k] + inp[base_r + k]));

            let mut o = [f32x4::zero(token); 3];
            for band in 0..3 {
                let mut acc = s[0] * mul_in[band][0];
                acc = mul_in[band][1].mul_add(s[1], acc);
                acc = mul_in[band][2].mul_add(s[2], acc);
                acc = mul_in[band][3].mul_add(s[3], acc);
                acc = mul_prev2[band].mul_add(prev2[band], acc);
                acc = mul_prev[band].mul_add(prev[band], acc);
                o[band] = acc;
            }

            // `Broadcast<LANES-2>` / `<LANES-1>`: the recurrence state for the
            // next iteration is the third and fourth output of this one.
            for band in 0..3 {
                let a = o[band].to_array();
                prev2[band] = f32x4::splat(token, a[2]);
                prev[band] = f32x4::splat(token, a[3]);
            }

            let result = (o[0] + o[1] + o[2]).to_array();
            out[n as usize..][..4].copy_from_slice(&result);
            n += LANES;
        }

        // Back to scalar state for the tail.
        p1 = prev[0].to_array()[0];
        p3 = prev[1].to_array()[0];
        p5 = prev[2].to_array()[0];
        q1 = prev2[0].to_array()[0];
        q3 = prev2[1].to_array()[0];
        q5 = prev2[2].to_array()[0];

        while n < width as isize {
            let sum = at(n - big_n - 1) + at(n + big_n - 1);
            let (mut pv, mut qv) = ((p1, p3, p5), (q1, q3, q5));
            let o = scalar_step(sum, &mut pv, &mut qv);
            (p1, p3, p5) = pv;
            (q1, q3, q5) = qv;
            if n >= 0 {
                out[n as usize] = o;
            }
            n += 1;
        }
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

            // Fused, matching the C++ reference's `MulAdd`/`NegMulSub` pair.
            // See the horizontal pass for why this is not unfused for the sake
            // of the non-FMA `magetypes` arms.
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

            // Fused, matching the vectorised body above.
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
        // The buffer carries `TEMP_DEALIAS_SLACK` floats beyond the plane so
        // `temp_offset` can place the temp plane out of 4 KiB congruence with
        // the source. Still an exact assertion, just of the new intended size.
        assert_eq!(g.temp_buffer.len(), 64 * 64 + TEMP_DEALIAS_SLACK);
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
