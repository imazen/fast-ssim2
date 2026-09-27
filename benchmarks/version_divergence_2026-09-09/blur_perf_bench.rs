//! Is jpegli's 4-unrolled horizontal Gaussian faster than the 8-rows-per-lane
//! -group pass fast-ssim2 ships?
//!
//! The two vectorise the *same* serial recurrence along opposite axes:
//!
//! * **ours** — one row per SIMD lane, 8 rows at a time. The recurrence stays
//!   scalar-shaped but 8 independent copies run in lockstep. Cost: every column
//!   access is a manual 8-wide gather (8 loads at stride `width`) and an 8-wide
//!   scatter, so it never touches contiguous memory.
//! * **jpegli** — four *columns* at a time within one row, from the closed forms
//!   for 2/3/4 recurrence steps (`CPP_MUL_IN` lanes 1..3). Loads and stores are
//!   contiguous, but each iteration needs a broadcast per input and the serial
//!   state comes out of lanes 2 and 3 of the previous result.
//!
//! `magetypes` 0.9.29 has no `Broadcast<N>` / `ShiftLeftLanes<N>`, so the port
//! dodges both: the four `ShiftLeftLanes<i>(mul_in_k)` vectors are compile-time
//! constants built once outside the loop, and each input sum is splatted from a
//! scalar load rather than broadcast out of a vector.
//!
//! Run: `cargo bench -p fast-ssim2 --bench blur_perf`

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::generic::f32x4 as GenericF32x4;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

mod consts {
    #![allow(clippy::unreadable_literal)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

const VERT_STATE_LANES: usize = 8;

// ---------------------------------------------------------------------------
// reference: the scalar recurrence fast-ssim2 ships (also its remainder rows)
// ---------------------------------------------------------------------------

#[inline(always)]
fn horizontal_row(input: &[f32], output: &mut [f32], width: usize) {
    let big_n = consts::RADIUS as isize;
    let (mut prev_1, mut prev_3, mut prev_5) = (0f32, 0f32, 0f32);
    let (mut prev2_1, mut prev2_3, mut prev2_5) = (0f32, 0f32, 0f32);

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

/// jpegli's `FastGaussian1D`, scalar transliteration — the value the vectorised
/// port below must reproduce bit-for-bit (same code as the repo's
/// `cpp_parity_diag::HorizKind::CppUnrolled4`).
fn horizontal_row_cpp_scalar(input: &[f32], output: &mut [f32], width: usize) {
    let big_n = consts::RADIUS as isize;
    let at = |i: isize| -> f32 {
        if i >= 0 && (i as usize) < width {
            input[i as usize]
        } else {
            0.0
        }
    };
    let (mut p1, mut p3, mut p5) = (0f32, 0f32, 0f32);
    let (mut q1, mut q3, mut q5) = (0f32, 0f32, 0f32);
    let step = |sum: f32, p: &mut (f32, f32, f32), q: &mut (f32, f32, f32)| -> f32 {
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
    const LANES: isize = 4;
    let mut n = -big_n + 1;
    let first_aligned = ((big_n + 1) + LANES - 1) / LANES * LANES;
    while n < first_aligned.min(width as isize) {
        let sum = at(n - big_n - 1) + at(n + big_n - 1);
        let (mut pv, mut qv) = ((p1, p3, p5), (q1, q3, q5));
        let o = step(sum, &mut pv, &mut qv);
        (p1, p3, p5) = pv;
        (q1, q3, q5) = qv;
        if n >= 0 {
            output[n as usize] = o;
        }
        n += 1;
    }
    while n < width as isize - big_n + 1 - (LANES - 1) {
        let s: [f32; 4] = core::array::from_fn(|k| {
            let m = n + k as isize;
            at(m - big_n - 1) + at(m + big_n - 1)
        });
        let mut o = [[0f32; 4]; 3];
        for (band, base) in [(0usize, 0usize), (1, 4), (2, 8)] {
            let (prev, prev2) = match band {
                0 => (p1, q1),
                1 => (p3, q3),
                _ => (p5, q5),
            };
            for (lane, slot) in o[band].iter_mut().enumerate() {
                let mut acc = 0f32;
                for (i, si) in s.iter().enumerate().take(lane + 1) {
                    acc = consts::CPP_MUL_IN[base + (lane - i)].mul_add(*si, acc);
                }
                acc = consts::CPP_MUL_PREV2[base + lane].mul_add(prev2, acc);
                acc = consts::CPP_MUL_PREV[base + lane].mul_add(prev, acc);
                *slot = acc;
            }
        }
        q1 = o[0][2];
        q3 = o[1][2];
        q5 = o[2][2];
        p1 = o[0][3];
        p3 = o[1][3];
        p5 = o[2][3];
        for lane in 0..4usize {
            output[(n + lane as isize) as usize] = o[0][lane] + o[1][lane] + o[2][lane];
        }
        n += LANES;
    }
    while n < width as isize {
        let sum = at(n - big_n - 1) + at(n + big_n - 1);
        let (mut pv, mut qv) = ((p1, p3, p5), (q1, q3, q5));
        let o = step(sum, &mut pv, &mut qv);
        (p1, p3, p5) = pv;
        (q1, q3, q5) = qv;
        if n >= 0 {
            output[n as usize] = o;
        }
        n += 1;
    }
}

// ---------------------------------------------------------------------------
// kernel A: what fast-ssim2 ships — 8 rows per lane group
// ---------------------------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
fn horiz_rows8_inner(token: Token, input: &[f32], output: &mut [f32], width: usize, row_limit: usize) {
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
    while row_base + LANES <= row_limit {
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

fn horiz_rows8(input: &[f32], output: &mut [f32], width: usize) {
    let height = input.len() / width;
    let groups = height / VERT_STATE_LANES;
    if groups > 0 {
        incant!(
            horiz_rows8_inner(input, output, width, groups * VERT_STATE_LANES),
            [v3, neon, wasm128, scalar]
        );
    }
    let start = groups * VERT_STATE_LANES * width;
    if start < input.len() {
        input[start..]
            .chunks_exact(width)
            .zip(output[start..].chunks_exact_mut(width))
            .for_each(|(inp, out)| horizontal_row(inp, out, width));
    }
}

// ---------------------------------------------------------------------------
// kernel B: jpegli's 4-unrolled pass, vectorised without Broadcast/ShiftLanes
// ---------------------------------------------------------------------------

#[magetypes(v3, neon, wasm128, scalar)]
fn horiz_unrolled4_inner(token: Token, input: &[f32], output: &mut [f32], width: usize) {
    #[allow(non_camel_case_types)]
    type f32x4 = GenericF32x4<Token>;
    const LANES: isize = 4;

    let big_n = consts::RADIUS as isize;

    // `ShiftLeftLanes<i>(mul_in_k)` for i in 0..4, precomputed: lane j holds
    // `CPP_MUL_IN[base + j - i]` when `j >= i`, else 0. Constant across the
    // whole pass, so the shift the C++ does per iteration costs nothing here.
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

    let height = input.len() / width;
    for row in 0..height {
        let inp = &input[row * width..][..width];
        let out = &mut output[row * width..][..width];

        let at = |i: isize| -> f32 {
            if i >= 0 && (i as usize) < width {
                inp[i as usize]
            } else {
                0.0
            }
        };
        // Scalar prologue / epilogue, exactly as the C++ does.
        let (mut p1, mut p3, mut p5) = (0f32, 0f32, 0f32);
        let (mut q1, mut q3, mut q5) = (0f32, 0f32, 0f32);
        let mut scalar_step = |sum: f32,
                               p: &mut (f32, f32, f32),
                               q: &mut (f32, f32, f32)|
         -> f32 {
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

        // Vector state: prev / prev2 broadcast across all four lanes.
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
            // Four input sums, splatted from scalar loads (the C++ loads one
            // vector and broadcasts each lane; magetypes has no Broadcast<N>).
            let s: [f32x4; 4] = core::array::from_fn(|k| {
                f32x4::splat(token, inp[base_l + k] + inp[base_r + k])
            });

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
        let pa: [[f32; 4]; 3] = core::array::from_fn(|b| prev[b].to_array());
        let qa: [[f32; 4]; 3] = core::array::from_fn(|b| prev2[b].to_array());
        p1 = pa[0][0];
        p3 = pa[1][0];
        p5 = pa[2][0];
        q1 = qa[0][0];
        q3 = qa[1][0];
        q5 = qa[2][0];

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

fn horiz_unrolled4(input: &[f32], output: &mut [f32], width: usize) {
    incant!(
        horiz_unrolled4_inner(input, output, width),
        [v3, neon, wasm128, scalar]
    )
}

fn horiz_scalar_rows(input: &[f32], output: &mut [f32], width: usize) {
    input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
        .for_each(|(inp, out)| horizontal_row(inp, out, width));
}

fn make_plane(width: usize, height: usize) -> Vec<f32> {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..width * height)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32) / ((1u32 << 24) as f32)
        })
        .collect()
}

fn agreement() {
    let (w, h) = (301usize, 37usize); // deliberately not a multiple of 4 or 8
    let plane = make_plane(w, h);
    let mut a = vec![0f32; w * h];
    let mut b = vec![0f32; w * h];
    let mut c = vec![0f32; w * h];
    horiz_rows8(&plane, &mut a, w);
    horiz_unrolled4(&plane, &mut b, w);
    for (inp, out) in plane.chunks_exact(w).zip(c.chunks_exact_mut(w)) {
        horizontal_row_cpp_scalar(inp, out, w);
    }
    let maxd = |u: &[f32], v: &[f32]| {
        u.iter()
            .zip(v)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
            .fold(0.0f64, f64::max)
    };
    let exact = b.iter().zip(&c).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
    println!(
        "horizontal blur, {w}x{h}: |rows8 - unrolled4| {:.3e}   |unrolled4 - cpp_scalar| {:.3e}  ({}/{} bit-exact)",
        maxd(&a, &b),
        maxd(&b, &c),
        exact,
        b.len()
    );
}

fn bench(c: &mut Criterion) {
    agreement();
    // Stride probe: the shipped kernel gathers 8 rows at stride `width * 4`
    // bytes. Power-of-two widths make every lane of a gather land in the same
    // cache set. Widths chosen to straddle 1024 / 2048 / 4096.
    for (w, h) in [
        (248usize, 512usize),
        (256, 512),
        (264, 512),
        (504, 512),
        (512, 512),
        (520, 512),
        (1000, 512),
        (1024, 512),
        (1032, 512),
        (2040, 512),
        (2048, 512),
        (2056, 512),
        (4088, 256),
        (4096, 256),
        (4104, 256),
    ] {
        let plane = make_plane(w, h);
        let mut out = vec![0f32; w * h];
        // Same run, but with the destination shifted 64 floats (256 B) inside a
        // larger allocation: if a cliff is 4K-aliasing between the two planes
        // rather than the kernel's own access pattern, it moves or vanishes.
        let mut out_off = vec![0f32; w * h + 64];
        let mut group = c.benchmark_group(format!("stride_{w}x{h}"));
        group.bench_function("rows8_shipped", |bencher| {
            bencher.iter(|| {
                horiz_rows8(&plane, &mut out, w);
                std::hint::black_box(&out);
            })
        });
        group.bench_function("jpegli_unrolled4", |bencher| {
            bencher.iter(|| {
                horiz_unrolled4(&plane, &mut out, w);
                std::hint::black_box(&out);
            })
        });
        group.bench_function("rows8_shipped_offset", |bencher| {
            bencher.iter(|| {
                horiz_rows8(&plane, &mut out_off[64..], w);
                std::hint::black_box(&out_off);
            })
        });
        group.bench_function("jpegli_unrolled4_offset", |bencher| {
            bencher.iter(|| {
                horiz_unrolled4(&plane, &mut out_off[64..], w);
                std::hint::black_box(&out_off);
            })
        });
        group.finish();
    }

    for (w, h) in [(320usize, 240usize), (1024, 768), (1920, 1080)] {
        let plane = make_plane(w, h);
        let mut out = vec![0f32; w * h];
        let mut group = c.benchmark_group(format!("horiz_{w}x{h}"));
        group.bench_function("rows8_shipped", |bencher| {
            bencher.iter(|| {
                horiz_rows8(&plane, &mut out, w);
                std::hint::black_box(&out);
            })
        });
        group.bench_function("jpegli_unrolled4", |bencher| {
            bencher.iter(|| {
                horiz_unrolled4(&plane, &mut out, w);
                std::hint::black_box(&out);
            })
        });
        group.bench_function("scalar_rows", |bencher| {
            bencher.iter(|| {
                horiz_scalar_rows(&plane, &mut out, w);
                std::hint::black_box(&out);
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
