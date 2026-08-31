#!/usr/bin/env python3
"""cppmax = jpegli cbrt + fused opsin matmul + jpegli's 4-unrolled horizontal blur."""
import pathlib, subprocess, sys
ROOT = pathlib.Path.home()/"tmp/ssim2-parity-study/trees/cppmax"

# reuse the cbrt + matmul patches
sys.argv = [sys.argv[0]]
src = (pathlib.Path.home()/"tmp/ssim2-parity-study/patch_variants.py").read_text()
src = src.replace('VARIANTS = {\n    "v071repro":         ("f64", True),\n    "cbrt64":            ("f64", False),\n    "cppcbrt":           ("jpegli", True),\n    "cppcbrt_unfusedmm": ("jpegli", False),\n}', 'VARIANTS = {"cppmax": ("jpegli", True)}')
exec(compile(src, "patch_variants_reused", "exec"))

# --- now swap the horizontal blur for jpegli's 4-unrolled form ---
ROOT = pathlib.Path.home()/"tmp/ssim2-parity-study/trees/cppmax"
p = ROOT/"fast-ssim2/src/blur/simd_gaussian.rs"
t = p.read_text()

OLD_BODY = """    let groups = height / VERT_STATE_LANES;
    if groups > 0 {
        horizontal_pass_simd(input, output, width, groups * VERT_STATE_LANES);
    }
    // Scalar remainder rows (height not a multiple of 8).
    horizontal_pass_rows(input, output, width, groups * VERT_STATE_LANES);"""
NEW_BODY = """    // STUDY VARIANT: every row through jpegli's 4-unrolled FastGaussian1D form.
    let _ = height;
    for (inp, out) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        horizontal_row_cpp_unrolled4(inp, out, width);
    }
    if false {
        horizontal_pass_simd(input, output, width, 0);
        horizontal_pass_rows(input, output, width, 0);
    }"""
assert t.count(OLD_BODY) == 1
t = t.replace(OLD_BODY, NEW_BODY)

CPP_ROW = '''
/// jpegli `FastGaussian1D` as built for any non-scalar Highway target: scalar
/// until `RoundUpTo(N + 1, 4)`, then four outputs per iteration from the
/// f32-rounded 2nd/3rd/4th-power coefficients, then a scalar tail.
/// Transliterated from `~/work/jpegli/lib/base/gauss_blur.cc` (same code the
/// repo's `cpp_parity_diag::HorizKind::CppUnrolled4` carries).
fn horizontal_row_cpp_unrolled4(input: &[f32], output: &mut [f32], width: usize) {
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
    let mut n = -big_n + 1;
    const LANES: isize = 4;
    let first_aligned = ((big_n + 1) + LANES - 1) / LANES * LANES;
    while n < first_aligned.min(width as isize) {
        let sum = at(n - big_n - 1) + at(n + big_n - 1);
        let mut pv = (p1, p3, p5);
        let mut qv = (q1, q3, q5);
        let o = step(sum, &mut pv, &mut qv);
        (p1, p3, p5) = pv;
        (q1, q3, q5) = qv;
        if n >= 0 {
            output[n as usize] = o;
        }
        n += 1;
    }
    while n < width as isize - big_n + 1 - (LANES - 1) {
        let s: [f32; 4] = std::array::from_fn(|k| {
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
        let mut pv = (p1, p3, p5);
        let mut qv = (q1, q3, q5);
        let o = step(sum, &mut pv, &mut qv);
        (p1, p3, p5) = pv;
        (q1, q3, q5) = qv;
        if n >= 0 {
            output[n as usize] = o;
        }
        n += 1;
    }
}
'''
t = t + CPP_ROW
p.write_text(t)
print("patched cppmax blur")
