#!/usr/bin/env python3
"""Patch extracted fast-ssim2 HEAD trees into cbrt / matmul-fusion variants.

Variants:
  v071repro         fused opsin matmul + f64-Newton cbrt   (= fast-ssim2 0.7.1 numerics)
  cbrt64            HEAD's unfused matmul + f64-Newton cbrt
  cppcbrt           fused opsin matmul + jpegli CubeRootAndAdd (bias fused in)
  cppcbrt_unfusedmm HEAD's unfused matmul + jpegli CubeRootAndAdd
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "tmp/ssim2-parity-study/trees"

JPEGLI_FN = '''
/// Transliteration of jpegli `lib/base/fast_math-inl.h::CubeRootAndAdd`,
/// which is what the C++ SSIMULACRA2 binary evaluates.
/// Highway `MulAdd(a,b,c) = a*b+c`, `NegMulAdd(a,b,c) = c-a*b`, both fused.
#[inline(always)]
fn cbrt_and_add_jpegli(x: f32, add: f32) -> f32 {
    const K_EXP_BIAS: i32 = 0x5480_0000;
    const K_EXP_MUL: i32 = 0x002A_AAAA;
    const K1_3: f32 = 1.0 / 3.0;
    const K4_3: f32 = 4.0 / 3.0;
    let xa = x;
    let xa_3 = K1_3 * xa;
    let m1 = xa.to_bits() as i32;
    let m2 = if m1 == 0 { 0 } else { K_EXP_BIAS - (m1 >> 23) * K_EXP_MUL };
    let mut r = f32::from_bits(m2 as u32);
    for _ in 0..3 {
        let r2 = r * r;
        r = (-xa_3).mul_add(r2 * r2, K4_3 * r);
    }
    let mut r2 = r * r;
    r = K1_3.mul_add((-xa).mul_add(r2 * r2, r), r);
    r2 = r * r;
    r2.mul_add(x, add)
}
'''

# --- the region of the vector body that evaluates the cube root ---
VEC_CBRT_OLD = """        // Scalar initial estimates (integer bit manipulation — can't vectorize)
        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..LANES {
            est0[i] = cbrtf_initial_f32(est0[i]);
            est1[i] = cbrtf_initial_f32(est1[i]);
            est2[i] = cbrtf_initial_f32(est2[i]);
        }

        // Halley's method iterations in SIMD (3 channels interleaved for ILP)
        let mut t0 = f32x8::from_array(token, est0);
        let mut t1 = f32x8::from_array(token, est1);
        let mut t2 = f32x8::from_array(token, est2);

        // Iteration 1
        let mut r0 = t0 * t0 * t0;
        let mut r1 = t1 * t1 * t1;
        let mut r2 = t2 * t2 * t2;
        t0 *= mixed0.mul_add(two, r0) / (mixed0 + r0.mul_add(two, zero));
        t1 *= mixed1.mul_add(two, r1) / (mixed1 + r1.mul_add(two, zero));
        t2 *= mixed2.mul_add(two, r2) / (mixed2 + r2.mul_add(two, zero));

        // Iteration 2
        r0 = t0 * t0 * t0;
        r1 = t1 * t1 * t1;
        r2 = t2 * t2 * t2;
        t0 *= mixed0.mul_add(two, r0) / (mixed0 + r0.mul_add(two, zero));
        t1 *= mixed1.mul_add(two, r1) / (mixed1 + r1.mul_add(two, zero));
        t2 *= mixed2.mul_add(two, r2) / (mixed2 + r2.mul_add(two, zero));

        let mixed0 = t0 + absorb_bias;
        let mixed1 = t1 + absorb_bias;
        let mixed2 = t2 + absorb_bias;
"""

# per-lane scalar cbrt, applied inside the vector body (study only: same value
# the vector form computes on an FMA target, just slower)
VEC_CBRT_F64 = """        // STUDY VARIANT: f64-Newton cbrt (fast-ssim2 0.7.1 numerics), per lane.
        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..LANES {
            est0[i] = cbrtf_fast(est0[i]);
            est1[i] = cbrtf_fast(est1[i]);
            est2[i] = cbrtf_fast(est2[i]);
        }
        let mixed0 = f32x8::from_array(token, est0) + absorb_bias;
        let mixed1 = f32x8::from_array(token, est1) + absorb_bias;
        let mixed2 = f32x8::from_array(token, est2) + absorb_bias;
"""

VEC_CBRT_JPEGLI = """        // STUDY VARIANT: jpegli CubeRootAndAdd, per lane, bias fused into the
        // last multiply-add exactly as the C++ does.
        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..LANES {
            est0[i] = cbrt_and_add_jpegli(est0[i], absorbance_bias);
            est1[i] = cbrt_and_add_jpegli(est1[i], absorbance_bias);
            est2[i] = cbrt_and_add_jpegli(est2[i], absorbance_bias);
        }
        let mixed0 = f32x8::from_array(token, est0);
        let mixed1 = f32x8::from_array(token, est1);
        let mixed2 = f32x8::from_array(token, est2);
"""

MM_UNFUSED = """        let mixed0 = m00 * r + (m01 * g + (m02 * b + bias));
        let mixed1 = m10 * r + (m11 * g + (m12 * b + bias));
        let mixed2 = m20 * r + (m21 * g + (m22 * b + bias));
"""
MM_FUSED = """        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)));
"""

# --- scalar per-pixel path (also the len%8 remainder) ---
SC_CBRT_OLD = """    mixed0 = cbrtf_halley_f32(mixed0) + absorbance_bias;
    mixed1 = cbrtf_halley_f32(mixed1) + absorbance_bias;
    mixed2 = cbrtf_halley_f32(mixed2) + absorbance_bias;
"""
SC_CBRT_F64 = """    mixed0 = cbrtf_fast(mixed0) + absorbance_bias;
    mixed1 = cbrtf_fast(mixed1) + absorbance_bias;
    mixed2 = cbrtf_fast(mixed2) + absorbance_bias;
"""
SC_CBRT_JPEGLI = """    mixed0 = cbrt_and_add_jpegli(mixed0, absorbance_bias);
    mixed1 = cbrt_and_add_jpegli(mixed1, absorbance_bias);
    mixed2 = cbrt_and_add_jpegli(mixed2, absorbance_bias);
"""
SC_MM_UNFUSED = """    let mut mixed0 = m[0] * r + (m[1] * g + (m[2] * b + OPSIN_ABSORBANCE_BIAS));
    let mut mixed1 = m[3] * r + (m[4] * g + (m[5] * b + OPSIN_ABSORBANCE_BIAS));
    let mut mixed2 = m[6] * r + (m[7] * g + (m[8] * b + OPSIN_ABSORBANCE_BIAS));
"""
SC_MM_FUSED = """    let mut mixed0 = m[0].mul_add(r, m[1].mul_add(g, m[2].mul_add(b, OPSIN_ABSORBANCE_BIAS)));
    let mut mixed1 = m[3].mul_add(r, m[4].mul_add(g, m[5].mul_add(b, OPSIN_ABSORBANCE_BIAS)));
    let mut mixed2 = m[6].mul_add(r, m[7].mul_add(g, m[8].mul_add(b, OPSIN_ABSORBANCE_BIAS)));
"""

VARIANTS = {
    "v071repro":         ("f64", True),
    "cbrt64":            ("f64", False),
    "cppcbrt":           ("jpegli", True),
    "cppcbrt_unfusedmm": ("jpegli", False),
}


def sub(text, old, new, what, path):
    if old not in text:
        sys.exit(f"FAILED to find {what} in {path}")
    if text.count(old) != 1:
        sys.exit(f"{what} appears {text.count(old)}x in {path}")
    return text.replace(old, new)


for name, (cbrt, fused_mm) in VARIANTS.items():
    p = ROOT / name / "fast-ssim2/src/xyb_simd.rs"
    t = p.read_text()
    if cbrt == "f64":
        t = sub(t, VEC_CBRT_OLD, VEC_CBRT_F64, "vector cbrt", p)
        t = sub(t, SC_CBRT_OLD, SC_CBRT_F64, "scalar cbrt", p)
    else:
        t = sub(t, VEC_CBRT_OLD, VEC_CBRT_JPEGLI, "vector cbrt", p)
        t = sub(t, SC_CBRT_OLD, SC_CBRT_JPEGLI, "scalar cbrt", p)
        t = t.replace("const OPSIN_ABSORBANCE_BIAS: f32 = K_B0;",
                      "const OPSIN_ABSORBANCE_BIAS: f32 = K_B0;\n" + JPEGLI_FN)
    if fused_mm:
        t = sub(t, MM_UNFUSED, MM_FUSED, "vector matmul", p)
        t = sub(t, SC_MM_UNFUSED, SC_MM_FUSED, "scalar matmul", p)
    # keep the compiler quiet about now-unused helpers
    t = t.replace("#[inline(always)]\nfn cbrtf_initial_f32", "#[allow(dead_code)]\n#[inline(always)]\nfn cbrtf_initial_f32")
    t = t.replace("#[inline(always)]\nfn cbrtf_halley_f32", "#[allow(dead_code)]\n#[inline(always)]\nfn cbrtf_halley_f32")
    t = t.replace("#[inline]\nfn cbrtf_fast", "#[allow(dead_code)]\n#[inline]\nfn cbrtf_fast")
    p.write_text(t)
    # unique package name so cargo can hold all variants in one graph
    mani = ROOT / name / "fast-ssim2/Cargo.toml"
    m = mani.read_text().replace('name = "fast-ssim2"', f'name = "fast-ssim2-{name}"', 1)
    mani.write_text(m)
    print(f"patched {name}: cbrt={cbrt} fused_matmul={fused_mm}")
