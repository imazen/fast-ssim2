//! Which cube root is closest to the one the C++ SSIMULACRA2 binary evaluates?
//!
//! Implementations, copied verbatim from their sources:
//!   * `cbrtf_fast`          — fast-ssim2 0.7.1 (and HEAD's bias helper):
//!                             bit-hack estimate + 2 Newton steps in f64.
//!   * `cbrtf_halley_f32`    — fast-ssim2 0.8.2 / 0.9.0 vector body:
//!                             bit-hack estimate + 2 Halley steps in f32.
//!   * `cbrt_and_add_jpegli` — transliteration of jpegli
//!                             lib/base/fast_math-inl.h::CubeRootAndAdd
//!                             (reciprocal-cbrt Newton, add fused into the last op).

// ---- fast-ssim2 0.7.1 / HEAD `cbrtf_fast` (verbatim) ----
#[inline]
fn cbrtf_fast(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7FFF_FFFF;
    hx = hx / 3 + B1;
    ui &= 0x8000_0000;
    ui |= hx;
    let mut t: f64 = f64::from(f32::from_bits(ui));
    let xf64 = f64::from(x);
    let mut r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

// ---- fast-ssim2 0.8.2 / 0.9.0 vector cbrt (verbatim) ----
#[inline(always)]
fn cbrtf_initial_f32(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let hx = (ui & 0x7FFF_FFFF) / 3 + B1;
    f32::from_bits((ui & 0x8000_0000) | hx)
}

#[inline(always)]
fn cbrtf_halley_f32(x: f32) -> f32 {
    let mut t = cbrtf_initial_f32(x);
    for _ in 0..2 {
        let r = t * t * t;
        t *= x.mul_add(2.0, r) / (x + r.mul_add(2.0, 0.0));
    }
    t
}

// ---- jpegli CubeRootAndAdd, transliterated ----
// Highway: MulAdd(a,b,c) = a*b + c (fused); NegMulAdd(a,b,c) = c - a*b (fused).
#[inline(always)]
fn cbrt_and_add_jpegli(x: f32, add: f32) -> f32 {
    const K_EXP_BIAS: i32 = 0x5480_0000;
    const K_EXP_MUL: i32 = 0x002A_AAAA;
    const K1_3: f32 = 1.0 / 3.0;
    const K4_3: f32 = 4.0 / 3.0;

    let xa = x;
    let xa_3 = K1_3 * xa;

    let m1 = xa.to_bits() as i32;
    let m2 = if m1 == 0 {
        0
    } else {
        K_EXP_BIAS - (m1 >> 23) * K_EXP_MUL
    };
    let mut r = f32::from_bits(m2 as u32);

    // Three reciprocal-cube-root Newton iterations.
    for _ in 0..3 {
        let r2 = r * r;
        r = (-xa_3).mul_add(r2 * r2, K4_3 * r);
    }
    // Final iteration + fused add.
    let mut r2 = r * r;
    r = K1_3.mul_add((-xa).mul_add(r2 * r2, r), r);
    r2 = r * r;
    r2.mul_add(x, add)
}

/// The transliteration currently in fast-ssim2 `src/cpp_parity_diag.rs`.
/// Same op *count* as jpegli's, but the FMAs group different products:
/// jpegli fuses `xa_3 * r4` into the subtract, this fuses `k4_3 * r` instead.
#[inline(always)]
fn cbrt_and_add_repo_diag(x: f32, add: f32) -> f32 {
    const K_EXP_BIAS: i32 = 0x5480_0000;
    const K_EXP_MUL: i32 = 0x002A_AAAA;
    const K1_3: f32 = 1.0 / 3.0;
    const K4_3: f32 = 4.0 / 3.0;
    let xa = x;
    let xa_3 = K1_3 * xa;
    let m1 = xa.to_bits() as i32;
    let m2 = if m1 == 0 { 0 } else { K_EXP_BIAS - ((m1 >> 23) * K_EXP_MUL) };
    let mut r = f32::from_bits(m2 as u32);
    for _ in 0..3 {
        let r2 = r * r;
        r = K4_3.mul_add(r, -(xa_3 * (r2 * r2)));
    }
    let mut r2 = r * r;
    r = K1_3.mul_add(r.mul_add(1.0, -(xa * (r2 * r2))), r);
    r2 = r * r;
    r2.mul_add(x, add)
}

/// Reference: correctly-rounded-ish cube root through f64.
#[inline]
fn cbrt_true(x: f32) -> f32 {
    (f64::from(x)).cbrt() as f32
}

fn ulp_at(v: f32) -> f64 {
    let a = v.abs();
    if a == 0.0 {
        return f64::from(f32::from_bits(1));
    }
    let next = f32::from_bits(a.to_bits() + 1);
    f64::from(next) - f64::from(a)
}

#[derive(Default)]
struct Stat {
    n: u64,
    sum_abs: f64,
    max_abs: f64,
    max_at: f32,
    sum_signed: f64,
    n_pos: u64,
    n_exact: u64,
    sum_ulp: f64,
    max_ulp: f64,
}

impl Stat {
    fn push(&mut self, got: f32, want: f32, x: f32) {
        let d = f64::from(got) - f64::from(want);
        self.n += 1;
        self.sum_abs += d.abs();
        self.sum_signed += d;
        if d > 0.0 {
            self.n_pos += 1;
        }
        if d == 0.0 {
            self.n_exact += 1;
        }
        if d.abs() > self.max_abs {
            self.max_abs = d.abs();
            self.max_at = x;
        }
        let u = d.abs() / ulp_at(want);
        self.sum_ulp += u;
        if u > self.max_ulp {
            self.max_ulp = u;
        }
    }
    fn report(&self, label: &str) {
        println!(
            "{label:<34} mean|d| {:>11.4e}  max|d| {:>11.4e}  mean_ulp {:>7.3}  max_ulp {:>7.3}  bitexact {:>6.2}%  bias {:>+11.4e}  n>0 {:>5.1}%",
            self.sum_abs / self.n as f64,
            self.max_abs,
            self.sum_ulp / self.n as f64,
            self.max_ulp,
            100.0 * self.n_exact as f64 / self.n as f64,
            self.sum_signed / self.n as f64,
            100.0 * self.n_pos as f64 / self.n as f64,
        );
    }
}

fn main() {
    // Constants, both spellings.
    const OURS_B0: f32 = 0.003_793_073_4_f32;
    const CPP_B0: f32 = 0.0037930732552754493_f32;
    println!("K_B0 ours bits {:#010x}  jpegli bits {:#010x}  equal {}",
        OURS_B0.to_bits(), CPP_B0.to_bits(), OURS_B0.to_bits() == CPP_B0.to_bits());

    // The additive constant: C++ uses libm cbrtf; fast-ssim2 uses cbrtf_fast.
    let add_cpp = -(f64::from(CPP_B0).cbrt() as f32); // libm cbrtf(f32) == correctly rounded here
    let add_ours = -cbrtf_fast(OURS_B0);
    println!(
        "absorbance_bias  C++ {:.10} ({:#010x})   ours {:.10} ({:#010x})   delta {:+.3e}",
        add_cpp, add_cpp.to_bits(), add_ours, add_ours.to_bits(),
        f64::from(add_ours) - f64::from(add_cpp)
    );
    println!();

    // Sweep the domain the opsin stage produces: [kB0, ~1.004].
    let lo = CPP_B0.to_bits();
    let hi = 1.004_f32.to_bits();
    let stride: u32 = std::env::var("STRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    println!("sweeping f32 in [{}, {}] stride {stride} => {} samples\n",
        CPP_B0, 1.004_f32, (hi - lo) / stride);

    // Accuracy of each cbrt against true cbrt (add = 0).
    let mut s_071_true = Stat::default();
    let mut s_082_true = Stat::default();
    let mut s_cpp_true = Stat::default();
    // Agreement of each *pipeline output* (cbrt(x) + add) with the C++ pipeline output.
    let mut s_071_cpp = Stat::default();
    let mut s_082_cpp = Stat::default();
    let mut s_translit_cpp = Stat::default();
    // Also: our own add constant vs the C++ one, isolated.
    let mut s_082_cppadd = Stat::default();
    let mut s_repodiag_cpp = Stat::default();
    let mut s_repodiag_true = Stat::default();

    let mut bits = lo;
    while bits <= hi {
        let x = f32::from_bits(bits);
        let t = cbrt_true(x);
        let c071 = cbrtf_fast(x);
        let c082 = cbrtf_halley_f32(x);
        let cpp_out = cbrt_and_add_jpegli(x, add_cpp);
        let cpp_cbrt_only = cbrt_and_add_jpegli(x, 0.0);

        s_071_true.push(c071, t, x);
        s_082_true.push(c082, t, x);
        s_cpp_true.push(cpp_cbrt_only, t, x);

        s_071_cpp.push(c071 + add_ours, cpp_out, x);
        s_082_cpp.push(c082 + add_ours, cpp_out, x);
        s_translit_cpp.push(cbrt_and_add_jpegli(x, add_cpp), cpp_out, x);
        s_082_cppadd.push(c082 + add_cpp, cpp_out, x);
        s_repodiag_cpp.push(cbrt_and_add_repo_diag(x, add_cpp), cpp_out, x);
        s_repodiag_true.push(cbrt_and_add_repo_diag(x, 0.0), t, x);

        bits = match bits.checked_add(stride) {
            Some(b) => b,
            None => break,
        };
    }

    println!("--- cube root itself, vs f64 cbrt (add = 0) ---");
    s_071_true.report("0.7.1  f64 Newton");
    s_082_true.report("0.8.2  f32 Halley");
    s_cpp_true.report("jpegli CubeRootAndAdd");
    s_repodiag_true.report("repo cpp_parity_diag translit");
    println!();
    println!("--- pipeline value cbrt(x)+bias, vs what C++ computes ---");
    s_071_cpp.report("0.7.1  (f64 cbrt + our bias)");
    s_082_cpp.report("0.8.2  (f32 Halley + our bias)");
    s_082_cppadd.report("0.8.2  cbrt, C++ bias const");
    s_translit_cpp.report("jpegli transliteration");
    s_repodiag_cpp.report("repo cpp_parity_diag translit");
    println!();
    println!("worst-case x: 0.7.1 {:.9}  0.8.2 {:.9}", s_071_cpp.max_at, s_082_cpp.max_at);
}
