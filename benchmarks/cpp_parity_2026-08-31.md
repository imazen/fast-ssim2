# C++ SSIMULACRA2 parity and cross-tier consistency — 2026-08-31

## Provenance

| | |
|---|---|
| repo commit (pre-change base) | `f6af3af5` |
| host | Apple M4 Pro, Darwin 25.5.0 arm64 |
| toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18), `--release` |
| reference binary | `/opt/homebrew/bin/ssimulacra2` -> `Cellar/jpeg-xl/0.12.0/bin/ssimulacra2` |
| reference source read | `~/work/jpegli/tools/ssimulacra2.cc`, `tools/gauss_blur.cc`, `lib/extras/xyb_transform.cc`, `lib/base/fast_math-inl.h` (read-only) |
| fast-ssim2 versions measured | 0.7.1 (crates.io), 0.8.2 (crates.io), local working tree |
| corpus | `~/work/zen/codec-corpus`: CID22-512, KADID-10k, gb82, CLIC2025 (6 refs each = 24) |
| harnesses | `examples/photo_parity.rs`, `examples/parity_report.rs`, `examples/flat_field_probe.rs`, `src/cpp_parity_diag.rs` |

**x86_64 (AVX2 / AVX-512) and wasm128 are NOT MEASURED.** This host is aarch64.
Every number below is aarch64/NEON unless stated otherwise. Nothing here has
been extrapolated to another ISA.

Large result files (over the 30 KB commit limit) live outside the repo:

| file | sha256 (first 16) | path |
|---|---|---|
| `photo_parity_v082.tsv` (before) | `878d8060d661c461` | `~/tmp/fast-ssim2-parity/` |
| `photo_parity_final.tsv` (after) | `a3e22d4448bac7f4` | `~/tmp/fast-ssim2-parity/` |
| `version_equivalence.tsv` | `fef85138f8942ccd` | `~/tmp/fast-ssim2-parity/` |

---

## 1. The synthetic reference table proves much less than it looks like

`src/reference_data.rs` holds 66 cases and `tests/reference_parity.rs` passes.
But **40 of the 66 compare an image against itself**, and every implementation
returns exactly `100.0` for an identical pair regardless of its arithmetic. The
zero error on `perfect_match`, `gradients`, `checkerboard`, `noise` and `edges`
is a constant, not evidence of parity.

That leaves 26 informative cases, of which 20 are `uniform_shift` — the one
family that is numerically degenerate (section 2). Signed error vs the live C++
binary, `examples/parity_report.rs`:

| pattern | n | min | max | mean | mean abs | n > 0 |
|---|--:|--:|--:|--:|--:|--:|
| perfect_match / gradients / checkerboard / noise / edges | 40 | 0 | 0 | 0 | 0 | 0 |
| synthetic_vs | 2 | -0.00074 | +0.00154 | +0.00040 | 0.00114 | 1 |
| distortions | 4 | -0.00088 | +0.18507 | +0.05655 | 0.05699 | 3 |
| **uniform_shift** | **20** | **-0.06373** | **+1.99695** | **+0.40412** | **0.41298** | **18** |

**The uniform_shift divergence is one-directional: 18 of 20 positive.** FP noise
scatters both ways; this does not. The comment in `tests/reference_parity.rs`
that attributed the 10.0 tolerance to "different FP rounding in SIMD paths" was
wrong twice — the two fast-ssim2 backends agree here to ~1e-8, so it is not a
SIMD-path effect at all, and the bias has a specific mechanism. Corrected in
place.

## 2. Root cause: SSIMULACRA2 is ill-conditioned on flat fields, by ~1e6

On a flat field `sigma11 - mu1^2`, `sigma22 - mu2^2` and `sigma12 - mu1*mu2`
are analytically zero away from the border, so both `num_s` and `denom_s`
collapse onto `kC2 = 9e-4` plus whatever the f32 recursive Gaussian left
behind. Measured at the image centre (`cpp_parity_diag::diag_flat_field_conditioning`,
Y plane, XYB value 0.4574):

| case | true signal `(a-b)^2` | blur residual in `sigma11 - mu1^2` | resulting `d` | rounding share |
|---|--:|--:|--:|--:|
| `uniform_shift_1_32x32` | 1.114e-5 | -5.51e-7 | 4.429e-5 | **+3.31e-5 = 3x the signal** |
| `uniform_shift_1_256x256` | 1.114e-5 | -4.92e-7 | **0.0 (clamped)** | -1.11e-5 = signal erased |
| `uniform_shift_50_32x32` | 2.637e-2 | -5.51e-7 | 2.696e-2 | +5.97e-4 = 2% |

Fraction of pixels where `|d - d_exact| > d_exact`: **60%** at
`uniform_shift_1_32x32`, **53%** at 256x256. `d = max(1 - num_m*num_s/denom_s, 0)`
then *rectifies* the residual, so it cannot cancel across the image — it
accumulates one-directionally, and the implementation with the noisier blur
reports the lower score. That is why fast-ssim2 scores higher: **its blur is
less noisy than the reference's, not wrong.**

Corroboration, three independent ways:

1. **The C++ values are not monotonic in the shift.** At 32x32: shift 1 ->
   97.749, shift 5 -> 98.808, shift 10 -> 96.531. A bigger distortion scoring
   *better* is what a noise-dominated measurement looks like.

2. **Perturbing any stage by ~1e-7 moves the score by ~0.5**
   (`diag_uniform_shift_by_stage`). Swapping only the cube root — ours (1.75
   ulp) for jpegli's own `CubeRootAndAdd` (3.34 ulp) — moves
   `uniform_shift_1_32x32` from 99.746 to 99.232 and `uniform_shift_5_32x32`
   from 98.527 to 98.013, in *opposite* directions relative to C++.

3. **Substituting jpegli's own horizontal Gaussian closes a third of the gap**
   (`diag_uniform_shift_with_cpp_blur`). jpegli's `FastGaussian1D` computes four
   outputs per iteration from f32-rounded 2nd/3rd/4th-power coefficients on
   every non-scalar Highway target; fast-ssim2 runs the plain single-step
   recurrence. Transliterating jpegli's form into our pipeline:

   | | mean abs delta vs C++ binary, 20 uniform_shift cases |
   |---|--:|
   | our sequential recurrence | 0.381 |
   | jpegli's 4-unrolled recurrence | **0.258** |

   At 64x64 it is near-total: all five cases land within 0.24, three within
   0.008. **The C++ reference is therefore not arch-consistent with itself** —
   its horizontal blur differs between `HWY_SCALAR` and its vector targets, so
   "bit-exact agreement with the C++ tool" is not achievable in principle
   without also matching the vector width it was built for.

**Verdict on the uniform_shift family: not a bug, and not usable as a parity
gate.** The reference values there are dominated by the reference's own rounding.
The 10.0 tolerance is retained because tightening it would assert agreement with
a noise measurement; the justification comment has been replaced with the above.

## 3. Real photographic content — where the metric is well conditioned

`examples/photo_parity.rs`, 24 references x 4 sizes (32, 64, 256, full<=1024) x 6
distortions (JPEG q90/q50/q10, box blur r2, +/-8 noise, +/-6 chroma shift) =
**576 pairs**, each scored by the C++ binary and by both fast-ssim2 backends.

| group | n | mean(simd - C++) | mean abs | max abs |
|---|--:|--:|--:|--:|
| ALL | 576 | +0.00219 | **0.02386** | 0.5233 |
| size 32 | 144 | -0.00676 | 0.04132 | 0.3224 |
| size 64 | 144 | -0.00339 | 0.01169 | 0.1101 |
| size 256 | 144 | +0.00353 | 0.01676 | 0.1129 |
| size full | 144 | +0.01539 | 0.02569 | 0.5233 |
| jpeg_q90 | 96 | +0.00575 | 0.03050 | 0.2616 |
| jpeg_q50 | 96 | -0.00167 | 0.01903 | 0.1012 |
| jpeg_q10 | 96 | -0.00498 | 0.01314 | 0.3224 |
| boxblur_r2 | 96 | -0.00176 | 0.01417 | 0.1285 |
| noise_a8 | 96 | -0.00086 | 0.01737 | 0.0678 |
| chroma_shift_6 | 96 | +0.01668 | 0.04897 | 0.5233 |
| CID22 | 144 | -0.00526 | 0.02114 | 0.1700 |
| KADID-10k | 144 | -0.00413 | 0.01989 | 0.3224 |
| gb82 | 144 | +0.01818 | 0.03205 | 0.5233 |
| CLIC2025 | 144 | -0.00002 | 0.02238 | 0.2010 |

**283 of 576 deltas are positive** — symmetric, no directional bias. The
`uniform_shift` bias does not exist on real content, exactly as the degeneracy
explanation predicts. Mean absolute agreement is **0.024 on a 0..100 scale**,
about 1/50 of a SSIMULACRA2 JND.

The largest residuals are the smooth ones (gb82 `girl`/`baby` under a chroma
shift, at full size), which is the same flat-field amplification: photographs
with large smooth regions re-enter the ill-conditioned regime.

`tests/jpeg_quality_reference.rs` against the same binary, after the changes:
Q20 0.0349, Q45 0.0447, Q70 0.0506, Q90 0.0084 (tolerance 0.15, unchanged).

## 4. Cross-tier consistency — two real bugs found and fixed

### 4a. `SimdImpl::Scalar` and `SimdImpl::Simd` computed different metrics

`SimdImpl::Scalar` routed XYB conversion through `yuvxyb`, whose cube root is
two Newton steps in f64 (0.50 ulp); `SimdImpl::Simd` used two Halley steps in
f32 (1.75 ulp). Amplified by the flat-field conditioning, that 1.5e-7 difference
produced score differences of up to **0.879** on real photographs — above the
0.5 tolerance `tests/simd_consistency.rs` carried, though its synthetic images
never provoked it.

Separately, inside `linear_rgb_to_xyb_inner` the `len % 8` remainder pixels used
the f64 cube root while the vectorised body used the f32 one, so a plane's last
seven pixels were converted with different math than the rest of it.

Fixed by giving every path one cube root (`cbrtf_halley_f32`) and adding
`linear_rgb_to_xyb_scalar`, a genuinely scalar conversion that is bit-identical
to the vector one.

Cube-root accuracy, exhaustive sweep of all 67 628 509 f32 values in the domain
the opsin stage produces (`[kB0, 1.004]`), `diag_cbrt_accuracy`:

| implementation | max abs error | ulp |
|---|--:|--:|
| 2 Halley steps f32 (fast-ssim2, all paths) | 1.92e-7 | 1.75 |
| 3 Halley steps f32 | 1.89e-7 | 1.58 |
| 4 Halley steps f32 | 1.89e-7 | 1.58 |
| 2 Newton steps f64 (old `SimdImpl::Scalar`) | 5.96e-8 | 0.50 |
| jpegli `CubeRootAndAdd` (the C++ reference) | 3.17e-7 | 3.34 |

Extra Halley steps do not help — the f32 iteration is rounding-limited, not
convergence-limited. Note the reference's own cube root is the least accurate of
the three, and documents itself as "6 ulp max error".

| | max abs(simd - scalar), 576 real pairs |
|---|--:|
| before | **0.878965** |
| after | **0.000000262** |

### 4b. `magetypes` does not fuse `mul_add` on every backend

Measured across all 25 archmage token permutations this host can emulate
(`diag_kernel_tier_divergence`): `linear_rgb_to_xyb_simd` returned values
**1.788e-7** apart between the NEON arm and the scalar-polyfill arm, which the
metric amplified to **0.085** end-to-end.

Cause, read from `magetypes` 0.9.28 source:

| backend | `mul_add` for `f32x8` | fused? |
|---|---|---|
| `impls/x86_v3.rs` | `_mm256_fmadd_ps` | yes |
| `impls/arm_neon.rs` | `vfmaq_f32` | yes |
| `impls/wasm128.rs` | `f32x4_add(f32x4_mul(a,b), c)` | **no** ("WASM has no native FMA") |
| `impls/scalar.rs` | `a[i] * b[i] + c[i]` | **no** |

`magetypes`' own 1-lane `f32x1::mul_add` uses `nostd_math::fmaf`, so the crate
is internally inconsistent about what `mul_add` means; the 8-lane polyfill could
use the same software FMA. **This is an upstream defect and the fix belongs in
`archmage`/`magetypes`, which was not touched.**

Worked around here by removing every fusion-sensitive `mul_add` from the
dispatched kernels *except* the blur:

- `ssim_map`: `num_m` is now `1 - mu_diff*mu_diff`, which also matches the
  reference (the C++ writes this term unfused).
- `xyb_simd`: the opsin matrix is now an explicit unfused chain in both arms.
- `edge_diff_map`: see 4c.
- `num_s = fma(2, x, kC2)` is left fused — `2*x` is exact, so fusion cannot
  change it.

After: all four dispatched kernels are **bit-identical across all 25
permutations** (`dispatched_kernels_are_bit_identical_across_tiers`).

**The blur was deliberately left fused**, because the C++ reference fuses it and
matching the authority won on measurement. Unfusing it was implemented and
measured over the same 576 pairs:

| blur `MUL_PREV` step | mean(simd - C++) | mean abs | max abs | cross-tier spread |
|---|--:|--:|--:|--:|
| **fused (shipped)** | +0.0022 | **0.0239** | 0.523 | up to 0.497 on smooth content |
| unfused | **-0.0580** | 0.0674 | 1.036 | 0 (bit-identical) |

Unfusing makes every target agree with every other at the cost of agreeing with
the reference 2.8x less well *and* acquiring a systematic -0.058 bias, so it was
rejected.

**Correction (same day, after review):** this section originally called the
magetypes behaviour "a fixable upstream gap." It is not fixable upstream and it
is not a gap. `magetypes/src/simd/impls/wasm128.rs:112` emits
`f32x4_add(f32x4_mul(a,b), c)` because **WASM SIMD128 has no FMA instruction**;
`scalar.rs:125` calls `nostd_math::fmaf`, documented at its definition as a
deliberate `a * b + c` fallback since there is no hardware FMA to use, and a
correct software FMA would be slow. archmage documents the whole class in a
table headed *"they are not bugs to fix"*, prescribing *"avoid near-zero
cancellation"* — which is precisely the flat-field regime analysed above. The
divergence is inherent to unfused targets, not pending an upstream fix. Perf, for the record: unfusing cost +2.9% / +3.2% / +1.8% on
512/1024/2048 blur (`examples/benchmark_blur.rs`), i.e. it was not rejected on
speed.

**Consequence, stated plainly: fast-ssim2 is bit-identical across every
FMA-capable target (all x86-64 with AVX2, all aarch64), and diverges by up to
0.497 on wasm128 and on no-SIMD builds until `magetypes` fuses `mul_add` there.**
The reference has the same property for the same reason.

### 4c. Catastrophic cancellation in the SIMD edge-diff kernel

`edge_diff_map_inner` computed `d1 = (1 + diff2)/(1 + diff1) - 1` in **f32**,
while `edge_diff_map_scalar` and the C++ reference compute it in f64. In f32
that subtraction carries ~1 ulp of 1.0 (6e-8) of *absolute* error however small
the true value is — 100% error on smooth content, which `max(d1, 0)` then
rectifies. The `len % 8` remainder of the same function used f64, so a pixel's
precision depended on its index.

Replaced with the algebraically identical `(diff2 - diff1) / (1 + diff1)`, which
has no cancellation (~1 ulp *relative*), in both the vector body and its
remainder, and in the f64 scalar kernel so the two share one expression.

### 4d. Measured tier-to-tier maxima (the numbers asked for)

All 25 token permutations x both backends = 50 combinations per image,
`tests/simd_consistency.rs`, which now partitions by blur-equivalence class
rather than trusting a single blanket number:

| image | combos | blur classes | spread, same blur | spread, across blur classes |
|---|--:|--:|--:|--:|
| textured 32x32 | 50 | 2 | 2.164e-8 | 4.750e-2 |
| smooth 32x32 | 50 | 2 | 9.421e-10 | 2.313e-1 |
| textured 64x64 | 50 | 2 | 1.719e-9 | 8.292e-2 |
| smooth 64x64 | 50 | 2 | 1.242e-9 | 1.549e-1 |
| textured 128x128 | 50 | 2 | 2.244e-8 | 5.793e-2 |
| smooth 128x128 | 50 | 2 | 5.240e-9 | **4.969e-1** |

Blur output itself, per sample, across tiers: **2.533e-6** (gated at 1e-5).

Tolerances changed from one blanket `0.5` to:

| gate | bound | measured |
|---|--:|--:|
| dispatched kernels, across tiers | bit-identical | bit-identical |
| blur plane, across tiers | 1e-5 / sample | 2.533e-6 |
| score, combinations with identical blur | 1e-4 | 2.244e-8 |
| score, across blur classes (known magetypes gap) | 6e-1 | 4.969e-1 |

## 5. Which version should the workspace unify on?

`~/tmp/ssim2-vercmp`, 24 references x 3 sizes x 5 distortions = **360 cells**,
each scored by the C++ binary, fast-ssim2 0.7.1, 0.8.2 and the local tree.
(0.7.2 and 0.7.3 are **yanked** on crates.io, so `^0.7.1` really does resolve to
0.7.1 for jxl-encoder, zengif and zenwebp.)

| version | mean(v - C++) | mean abs | max abs |
|---|--:|--:|--:|
| 0.7.1 | +0.00206 | **0.01294** | 0.1833 |
| 0.8.2 | +0.00196 | **0.01405** | 0.1640 |
| local (0.8.2 + this work) | +0.00128 | **0.01350** | 0.1954 |

| pair | mean | mean abs | max abs | n > 0 |
|---|--:|--:|--:|--:|
| 0.8.2 - 0.7.1 | -0.000097 | 0.00954 | 0.1428 | 177/360 |
| local - 0.7.1 | -0.000779 | 0.00916 | 0.1026 | 179/360 |
| local - 0.8.2 | -0.000682 | 0.00823 | 0.0983 | 175/360 |

Cells where 0.7.1 is closer to C++ than 0.8.2: **189/360** — a coin flip.

**Neither version is more faithful to the reference.** The 0.0011 difference in
mean absolute error is far inside the per-cell scatter, and the sign of the
version delta is random (177/360 positive). So there is no accuracy argument for
staying on 0.7.1, and there is a concrete cost to the split: two groups of
codecs currently report scores that differ by up to **0.143** on identical
input. **Unify on 0.8.2** (or later) — on recency and maintenance, not accuracy.

Note against the earlier reading that 0.7.1-vs-0.8.2 showed "max delta 0.416,
suspiciously close to the 0.413 mean uniform_shift error": on this real-content
corpus the max is 0.143, and the resemblance is a coincidence. The mechanisms
are unrelated — the uniform_shift figure is ours-vs-reference on degenerate
content, the version figure is symmetric FP scatter on well-conditioned content.
A corpus with more flat content would raise the version delta, for the reason in
section 2.

## 6. Not measured

- **x86_64** (AVX2, AVX-512) — no such host here. Deferred to the Linux 7950X.
  The 0.8.2 blur vectorisation touched the ARM path, so the aarch64 result
  should not be assumed to reproduce.
- **wasm128** — the analysis in 4b says it takes the non-FMA path and should
  therefore land in the second blur-equivalence class, but that is read from
  `magetypes` source, not run.
- **i686** — same reasoning as wasm; archmage's `v3` tier is AVX2-era, so an
  i686 build below that takes the scalar polyfill.
- Whether jpegli's `Downsample`, its skcms sRGB EOTF, or `intensity_target`
  handling contribute to the residual 0.024 on real content. The blur and the
  cube root are accounted for; the rest is not attributed.
