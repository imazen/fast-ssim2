# Why 0.8.2 scores differently from 0.7.1, and which one is right about the C++ reference

**Date:** 2026-09-09 · **Host:** Apple M4 Pro (arm64), macOS 26.5.2 ·
**Toolchain:** rustc 1.98.0 · **Repo commit:** `c386795` (v0.9.0) ·
**C++ reference:** `/opt/homebrew/bin/ssimulacra2` (jpeg-xl 0.12.0, Homebrew) ·
**Corpus:** `~/work/zen/codec-corpus` — CID22-512, KADID-10k, gb82, CLIC2025,
24 references each = 96 references.

Reproduction sources are committed next to this file in
[`version_divergence_2026-09-09/`](version_divergence_2026-09-09/); the raw
2016-row TSV is not (388 KB, over this repo's 30 KB data-file limit). It sits at
`~/tmp/ssim2-parity-study/cbrt_attribution.tsv` on the measuring host and
regenerates from `driver_main.rs` in about three minutes; ask before committing a
gzipped copy (138 KB) if it is wanted in-tree.

This answers two questions the earlier record
([`cpp_parity_2026-08-31.md`](cpp_parity_2026-08-31.md) §5) left open: it found
0.7.1 and 0.8.2 "equally faithful … the sign of the version delta is random",
but did not say **what code change** made them differ, and treated the residual
disagreement with the C++ binary as unattributed.

## TL;DR

1. **The 0.7.1 → 0.8.2 divergence is the cube root, and nothing else.**
   0.8.2 replaced the f64 Newton–Raphson `cbrtf_fast` in the vector XYB body
   with two Halley steps carried in f32 (`844605f`, shipped as part of the
   `magetypes` port `b130548`). Rebuilding HEAD with that one function reverted
   reproduces 0.7.1 to **mean |Δ| 6.1e-6, max 7.4e-5** over 2016 cells — i.e.
   every other change in 0.7.1…0.8.2 (blur row-vectorisation, 4→8 lane widths
   on NEON, zero-weight cell skipping, the whole per-arch → generic refactor)
   is numerically inert. The version-to-version scatter is **mean |Δ| 0.0145,
   max 0.200**. `844605f`'s own message says it: *"Pinned SIMD scores updated to
   reflect the f32 Halley precision (< 0.05 absolute delta)"* — the score move
   was deliberate and known at the time; what was never traced is that this is
   the whole of the 0.7.1-vs-0.8.2 split, and that on real content the worst case
   is 0.20, not 0.05.
2. **Neither version is more faithful to C.** 0.7.1 − 0.8.2 in agreement with
   the C++ binary is **+0.00036, 95% CI [−0.00050, +0.00124]** — not
   significant, 983/2016 cells favour 0.7.1. The 2026-08-31 "coin flip" verdict
   survives 4× the cells. Unifying the workspace on 0.8.2+ remains right, on
   recency, not accuracy.
> **Landed 2026-09-09.** Point 3 below is no longer a proposal: jpegli's cube
> root and horizontal Gaussian now ship, the bias is gone (mean(ours − C++)
> +0.00012), and the metric got *faster*. See
> [`jpegli_kernels_2026-09-09.md`](jpegli_kernels_2026-09-09.md). Everything
> above it — the attribution, and the 0.7.1-vs-0.8.2 verdict — is unaffected.

3. **Both versions are wrong in the same direction, and it is fixable.** Every
   shipped configuration scores **+0.0067 ± 0.0016 above** the C++ binary on
   real content. Adopting jpegli's *own* two approximations — its
   `CubeRootAndAdd` and its 4-unrolled `FastGaussian1D` horizontal pass —
   removes that bias (**+0.0010, CI [−0.0001, +0.0021]**, no longer detectable)
   and cuts mean |Δ| by **21%** and max |Δ| by **51%**. That is a real, measured
   accuracy gain, not a rounding argument.

## 1. What was measured

Seven builds, scored against the C++ binary on the same 2016 cells. All seven
are fed the *same* linear-RGB buffers (from HEAD's `srgb_u8_to_linear`, asserted
bit-identical across 0.7.1 / 0.8.2 / HEAD for all 256 inputs), so only the
metric differs — no linearisation term leaks into the comparison.

| build | opsin matmul | cube root | horizontal blur |
|---|---|---|---|
| `v071` | fused FMA chain | bit-hack + 2 Newton steps in **f64** | ours (sequential) |
| `v082` | fused FMA chain | bit-hack + 2 Halley steps in **f32** | ours |
| `head` (0.9.0) | **unfused** (arch-consistency, see `xyb_simd.rs`) | f32 Halley | ours |
| `repro` | fused | f64 Newton | ours |
| `cbrt64` | unfused | f64 Newton | ours |
| `cppcbrt` | fused | **jpegli `CubeRootAndAdd`** | ours |
| `cppcbrtu` | unfused | jpegli `CubeRootAndAdd` | ours |
| `cppmax` | fused | jpegli `CubeRootAndAdd` | **jpegli 4-unrolled** |

Cells: 96 references × 3 sizes (64×64 centre crop, 256×256 centre crop, full
resized to ≤768) × 7 distortions (JPEG q90/q50/q10, box blur r2, ±8 LCG noise,
plus two *flat* families — contrast pushed to 12% then JPEG q90 / ±2 noise,
because flat fields are where SSIMULACRA2's `1/kC2` amplification of tiny XYB
differences bites).

## 2. Attribution — the cube root is the whole story

Paired per-cell differences, 2016 cells:

| pair | what it isolates | mean | mean \|Δ\| | max \|Δ\| |
|---|---|--:|--:|--:|
| `v082 − v071` | the historical divergence | +0.00018 | **0.01454** | 0.2003 |
| `repro − v071` | **everything except cbrt + matmul fusion** | +0.0000003 | **0.0000061** | **0.000074** |
| `cbrt64 − head` | cube root alone | −0.00042 | 0.01500 | 0.1802 |
| `repro − cbrt64` | opsin matmul fusion alone | −0.00025 | 0.01215 | 0.1582 |
| `cppmax − cppcbrt` | jpegli's horizontal blur alone | −0.00496 | 0.01737 | 0.4665 |

`repro − v071` at 6.1e-6 is the load-bearing row: HEAD's pipeline, with only the
cube root and the matmul association put back to 0.7.1's, *is* 0.7.1. So the
0.7.1→0.8.2 move is attributable to the cube root by itself (0.8.2 kept the
fused matmul; 0.9.0 is the release that unfused it), and both knobs move scores
by the same order of magnitude, ~0.012–0.015 mean, ~0.16–0.20 worst case.

The residual 7.4e-5 is not zero because the vector XYB body's `len % LANES`
remainder and the blur's `width % LANES` columns change hands when the lane
count goes 4 → 8 on NEON; those pixels take a scalar path.

### Why 1-ulp cube-root differences move a 0–100 score by 0.2

Exhaustive sweep over every f32 the opsin stage can produce, `[kB0, 1.004]`
= 67 628 509 values (`cbrt_study_main.rs`; the repo's `diag_cbrt_accuracy`
covers the same ground):

| cube root | mean \|err\| vs f64 `cbrt` | max | mean ulp | max ulp | bit-exact |
|---|--:|--:|--:|--:|--:|
| 0.7.1 f64 Newton ×2 | 0 | 0 | 0.00 | 0.00 | 100% |
| 0.8.2 / 0.9.0 f32 Halley ×2 | 1.78e-8 | 1.79e-7 | 0.49 | 3.0 | 54.9% |
| jpegli `CubeRootAndAdd` | 2.60e-8 | 2.98e-7 | 0.72 | 5.0 | 41.3% |

Note the ordering: **0.7.1's cube root is the most accurate of the three, and
jpegli's is the least.** "More accurate" and "closer to the reference tool" are
different goals, which is exactly why the version question has no answer on
accuracy grounds. Distance from *what the C++ actually computes*
(`cbrt(x) + (−cbrt(kB0))`, the constant confirmed bit-identical in both):

| build | mean \|Δ\| vs the C++ value | max |
|---|--:|--:|
| 0.7.1 | 2.78e-8 | 3.58e-7 |
| 0.8.2 / 0.9.0 | 3.28e-8 | 4.17e-7 |
| jpegli transliteration | **0** (bit-exact) | **0** |

A ~3e-8 per-sample XYB difference is what produces the 0.15–0.20 worst-case
score moves above; the 2026-08-31 record measured the same amplification from
the other direction (1.79e-7 in XYB → 0.085 on the 0–100 scale).

## 3. Which version is right about the C++ reference: neither

Mean |variant − C++| over 2016 cells, with a 95% CI from 4000 paired bootstrap
resamples of the cells:

| build | mean(v − C++) | mean \|Δ\| | 95% CI | max \|Δ\| |
|---|--:|--:|:--|--:|
| `v071` | +0.00673 | 0.02087 | [0.01954, 0.02223] | 0.4902 |
| `v082` | +0.00691 | 0.02051 | [0.01925, 0.02183] | 0.4309 |
| `head` (0.9.0) | +0.00740 | 0.02056 | [0.01920, 0.02194] | 0.5223 |
| `repro` | +0.00673 | 0.02087 | [0.01954, 0.02223] | 0.4902 |
| `cbrt64` | +0.00698 | 0.02039 | [0.01907, 0.02175] | 0.4718 |
| `cppcbrt` | +0.00595 | 0.01953 | [0.01836, 0.02070] | 0.3808 |
| `cppcbrtu` | +0.00535 | 0.01968 | [0.01853, 0.02090] | 0.3420 |
| `cppmax` | **+0.00099** | **0.01656** | [0.01568, 0.01751] | **0.2404** |

Paired differences (negative = the first build is closer to the C++ binary):

| pair | Δ mean \|error\| | 95% CI | verdict |
|---|--:|:--|:--|
| `v071 − v082` | +0.00036 | [−0.00050, +0.00124] | **not significant** |
| `cbrt64 − v071` | −0.00047 | [−0.00124, +0.00029] | not significant |
| `cppcbrt − v071` | −0.00133 | [−0.00237, −0.00034] | significant |
| `cppcbrt − v082` | −0.00098 | [−0.00196, −0.00004] | significant |
| `cppcbrtu − v071` | −0.00119 | [−0.00221, −0.00014] | significant |
| `cppmax − v071` | −0.00431 | [−0.00547, −0.00316] | **significant** |
| `cppmax − v082` | −0.00396 | [−0.00509, −0.00280] | **significant** |
| `cppmax − head` | −0.00401 | [−0.00519, −0.00280] | **significant** |
| `cppmax − cppcbrt` | −0.00298 | [−0.00401, −0.00193] | **significant** |

So: swapping in jpegli's cube root is a small but real gain (~5%); swapping in
jpegli's *horizontal blur form* on top of it is the large one. Together they cut
mean |Δ| from 0.0209 (0.7.1) / 0.0205 (0.8.2, 0.9.0) to 0.0166, halve the worst
case, and remove the systematic positive bias every shipped version carries.

By content and size, `cppmax` wins in every bucket, most at full size and on
flat content:

| bucket | `v071` | `v082` | `head` | `cppmax` |
|---|--:|--:|--:|--:|
| size 64 | 0.01181 | 0.01154 | 0.01095 | **0.01006** |
| size 256 | 0.02325 | 0.02324 | 0.02398 | **0.02099** |
| size full | 0.02755 | 0.02676 | 0.02675 | **0.01862** |
| jpeg_q90 | 0.02656 | 0.02735 | 0.02643 | **0.02213** |
| jpeg_q50 | 0.01555 | 0.01462 | 0.01422 | 0.01481 |
| jpeg_q10 | 0.00725 | 0.00682 | 0.00709 | **0.00600** |
| boxblur_r2 | 0.00697 | 0.00726 | 0.00705 | **0.00655** |
| noise_a8 | 0.01685 | 0.01572 | 0.01571 | **0.01309** |
| flat_jpeg_q90 | 0.03794 | 0.03462 | 0.03511 | **0.02957** |
| flat_noise_a2 | 0.03495 | 0.03721 | 0.03833 | **0.02374** |

## 4. Corrected: the repo's own `CubeRootAndAdd` transliteration

`src/cpp_parity_diag.rs::cpp_cube_root_and_add` grouped its FMAs the wrong way
round. Highway's `NegMulAdd(a, b, c)` is the fused `c − a*b`, so in
`NegMulAdd(xa_3, Mul(r2, r2), Mul(k4_3, r))` the product that escapes rounding
is `xa_3 * r4` while `k4_3 * r` rounds first; the old code fused `k4_3 * r`
instead. Same for the final `MulAdd(k1_3, NegMulAdd(xa, Mul(r2,r2), r), r)`.
The two spellings agree on 89.7% of the domain and are up to 3.6e-7 apart on the
rest. Fixed in this commit. Consequence for the record: jpegli's cube root
measures **2.62 ulp** max error over the opsin domain, not the 3.34 ulp quoted
before (that figure came from the mis-associated form).

## 5. What this does *not* say

- **x86_64 and wasm128 are still not measured.** This was an aarch64 host, so
  every "fused" row here means NEON `fmla`. On a target whose `magetypes`
  `mul_add` is the two-rounding polyfill (wasm128, the scalar tier), the jpegli
  transliteration is not bit-exact with the C++ either — it is FMA-shaped by
  construction, which is precisely the arch-consistency hazard 0.9.0's unfused
  opsin matmul was written to avoid. Adopting it means either accepting a
  per-target difference again, or spending a correctly-fused software `fma` on
  the tiers that lack hardware FMA (Rust's `f32::mul_add` *is* correctly fused
  everywhere — it is only the `magetypes` SIMD polyfill that is not).
- **The cube-root half is no longer unmeasured — and it is faster.** See
  [`cbrt_perf_2026-09-09.md`](cbrt_perf_2026-09-09.md): jpegli's
  `CubeRootAndAdd` beats the shipped cube root by 2.7-2.9% (NEON) and 6-8%
  (AVX2, >=64K px) in the XYB kernel, because it iterates the *reciprocal* cube
  root and needs no divides. The blur half is still unmeasured.
- **`cppmax` is not a performance proposal.** Its horizontal blur is a scalar
  per-row transliteration run for every row, chosen so the *form* could be
  measured without also porting it to `magetypes`; its speed was not measured
  and is certainly worse than the shipped 8-rows-per-lane-group pass. Adopting
  the form for real means re-vectorising jpegli's 4-unrolled recurrence, which
  is a different piece of work from establishing that the form is right.
- **Landing either change re-pins the parity fixtures.**
  `tests/implementation_parity.rs::REAL_IMAGE_CASES` asserts four SIMD scores to
  1e-5, and its own message spells out the procedure: re-pin from one
  architecture only after confirming another agrees. So adopting jpegli's cube
  root is not a one-file change — it needs a second architecture in the loop,
  which is the same x86_64 gap listed above.
- **The remaining 0.0166 is still unattributed.** With both jpegli
  approximations in place the *bias* is gone, but the per-cell scatter is not.
  Candidates narrowed from 2026-08-31: `Downsample` for the pyramid levels, and
  the skcms linear→linear leg that runs after the sRGB `ExtraTF`. Two former
  suspects are ruled out:
  - **The sRGB transfer function is not one.** `jpegli_cms.cc:1188` routes
    sRGB→linear with matching primaries through `ExtraTF::kSRGB`, i.e.
    `TF_SRGB().DisplayFromEncoded` — the `af_cheb_rational` k=100 degree-4/4
    rational polynomial, Horner with `MulAdd`, threshold 0.04045 and `1/12.92`
    below it. That is what `input.rs::srgb_to_linear` mirrors, and all ten
    coefficients were checked **bit-identical as f32** against
    `lib/cms/transfer_functions-inl.h`.
  - **`intensity_target` is not one.** It is 255 for an 8-bit sRGB PNG, so
    `ComputePremulAbsorb`'s `mul = intensity_target / 255` is exactly 1.0 and
    the opsin matrix is unscaled.

## 6. Reproduce

```sh
# 1. cube-root study (exhaustive f32 sweep, ~20 s)
mkdir -p ~/tmp/cbrt-study/src && cp benchmarks/version_divergence_2026-09-09/cbrt_study_main.rs \
  ~/tmp/cbrt-study/src/main.rs   # + a two-line Cargo.toml, no dependencies
cd ~/tmp/cbrt-study && cargo run --release

# 2. build the variant trees out of the current HEAD, patch them, score the corpus
mkdir -p ~/tmp/ssim2-parity-study/trees
for v in v071repro cbrt64 cppcbrt cppcbrt_unfusedmm cppmax; do
  mkdir -p ~/tmp/ssim2-parity-study/trees/$v
  git archive HEAD | tar -x -C ~/tmp/ssim2-parity-study/trees/$v
done
# both scripts want to live in ~/tmp/ssim2-parity-study/ (patch_cppmax.py
# re-execs patch_variants.py from there, and both write into trees/ beside them)
cp benchmarks/version_divergence_2026-09-09/patch_*.py ~/tmp/ssim2-parity-study/
python3 ~/tmp/ssim2-parity-study/patch_variants.py
python3 ~/tmp/ssim2-parity-study/patch_cppmax.py
# driver_Cargo.toml -> ~/tmp/ssim2-parity-study/driver/Cargo.toml (its path
# deps are absolute — adjust the /Users/lilith prefix), driver_main.rs -> src/
cd ~/tmp/ssim2-parity-study/driver
SSIMULACRA2_BIN=/opt/homebrew/bin/ssimulacra2 PHOTO_LIMIT=24 cargo run --release

# 3. paired bootstrap over the TSV it writes
python3 benchmarks/version_divergence_2026-09-09/paired.py
```

Build without `-C target-cpu=native`; the C++ binary must be the same build for
every cell (a differently-vectorised `ssimulacra2` is a different reference —
see §2 of `cpp_parity_2026-08-31.md`).
