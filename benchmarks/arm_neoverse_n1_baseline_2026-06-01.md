# fast-ssim2 — ARM Neoverse-N1 perf baseline + horizontal-blur vectorisation

Date: 2026-06-01
Box: Hetzner CAX31, Ampere Altra (Neoverse-N1), 8 cores / 16 GB
Toolchain: rustc 1.96, `RUSTFLAGS=-C target-cpu=neoverse-n1` (and runtime-dispatch
cross-check with `RUSTFLAGS` empty — numbers within noise of each other since the
NEON kernel is selected at runtime either way).
Crate: `fast-ssim2` v0.8.1, workspace `imazen/fast-ssim2`.

## Baseline (before)

CRITERION (`cargo bench -p fast-ssim2`, neoverse-n1):

| bench | time |
|---|---|
| ssimulacra2_320x240 | 20.5 ms |
| ssimulacra2_1920x1080 | 540.9 ms |
| ssimulacra2_3840x2160 | 2.17 s |
| ssimulacra2_rgb_320x240 | 12.2 ms |
| ssimulacra2_rgb_1920x1080 | 453.1 ms |
| ssimulacra2_rgb_3840x2160 | 1.91 s |
| blur (tank image) | 22.8 ms |

`benchmark_simd` example, Scalar vs SIMD (neoverse-n1):

- Full SSIMULACRA2 SIMD only **1.13–1.22×** faster than scalar.
- Blur-only SIMD only **1.1–1.4×** faster than scalar.

x86 7950X cross-check (runtime dispatch, no `target-cpu=native`):

- Full SSIMULACRA2 SIMD **3.5×** faster than scalar.
- Blur-only SIMD **7×** faster than scalar.

The ~3× ARM/x86 gap on the SIMD speedup ratio flagged the pathology: the
NEON path was barely beating scalar.

## Profiling (perf, SIMD-only driver, 1536×1536)

Self-time by symbol:

| % | symbol |
|---|---|
| 40.4% | `blur::Blur::blur_plane_into` (recursive-Gaussian blur) |
| 12.0% | `yuvxyb …::to_linear` (sRGB→linear; in yuvxyb, not ours) |
| 9.5% | `fast_ssim2::linear_rgb_to_xyb` (SIMD XYB; 12 `vdivq_f32`/chunk from 6 Halley FDIVs) |
| 3.2% | `edge_diff_map` |
| 2.7% | `compute_…_impl` (downscale etc.) |
| 2.4% | `image_multiply` |

`perf annotate` on the blur kernel showed the hottest cluster (~50%+ of the blur
self-time) was **scalar** single-lane code (`fmov s`, `ldur s`, scalar FMA) — the
**horizontal pass**, which ran a serial IIR recurrence one row at a time. The
vertical pass was already SIMD (8 columns per lane group); the horizontal pass was
not.

## Optimisation shipped — horizontal blur vectorised across rows

The recursive-Gaussian horizontal IIR is a serial recurrence *within* a row, but
fully independent *across* rows. Mirroring the across-columns trick the vertical
pass already uses, `horizontal_pass` now processes **8 rows per SIMD lane group**
(one row per lane), with the 6 IIR-state variables held in `f32x8` vectors and a
scalar remainder for `height % 8` leftover rows.

Per column position the same column is loaded from 8 consecutive rows (a manual
8-wide gather, 8 scalar loads at stride `width`) and the result stored back across
the 8 rows. The `left`/`right`/`n` bounds branches depend only on the column index,
so they are taken once per group, not per lane.

**Correctness — bit-identical.** The SIMD IIR replicates the scalar `horizontal_row`
operation order exactly (`sum*MUL_IN` rounded, then `-prev2`, then
`MUL_PREV.mul_add(prev, …)`; `MUL_PREV2 == -1`). Verified: the SIMD scores match the
pre-existing pinned SIMD values (`test_simd_scores_pinned_real_images`, strict
`< 1e-5`) to 6 decimals, i.e. zero change. All 56 tests pass on arm-big NEON
(incl. strip-processing parity) and on x86.

### Result (measured, neoverse-n1)

| bench | before → after | speedup |
|---|---|---|
| ssimulacra2_320x240 | 20.5 → 19.3 ms | +6.2% |
| ssimulacra2_1920x1080 | 540.9 → 503.0 ms | +7.5% |
| ssimulacra2_3840x2160 | 2.17 → 2.04 s | +6.4% |
| ssimulacra2_rgb_320x240 | 12.2 → 11.1 ms | +9.9% |
| ssimulacra2_rgb_1920x1080 | 453.1 → 428.3 ms | +5.8% |
| ssimulacra2_rgb_3840x2160 | 1.91 → 1.70 s | +12.4% |
| blur (tank image) | 22.8 → 19.4 ms | +17.5% |

`benchmark_simd` blur-only SIMD: +9–12% across 512²..3840². x86 unaffected (blur was
already well-vectorised there).

## Falsified (committed as finding)

**XYB Halley-iteration FDIV → Newton-Raphson reciprocal.** The XYB conversion's
cbrt-via-Halley does 6 vector divisions per 8-pixel chunk (`vdivq_f32`, the
non-pipelined N1 FDIV). Replacing `a / b` with `a * recip_nr(b)` (`vrecpeq_f32` seed
+ 2 or 3 Newton-Raphson FMA steps) was **slower** on N1 — the 3-step NR adds ~9 FMA
per site (×6 = 54 FMA/chunk) which costs more than the 12 `vdivq` it removes, because
the Halley loop already has enough ILP (3 channels interleaved) to latency-hide the
FDIVs — **and** less accurate (`|simd − f64-scalar|` at q90 rose 0.021 → 0.074, since
`vdivq_f32` is correctly-rounded and the NR reciprocal is not). Reverted; the `/`
stays.

## Next hypotheses (not yet done)

1. The horizontal-pass gather/scatter is now visible in the profile (8 scalar
   loads + 8 lane-extract stores per column). A transpose-tile approach (load
   contiguous, transpose in-register, contiguous store) could remove the strided
   access — worth measuring vs the current manual gather.
2. The vertical pass round-trips all 6 IIR-state arrays through memory every group
   every row (`to_array`/`from_array`); keeping per-group state register-resident
   across the height traversal (loop interchange) could cut the load/store traffic
   that dominates the remaining blur self-time.
