# SSIMULACRA2 perf work — 2026-05-21 (Kanetaka IWAIT 2026 lossless techniques)

Benchmark notes for the three landed changes:
1. `bc9d011 perf: hoist SIMD blur state vectors out of vertical_pass_inner`
2. `75e3234 perf: skip per-channel SSIM/edge-diff cells with zero weights`
3. `c419b3d perf: add CompareContext for zero-alloc compare_with`

All three are bit-identical to the prior path — the reference parity test
(`tests/reference_parity.rs`, 66 C++-reference comparisons across 64x64
synthetic and JPEG corpora) passes unchanged.

## Host

AMD Ryzen 9 7950X, 128 GB RAM, Linux 6.6 / WSL2. Rust 1.93 release build.
Runtime SIMD dispatch enabled (no `target-cpu=native` — measures what users
actually get).

## Harness

`examples/precompute_benchmark.rs` — 20 iters per cell, three paths timed:

- **Full** — `compute_frame_ssimulacra2(source, distorted)` (allocates everything every call).
- **compare** — `Ssimulacra2Reference::compare(distorted)` (allocates working buffers per call).
- **compare_with** — `Ssimulacra2Reference::compare_with(&mut ctx, distorted)` (zero-alloc after the first call).

## Baseline (parent commit `fd41c6c`, before any of these changes)

| size      | Full      | compare   |
|-----------|-----------|-----------|
| 256x256   |   7.53 ms |   5.23 ms |
| 512x512   |  36.03 ms |  24.51 ms |
| 1024x1024 | 146.84 ms |  99.39 ms |

## After all three commits (HEAD `c419b3d`)

Median of two 20-iter runs.

| size      | Full      | compare   | compare_with |
|-----------|-----------|-----------|--------------|
| 256x256   |   7.77 ms |   6.05 ms |     4.19 ms  |
| 512x512   |  33.44 ms |  25.38 ms |    20.48 ms  |
| 1024x1024 | 148.57 ms | 107.62 ms |    90.35 ms  |
| 1920x1080 | 279.77 ms | 207.41 ms |   160.34 ms  |

## Speedups vs baseline `compare`

| size      | new compare_with | speedup |
|-----------|------------------|---------|
| 256x256   |          4.19 ms |  1.25×  |
| 512x512   |         20.48 ms |  1.20×  |
| 1024x1024 |         90.35 ms |  1.10×  |

(1920x1080 has no baseline number; the harness only went up to 1024x1024
before this PR.)

## Where the wins come from

- **CompareContext is the big lever for batch use cases.** Encoder RD
  search, simulated annealing, picker training: the buffers persist
  across calls and the per-call heap traffic disappears. 10-25% on top
  of the existing `compare`.
- **Skip-map** (`weights::SSIM_HAS_WEIGHT` / `EDGE_HAS_WEIGHT`) drops
  3 SSIM and 2 edge-diff `(channel, scale)` cells at full 6-scale
  inputs. Modest single-digit % on `Full`; the biggest cells (X+B SSIM
  at full resolution) are among those dropped. Bit-identical because
  the dropped contributions multiply by zero downstream.
- **State hoisting** removes ~180 small `Vec<f32>` allocations per
  frame (6 IIR-state vectors × 5 blurs × 6 scales) into a single
  pre-allocated buffer split in place. Most visible on the Full path
  at large sizes; small at 256x256 where the state vectors were tiny.

## Why no FIR

Kanetaka et al.'s Technique 1 — replacing the recursive IIR Charalampidis
blur with a separable 5-tap FIR — produces *different* per-image scores
(by design; the FIR has narrower effective impulse-response support).
The paper documents this in Table 2 on CID22: D=5 hits SROCC 0.890387
vs the libjxl IIR's 0.889297, but per-image values diverge. Out of
scope for this lossless pass; revisit if/when a fast-variant API
surface is acceptable.
