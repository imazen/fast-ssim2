# The horizontal blur fell off a cliff at power-of-two widths

**Date:** 2026-09-09 · **Repo commit (before):** `40e366c` ·
**Hosts:** Apple M4 Pro (aarch64, rustc 1.98.0) and Ryzen 9 7900X (Zen 4,
rustc 1.98.1) · **Bench:** `benches/blur_stride.rs` (public API) plus the
kernel-level probe in
[`version_divergence_2026-09-09/blur_perf_bench.rs`](version_divergence_2026-09-09/blur_perf_bench.rs).

Found while measuring whether jpegli's 4-unrolled horizontal Gaussian is faster
than ours ([`cbrt_perf_2026-09-09.md`](cbrt_perf_2026-09-09.md)). It is — but the
much larger effect was that **our** pass collapses at specific widths, and those
widths are the ones real images use.

## Symptom

Horizontal pass alone, ns per pixel, medians of three runs (all reproducible to
±0.01 ns/px):

| width | ours, x86 | jpegli, x86 | ours, aarch64 | jpegli, aarch64 |
|---|--:|--:|--:|--:|
| 1000 | 0.70 | 0.65 | 0.71 | 0.62 |
| **1024** | **5.34** | 0.65 | 0.71 | **2.74** |
| 1032 | 0.70 | 0.65 | 0.71 | 0.61 |
| 2040 | 0.70 | 0.65 | 0.71 | 0.61 |
| **2048** | **5.40** | 0.65 | 0.91 | 0.61 |
| 2056 | 0.70 | 0.65 | 0.71 | 0.61 |
| 4088 | 0.70 | 0.65 | 0.76 | 0.61 |
| **4096** | 0.74 | 0.64 | **3.89** | 0.61 |
| 4104 | 0.70 | 0.65 | 0.72 | 0.61 |

A 7.6× cliff on x86 at widths 1024 and 2048; a 5.5× cliff on aarch64 at 4096.
Every neighbouring width is unaffected, so this is not a size effect.

## Cause: 4 KiB congruence between the two planes

The horizontal pass runs the IIR over eight rows at once, one row per SIMD lane,
so each column access is eight loads at stride `width * 4` bytes. When `width`
is a power of two those eight addresses are congruent modulo 4096 — they all
map to the same cache set. That alone is survivable (L1 is 8-way). It stops
being survivable when the *destination* plane is congruent with the source too,
which is what happens whenever both are page-aligned mappings — i.e.
deterministically for planes of a few MB, which is exactly where it hurts.

Confirmed by moving the destination 256 bytes inside a larger allocation and
changing nothing else:

| width | ours | ours, destination +256 B | jpegli | jpegli, +256 B |
|---|--:|--:|--:|--:|
| x86 1024 | 5.34 | **0.74** | 0.65 | 0.64 |
| x86 2048 | 5.29 | **0.76** | 0.64 | 0.64 |
| x86 4096 | 3.17 | **0.74** | 0.65 | 0.64 |
| aarch64 4096 | 3.85 | **0.72** | 0.61 | 0.61 |
| aarch64 1024 | 0.71 | 0.71 | 2.74 | **0.62** |

Every cliff on both hosts, in both kernels, disappears. So none of them are
intrinsic costs of either algorithm — they are placement effects. Note the
asymmetry, though: ours touches eight rows at once, so it is exposed at any
power-of-two width; jpegli's touches one row contiguously, so only the
source/destination pair can collide, and on x86 it never did.

## It reached the public API

`compute_ssimulacra2` on synthetic pairs, paired A/B (both binaries built, then
run interleaved, three rounds each), medians:

### aarch64 (M4 Pro)

| case | before | after | |
|---|--:|--:|--:|
| 1000×1000 | 33.75 ms | 33.64 ms | −0.3% |
| 1024×1024 | 35.81 ms | 36.05 ms | +0.7% |
| 1032×1032 | 36.83 ms | 37.31 ms | +1.3% |
| 2040×1024 | 71.33 ms | 71.28 ms | −0.1% |
| 2048×1024 | 73.39 ms | 72.41 ms | −1.3% |
| 2056×1024 | 72.56 ms | 72.50 ms | −0.1% |
| 4088×512 | 74.26 ms | 73.41 ms | −1.1% |
| **4096×512** | **119.85 ms** | **77.82 ms** | **−35.1%** |
| 4104×512 | 73.85 ms | 73.39 ms | −0.6% |

Unaffected sizes move ±1.3% with both signs — drift, not effect.

### x86_64 (Ryzen 9 7900X)

| case | before | after | |
|---|--:|--:|--:|
| 1000×1000 | 96.09 ms | 96.51 ms | +0.4% |
| **1024×1024** | **118.12 ms** | **104.17 ms** | **−11.8%** |
| 1032×1032 | 101.11 ms | 100.93 ms | −0.2% |
| 2040×1024 | 201.29 ms | 200.20 ms | −0.5% |
| **2048×1024** | **246.34 ms** | **210.02 ms** | **−14.7%** |
| 2056×1024 | 218.14 ms | 217.96 ms | −0.1% |
| 4088×512 | 201.28 ms | 198.82 ms | −1.2% |
| **4096×512** | **228.15 ms** | **199.45 ms** | **−12.6%** |
| 4104×512 | 209.78 ms | 209.83 ms | 0.0% |

Same shape as aarch64: every power-of-two width recovers, every other width
moves inside ±1.2%. Before the fix, 2048×1024 cost **22.5% more** than
2040×1024 for 0.4% more pixels; after it, 4.9%.

## The fix

`SimdGaussian::blur_single_plane_into` writes the horizontal result into
`self.temp_buffer`, an *internal* buffer. It now carries
`TEMP_DEALIAS_SLACK = 1024` spare floats, and `temp_offset()` picks the start
index so the temp plane sits 256 bytes past the source plane's own position
within a 4 KiB page:

```rust
let want  = (plane_addr + TEMP_DEALIAS_BYTES) % PAGE;
let have  = temp_addr % PAGE;
let delta = (want + PAGE - have) % PAGE;   // floats: delta / 4
```

Computed from the actual pointers, so it does not depend on the allocator
handing back page-aligned blocks; and 256 is a multiple of 64, so the temp
plane keeps whatever 64-byte alignment the source plane has.

**This moves where the intermediate lives, never what it contains.** Scores are
bit-identical: `implementation_parity` (four real-image SIMD scores pinned to
1e-5), `reference_parity` and the other 77 tests pass unchanged.

## What this does not fix

- **The residual at power-of-two widths.** After the fix, x86 1024×1024 sits at
  104.17 ms against 100.93 ms for 1032×1032 (+3.2%), and 4096×512 at 199.45 ms
  against 198.82 ms for 4088×512 (+0.3%). Something else is still mildly
  congruent — most likely the caller-owned plane pairs in `lib.rs` (`mul` →
  `sigma1_sq` and friends, each its own `Vec<f32>` of exactly `width * height`).
  The same trick would apply there, but those buffers are allocated in the
  metric, not the blur.
- **Dodging the destination plane as well.** `temp_offset` was extended to skip
  an offset that collides with `out` too, on the theory that the vertical pass
  (which reads temp and writes `out`) could inherit the cliff. It measured no
  difference at any size on either host, so it was reverted rather than kept as
  unjustified complexity. **Re-tested 2026-09-09** after the horizontal pass was
  replaced with jpegli's row-contiguous form — which removes the eight-row
  gather entirely, so it was the natural second suspect — and it *still* made no
  difference (33.02 vs 33.00 ns/px at width 1024; 35.14 vs 35.25 at 4096).
  Reverted again.
- **The residual survived the kernel swap.** With jpegli's horizontal pass in
  place, width 1024 still costs +2.4% against 1032 and 4096 +7.5% against 4104
  end-to-end. Since that pass no longer gathers across rows, whatever is left is
  not the blur's access pattern: the remaining suspects are the ~24 plane buffers
  the metric allocates in `lib.rs` (`mul`, `sigma1_sq`, `sigma2_sq`, `sigma12`,
  `mu1`, `mu2`, two planar copies — three channels each), which are plain
  `Vec<f32>` of exactly `width * height` and therefore mutually page-congruent at
  those sizes. Chasing it further needs profiling to find *which* pair collides,
  not another guess; both guesses so far measured zero.
- **jpegli's cliff at width 1024 on aarch64.** It is in the study kernel, not in
  anything fast-ssim2 ships; recorded because it shows the effect is about
  placement rather than about which kernel is "better".
- **Any width the probe did not cover.** The sweep is 248/256/264, 504/512/520,
  1000/1024/1032, 2040/2048/2056, 4088/4096/4104. Widths 256 and 512 showed no
  cliff on either host, which is worth knowing because SSIMULACRA2's pyramid
  halves the width at every scale — a 4096-wide input becomes 2048, 1024, 512,
  256, 128, so the deeper scales were never the problem.
