# Streaming the blur through a ring buffer: a negative result

> **Not merged, and should not be re-attempted** without first refuting the
> 8K measurement below. Kept on branch `fused-blur`.

**Branch:** `fused-blur` (not merged) · **Base:** `e4df4dc` · **Date:** 2026-09-10 ·
**Measured on:** Ryzen 9 5900XT (Zen 3, 16C/32T, 64 MiB L3, idle).

## The idea, and why it looked good

The blur ran as two full passes with a whole plane of horizontal output between
them: written once, read back once. At 4K that is 33 MiB each way, per channel,
per blur, five blurs per scale — about 995 MiB of the ~2289 MiB a 4K scale
moves.

But the vertical recurrence at output row `n` reads horizontal rows `n−RADIUS−1`
and `n+RADIUS−1`, and `RADIUS` is 5, so **at most 11 rows are ever live**.
Producing horizontal rows just ahead of the vertical pass and keeping them in a
16-row ring replaces the 33 MiB intermediate with 245 KiB. Implemented, and
bit-identical by construction — same row function, same recurrence, same order;
the pinned `implementation_parity` scores pass unchanged.

## The measurement: it does nothing, and costs parallelism

Paired, interleaved, on an idle box:

| | main | fused | |
|---|--:|--:|--:|
| 4K single-threaded | 1069.6 / 1075.0 / 1061.1 ms | 1068.0 / 1070.8 / 1068.6 ms | **±0%** |
| 4K multi-threaded | 706.6 / 711.6 ms | 1026.7 / 1024.8 ms | **+45%** |
| 8K single-threaded | 4231.0 / 4248.7 ms | 4203.7 / 4200.2 ms | −1% |
| 1024×1024 single-threaded | 122.8 / 124.0 ms | 124.5 / 124.5 ms | +1% |

The 8K row is the one that settles it. A 133 MiB plane cannot sit in a 64 MiB
L3, so if the intermediate round trip were costing DRAM traffic, that is where
removing it would pay. It buys 1%.

**Why the traffic model was wrong:** the intermediate is written sequentially
and read back sequentially. That is the access pattern hardware is best at —
streaming stores, then a linear read a prefetcher predicts perfectly. Removing
995 MiB of *sequential, prefetchable* traffic is worth about nothing. Bytes
moved is the wrong cost model; access *pattern* and *latency* are the right ones.

**And it costs the horizontal pass's parallelism.** In `main` the horizontal
pass is row-parallel — every row at once. Fused, rows are produced on demand
inside the vertical loop, which is inherently sequential, so with `rayon` the
whole horizontal stage becomes serial. Hence +45% MT. That was foreseeable and
was foreseen; it was worth measuring anyway because a large enough ST win would
have justified restoring the parallelism differently (produce rows in parallel
batches). There is no ST win to justify it.

## A correction to the record

[`vs_cpp_and_mt_2026-09-10.md`](vs_cpp_and_mt_2026-09-10.md) and the fleet
comparison concluded from this measurement —

> Eight concurrent single-threaded 4K runs on a 16-core Zen 3 box took 3.05×
> as long each as one run alone: 8 cores bought 2.6× throughput. The metric is
> strongly memory-bandwidth bound.

— that a *single* run is bandwidth-bound. **It is not.** A single 4K run moves
roughly 3 GiB in 1.06 s, about 2.8 GiB/s, which is a few percent of what that
machine can stream. What the 8-instance result actually shows is that eight
concurrent runs, each with an ~800 MiB working set, destroy each other's L3
residency — a *capacity* effect that governs batch throughput, not single-image
latency.

Both facts are useful, but they point at different work:

- **Batch/throughput** (many images at once): the working set per instance is
  what matters. Fewer live planes would help; fusing *across stages* — multiply
  → blur → map over a tile that stays resident — is the version of the locality
  idea that has not been refuted.
- **Single-image latency** (what the CLI does): compute- and latency-bound, so
  the levers are the serial stages (Amdahl), wider SIMD where the hardware has
  it, and doing less arithmetic.

## Status

The branch is kept, unmerged, as the record of a plausible idea that measured
flat. Do not re-attempt the intermediate-plane fusion without first refuting the
8K number above.
