# How fast is this against the C++ tool, single- and multi-threaded

**Date:** 2026-09-10 · **Host:** Apple M4 Pro (aarch64, 12 cores), macOS 26.5.2 ·
**Reference:** `/opt/homebrew/bin/ssimulacra2` (jpeg-xl 0.12.0) ·
**Ours:** `fast-ssim2-cli 0.6.1` at `dc3c415`+ · `hyperfine`, 3 warmups, 10 runs.

## Single-threaded, CLI against CLI

Both binaries do the same job: read two PNGs, print a score. Eight sizes from a
Lanczos-resized CLIC photo, distorted by a JPEG q75 round trip.

| size | pixels | C++ | ours | speedup |
|---|--:|--:|--:|--:|
| 64×86 | 5.5k | 7.08 ms | 1.46 ms | **4.83×** |
| 128×173 | 22k | 8.88 ms | 2.68 ms | 3.32× |
| 256×347 | 89k | 14.48 ms | 6.14 ms | 2.36× |
| 512×695 | 356k | 36.44 ms | 19.71 ms | 1.85× |
| 768×1043 | 801k | 72.51 ms | 42.60 ms | 1.70× |
| 1024×1391 | 1.42 MP | 124.19 ms | 74.16 ms | 1.67× |
| 1536×2087 | 3.21 MP | 300.19 ms | 178.27 ms | 1.68× |
| 2048×2783 | 5.70 MP | 560.82 ms | 314.90 ms | 1.78× |

Two regimes, and they should be quoted separately:

- **Per-pixel**, from the two largest sizes where the linear term dominates:
  C++ 104.5 ms/MP, ours 54.8 ms/MP — **1.9× faster**.
- **Fixed cost**, read off the smallest size rather than from a least-squares
  intercept (a fit across three orders of magnitude is dominated by the large
  end and puts α near zero for both, which the 64×86 row plainly contradicts):
  C++ ≈ 7 ms, ours ≈ 1.4 ms. That is what makes small images 3–5× rather than
  1.9×, and it matters for anyone scoring thumbnails in bulk.

Both figures include PNG decode and colour conversion, because that is what a
user actually runs. For context, the library alone scores 1920×1080 in 81.8 ms
single-threaded (`cargo bench`), i.e. ~39 ms/MP of the 54.8.

## Multi-threaded

**The C++ tool has no multi-threaded path.** `tools/ssimulacra2.cc` calls
`ToXYB(c_desired, intensity_orig, nullptr, nullptr, …)` — the fourth argument is
its `ThreadPool*`, and it is null, so `RunOnPool` runs inline. Every number in
the table above is single-threaded on both sides, and there is no C++ MT number
to compare against.

On our side `rayon` is an off-by-default feature, and the CLI does not expose it
at all, so today's users get the single-threaded numbers above. What the feature
buys, measured on the library benchmarks (12 cores):

| case | 1 thread | rayon, before this work | rayon, now |
|---|--:|--:|--:|
| `ssimulacra2_320x240` | 2.99 ms | 5.29 ms (**0.57×**) | 2.99 ms (1.00×) |
| `ssimulacra2_1920x1080` | 81.76 ms | 68.76 ms (1.19×) | 54.75 ms (**1.49×**) |
| `ssimulacra2_3840x2160` | 328.08 ms | 253.89 ms (1.29×) | 202.93 ms (**1.62×**) |
| `ssimulacra2_rgb_1920x1080` | 70.19 ms | 56.03 ms (1.25×) | 43.04 ms (**1.63×**) |
| `ssimulacra2_rgb_3840x2160` | 282.27 ms | 208.55 ms (1.35×) | 155.72 ms (**1.81×**) |
| `blur` | 4.94 ms | 2.85 ms (1.73×) | 2.55 ms (1.94×) |

1.6–1.8× on twelve cores is still poor. The reason is not the cache.

## Why MT scales badly: Amdahl, not locality

Stage shares, measured at 1024×1024 (`examples/profile_simd`, `benchmark_simd`):
the metric takes 36.2 ms and one 3-plane blur takes 3.15 ms. The pyramid runs
five blurs per scale over six scales, which is ~21 ms — **blur is ~58% of the
metric**, and the horizontal/vertical split inside it is roughly 56/44.

Before this work, the only parallel stage was the horizontal blur pass: ~32% of
runtime. Amdahl with p = 0.32 on 12 cores caps at 1.42×; measured 1.29×. The
model and the measurement agree, which is the point — there was no cache mystery
to solve, just 68% of the work running on one core.

**The locality hypothesis was tested directly and is wrong for this shape of
work.** The crate already has a strip walker whose entire purpose is bounded
memory and locality. At 4K it is *slower* than the full-image path — 0.41 s at
512-row strips, 0.66 s at 128-row strips, against 0.32 s full-image — because
each strip re-runs the pyramid with halo rows, and that costs more than the
locality saves. Locality is not free here; it is paid for in redundant work.

## What was changed, and what is left

Three stages gained bit-identical parallel paths (the pinned
`implementation_parity` scores and `simd_consistency` pass with `rayon` on and
off):

- **XYB conversion** — pixels are independent, so it splits by chunks that are
  multiples of the vector body's lane count. No reduction, nothing to reorder.
- **`image_multiply`** — elementwise, split per channel.
- **`ssim_map`** — split per channel. Each channel owns its `f64` accumulators
  and its own output slots, so no summation order changes. This one barely moved
  the needle (207.4 → 205.8 ms at 4K); it is a small share of runtime.

And two overhead fixes, which turned out to matter more than the third stage:

- **A size threshold** (`PAR_MIN_SAMPLES`, 2¹⁸ samples). The pyramid shrinks 4×
  per scale, so even a 4K input spends its later scales on planes where a
  `rayon` split costs more than it saves.
- **Coarser blur tasks.** The horizontal pass was splitting *per row* — a 320-
  wide row is about a microsecond of work, so 240 rows meant 240 joins. It now
  chunks into `rows / (threads * 4)` groups. Together these removed the
  small-image regression entirely: 320×240 went 5.93 ms → 2.99 ms, exactly
  matching the single-threaded time instead of doubling it.

**Still serial: the vertical blur pass**, ~26% of runtime and now the largest
remaining block. Its columns are fully independent — the IIR state is per-column
— so it is parallel in principle, but each worker would write a *column band*,
i.e. a strided region of the output plane, which safe Rust cannot hand out as
disjoint `&mut` slices. The options are a per-band staging buffer (one extra
plane of memory, plus a scatter) or restructuring the pass. That is a real
memory-versus-parallelism decision, not a mechanical change, so it is written
down rather than guessed at.

If it were parallelised, p would reach ~0.58 and the 12-core ceiling would move
from ~1.9× to ~2.3×; getting past that needs the reduction kernels to split by
rows too, which requires a deterministic tree reduction to stay bit-identical.

## Reproduce

```sh
# single-threaded, CLI vs CLI
hyperfine --warmup 3 --runs 10 \
  "/opt/homebrew/bin/ssimulacra2 src.png dis.png" \
  "target/release/fast-ssim2-cli image src.png dis.png"

# multi-threaded
cargo bench -p fast-ssim2 --features rayon --bench benches

# stage shares
cargo run --release --example profile_simd
cargo run --release --example benchmark_simd

# the locality check
cargo run --release --example strip_memory full  3840 2160
cargo run --release --example strip_memory strip 3840 2160 512
```
