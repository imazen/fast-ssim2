# Parallelising the vertical blur with pre-sliced column bands

> **Not merged.** The technique is sound and bit-identical, but the fleet
> measurement is split: it helps an M4 Pro (−21%) and WSL2 (−13%) and hurts
> four other x86 machines (r7900x +25%, r5900xt +16%, i265 +14%, dev +11%).
> Kept on branch `vertical-band-blur` with all numbers. See
> [`fleet_4k_2026-09-10.md`](fleet_4k_2026-09-10.md).

**Branch:** `vertical-band-blur` · **Base:** `520005f` (main) · **Date:** 2026-09-10 ·
**Host:** Apple M4 Pro (aarch64, 12 cores).

[`vs_cpp_and_mt_2026-09-10.md`](vs_cpp_and_mt_2026-09-10.md) left the vertical
blur pass — ~26% of the metric and the largest remaining serial block — with
this claim:

> Its columns are fully independent … but each worker would write a *column
> band*, i.e. a strided region of the output plane, which safe Rust cannot hand
> out as disjoint `&mut` slices. The options are a per-band staging buffer (one
> extra plane of memory, plus a scatter) or restructuring the pass.

**That was wrong, and the correction is the whole point of this branch.** Safe
Rust hands out exactly those slices, with no staging buffer and no extra plane:
split each *row* with `split_at_mut` and group the pieces by band.

```rust
let mut band_rows: Vec<Vec<&mut [f32]>> = …;
for row in output.chunks_exact_mut(width) {
    let (mut rest, _tail) = row.split_at_mut(covered);
    for (b, g) in band_groups.iter().enumerate() {
        let (head, tail) = rest.split_at_mut(g * LANES);
        band_rows[b].push(head);
        rest = tail;
    }
}
```

`height` pointer pairs per band — about 26 000 for a 4K plane, built once per
plane blur, which measured 0.07% of runtime. The input plane needs none of this;
it is shared immutably. ([`rav1d-disjoint-mut`](https://crates.io/crates/rav1d-disjoint-mut)
is the other way to solve it, with runtime-tracked disjointness; it is not
needed here because the disjointness is static.)

The bands are **not sliding windows**. The vertical IIR's state is per column,
so a band of columns depends on nothing outside itself — no halo, no overlap,
nothing to stitch afterwards. Bands are whole `LANES`-wide groups, so the vector
body is bit-identical to the unsplit one however the columns are divided, and
one band *is* the serial path — the same code runs single- and multi-threaded,
so the two cannot drift apart (the mistake that cost a 1.5e-3 backend split when
the horizontal pass was swapped).

## Result: multi-threaded

Library benchmarks, `--features rayon`, 12 cores:

| case | 1 thread | main (`520005f`) | this branch |
|---|--:|--:|--:|
| `ssimulacra2_320x240` | 2.99 ms | 2.99 ms (1.00×) | 2.98 ms (1.00×) |
| `ssimulacra2_1920x1080` | 81.76 ms | 54.75 ms (1.49×) | **47.53 ms (1.72×)** |
| `ssimulacra2_3840x2160` | 328.08 ms | 202.93 ms (1.62×) | **171.45 ms (1.91×)** |
| `ssimulacra2_rgb_1920x1080` | 70.19 ms | 43.04 ms (1.63×) | **36.81 ms (1.91×)** |
| `ssimulacra2_rgb_3840x2160` | 282.27 ms | 155.72 ms (1.81×) | **126.66 ms (2.23×)** |
| `blur` | 4.94 ms | 2.55 ms (1.94×) | **2.15 ms (2.30×)** |

Bit-identical: the pinned `implementation_parity` scores and `simd_consistency`
pass with `rayon` on and off, and the whole suite (77 tests) is green in both
configurations. `clippy --all-targets` is clean in both.

This is roughly what the Amdahl model predicted. With the vertical pass folded
in, the parallel fraction goes from ~0.58 to ~0.84; the remaining serial work is
the downscale between scales, the edge-diff map, and the per-scale glue.

## The x86 MT regression, and a hypothesis that measured false

The first version of this branch cut bands on whole `LANES` groups — 8 floats,
**32 bytes**. On the M4 Pro that measured 23% faster than main at 4K, which
looked like a finished result. On a Ryzen 9 5900XT it was a disaster:

| MT (`rayon`), r5900xt | main | 32-byte-aligned bands | |
|---|--:|--:|--:|
| `ssimulacra2_1920x1080` | 270.70 ms | 441.86 ms | **+63%** |
| `ssimulacra2_rgb_1920x1080` | 370.83 ms | 529.26 ms | +43% |
| `ssimulacra2_3840x2160` | 1130.82 ms | 1386.27 ms | +23% |
| `ssimulacra2_rgb_3840x2160` | 1517.74 ms | 1773.82 ms | +17% |

Not just slower than main — **slower than its own serial path** (0.67× at
1920×1080, i.e. turning threads on made it worse).

**First hypothesis: false sharing at the band edges.** A `LANES` group is 8
floats = 32 bytes, so an odd group count puts two bands in one 64-byte line and
their two threads write it once per row. Bands were changed to carry an even
number of groups, putting every boundary on a line.

**Measured: it changed nothing.** Zen 3 with 64-byte-aligned bands is +63.3% at
1920×1080 against main's +63.2% — the same number. The hypothesis was wrong, and
it is recorded here as wrong rather than quietly dropped. (aarch64 was unaffected
by the alignment either way, so it could not adjudicate.) The alignment is kept
because it is free and correct in principle, but it is not the cause.

**The actual cause: band geometry.** Each band walks the full height, reading a
narrow vertical strip — `band_columns * 4` bytes out of every `width * 4` byte
row. With 32 hardware threads the branch cut 32 bands, so at 1920×1080 each band
was 240 bytes of a 7680-byte row: one sequential pass over the plane became 32
strided ones. Forcing the band count on the Zen 3 box, MT against main's MT:

| bands | 1920×1080 | 3840×2160 | bytes/row @1080p |
|---|--:|--:|--:|
| 2 | +2% | −7% | 3840 |
| 4 | **−7%** | −11% | 1920 |
| 8 | −4% | **−12%** | 960 |
| 32 | **+64%** | +23% | 240 |

One band per hardware thread — the obvious policy — lands squarely on the last
row. On aarch64 the same sweep is flat (47.5 ms at 2 bands, 48.3 at 32), so the
M4 Pro could not have found this at any band count.

## The policy: a width floor, not a thread count

`MIN_GROUPS_PER_BAND = 64` — 512 columns, **2 KiB of each row** — then capped by
the thread count. 1080p gets 3 bands, 4K gets 7; a plane too narrow to give one
band that much simply runs serial, which the `PAR_MIN_SAMPLES` floor already did
for small planes.

Verified on both architectures. Zen 3 (r5900xt, idle, two interleaved rounds),
MT against main's MT:

| case | main | width-governed | |
|---|--:|--:|--:|
| `ssimulacra2_3840x2160` | 1129.33 ms | 993.81 ms | **−12.0%** |
| `ssimulacra2_rgb_3840x2160` | 1517.55 ms | 1379.61 ms | **−9.1%** |
| `ssimulacra2_1920x1080` | 271.70 ms | 254.85 ms | −6.2% |
| `ssimulacra2_rgb_1920x1080` | 371.44 ms | 355.30 ms | −4.3% |
| `ssimulacra2_320x240` | 9.14 ms | 9.04 ms | −1.1% |
| `ssimulacra2_rgb_320x240` | 9.93 ms | 10.09 ms | +1.6% |
| `blur` (isolated kernel) | 5.71 ms | 6.34 ms | **+11.1%** |

M4 Pro, MT against its own serial: `3840x2160` 328.08 → **168.07 ms (1.95×)**,
`rgb_3840x2160` 282.27 → **124.26 (2.27×)**, `blur` 4.94 → **1.99 (2.48×)**.

**The one regression is real and is not being hidden.** The isolated `blur`
bench is 11% slower on Zen 3. It blurs a 1024-wide plane, which the width floor
gives only 2 bands, and the sweep says 8 would be better there. The floor is
tuned for the metric — where the large scales get enough bands and the small
ones fall back to serial — and the isolated kernel pays for that. Tuning the
floor per plane size would fix it; it has not been attempted, because the metric
is what ships and a second knob wants its own sweep on both machines.

## Result: single-threaded — settled on a quiet box

The serial path now walks `out_rows[n]` instead of computing a flat offset. The
row lookup is hoisted out of the group loop, so it costs one pointer load per
row rather than per 8 columns, and the pre-slice itself is 0.07% of runtime. It
*should* be free. The measurement does not say so cleanly:

| case | main | bands | delta |
|---|--:|--:|--:|
| `blur` | 4.99 ms | 5.25 ms | +5.3% |
| `ssimulacra2_1920x1080` | 85.19 ms | 87.30 ms | +2.5% |
| `ssimulacra2_320x240` | 3.22 ms | 3.19 ms | −1.0% |
| `ssimulacra2_3840x2160` | 363.25 ms | 352.37 ms | −3.0% |
| `ssimulacra2_rgb_1920x1080` | 77.46 ms | 73.75 ms | −4.8% |
| `ssimulacra2_rgb_320x240` | 2.78 ms | 2.71 ms | −2.4% |
| `ssimulacra2_rgb_3840x2160` | 306.25 ms | 309.46 ms | +1.0% |

Five interleaved rounds on the laptop, both binaries built and run alternately.
The deltas straddle zero with **mixed signs on code paths that differ only in
this one indirection** — `3840x2160` −3.0% while `rgb_3840x2160` +1.0%. That is
not a result; it is noise, and the *absolute* numbers drifted ~10% across the
session (4K main measured 328 ms early, 363 ms by then) after hours of
continuous benchmarking.

Re-run on `r5900xt` (idle, load ~0), two interleaved rounds, it resolves
cleanly — **the serial path does not regress**:

| case | main | bands | |
|---|--:|--:|--:|
| `blur` | 6.90 ms | 6.66 ms | −3.5% |
| `ssimulacra2_1920x1080` | 298.97 ms | 296.74 ms | −0.7% |
| `ssimulacra2_3840x2160` | 1252.49 ms | 1244.36 ms | −0.6% |
| `ssimulacra2_rgb_1920x1080` | 384.40 ms | 383.15 ms | −0.3% |
| `ssimulacra2_rgb_320x240` | 9.78 ms | 9.75 ms | −0.4% |
| `ssimulacra2_rgb_3840x2160` | 1627.26 ms | 1635.65 ms | +0.5% |
| `ssimulacra2_320x240` | 9.03 ms | 9.09 ms | +0.6% |

Everything inside ±1%, and `blur` is faster. That matches what the structure
predicts: the pre-slice is 0.07% of runtime and the row lookup is hoisted out of
the group loop, so one band costs a pointer load per row rather than a flat
offset computation.

## Reproduce

```sh
cargo bench -p fast-ssim2 --features rayon --bench benches   # MT
cargo bench -p fast-ssim2 --bench benches                    # ST
cargo test -p fast-ssim2 --features rayon                    # bit-identity
```
