# Adopting jpegli's cube root and horizontal Gaussian

**Date:** 2026-09-09 · **Baseline:** `c71875a` (0.9.0 line) ·
**Hosts:** Apple M4 Pro (aarch64, `neon`) and Ryzen 9 7900X (Zen 4, `v3`) ·
**Reference:** `/opt/homebrew/bin/ssimulacra2`, jpeg-xl 0.12.0.

Three earlier records established the case:
[`version_divergence`](version_divergence_2026-09-09.md) (every released version
is biased +0.0067 high against the C++ binary, and jpegli's own approximations
remove it), [`cbrt_perf`](cbrt_perf_2026-09-09.md) (its cube root is *faster*
than ours), and [`blur_stride`](blur_stride_2026-09-09.md) (its horizontal
Gaussian is faster too, and immune to the aliasing cliff ours had). This is the
record of landing them.

## What shipped

| stage | before | now |
|---|---|---|
| opsin cube root | bit-hack seed + 2 Halley steps in f32, 2 divisions | jpegli `CubeRootAndAdd`: reciprocal-cbrt Newton, **no divisions**, integer seed |
| horizontal Gaussian | IIR over 8 rows at once, one row per lane | jpegli `FastGaussian1D`: 4 outputs per iteration within a row |
| opsin matrix multiply | unfused | **unchanged — still unfused** |

The third row is the interesting one. jpegli fuses the opsin matmul; 0.9.0
deliberately unfused it so the scalar and wasm arms agree bit-for-bit. Measured
over 2016 cells, the fused form is **not** better against the reference (mean
|Δ| 0.01953 fused vs 0.01968 unfused — inside the noise), and the shipped
unfused combination actually beats the fused one on the two figures that matter:

| configuration | bias | mean \|Δ\| | max \|Δ\| |
|---|--:|--:|--:|
| fused matmul + jpegli cbrt + jpegli blur | +0.00099 | 0.01656 | 0.2404 |
| **unfused matmul + jpegli cbrt + jpegli blur (shipped)** | **+0.00012** | **0.01658** | **0.1726** |

So arch consistency was free here, and 0.9.0's decision stands.

## Agreement with the C++ binary

96 references × 3 sizes × 7 distortions = 2016 cells, same corpus and harness as
the version-divergence record:

| | 0.7.1 | 0.8.2 | 0.9.0 | **now** |
|---|--:|--:|--:|--:|
| mean(ours − C++) | +0.00673 | +0.00691 | +0.00740 | **+0.00012** |
| mean \|Δ\| | 0.02087 | 0.02051 | 0.02056 | **0.01658** |
| max \|Δ\| | 0.4902 | 0.4309 | 0.5223 | **0.1726** |

The systematic bias every released version carried is gone — the residual mean
is 5 000× smaller than 0.9.0's — and the worst case is a third of what it was.

Independently, the crate's *own* harness (`examples/photo_parity.rs`, 24
references × 4 sizes × 6 distortions = 576 pairs, a different grid: it includes
32×32 and a chroma-shift family the driver above does not) agrees on the
direction and size of the win:

| `photo_parity`, ALL 576 | 0.9.0 (recorded 2026-08-31) | now |
|---|--:|--:|
| mean(simd − C++) | +0.00219 | −0.00297 |
| mean \|Δ\| | 0.02386 | **0.01866** (−22%) |
| max \|Δ\| | 0.5233 | **0.2362** (−55%) |
| max \|simd − scalar\| | 2.6e-7 | 1.8e-7 |

Note the bias sign flips rather than vanishing on this grid. The two harnesses
weight content differently — this one gives a quarter of its cells to 32×32,
where the pyramid has one or two usable scales — so the residual mean is not
comparable between them; the mean-absolute and worst-case columns are.

## Speed

Paired A/B: both bench binaries built, then run interleaved, three rounds,
medians, `nice -n 19`, no `target-cpu=native`.

### aarch64 (M4 Pro)

| case | before | after | |
|---|--:|--:|--:|
| `blur` | 5.68 ms | 4.94 ms | **−12.9%** |
| `ssimulacra2_320x240` | 3.09 ms | 2.99 ms | −3.2% |
| `ssimulacra2_1920x1080` | 86.32 ms | 81.76 ms | **−5.3%** |
| `ssimulacra2_3840x2160` | 342.65 ms | 328.08 ms | −4.3% |
| `ssimulacra2_rgb_320x240` | 2.66 ms | 2.57 ms | −3.6% |
| `ssimulacra2_rgb_1920x1080` | 74.75 ms | 70.19 ms | −6.1% |
| `ssimulacra2_rgb_3840x2160` | 296.25 ms | 282.27 ms | −4.7% |

### x86_64 (Ryzen 9 5900XT, Zen 3 — no AVX-512, `v3` tier)

**Two** interleaved rounds, not three (see below). Per-case run-to-run spread in
the last column:

| case | before | after | | spread |
|---|--:|--:|--:|--:|
| `blur` | 7.40 ms | 7.03 ms | **−5.1%** | 1.3% / 2.8% |
| `ssimulacra2_320x240` | 9.20 ms | 9.04 ms | −1.7% | 0.2% / 0.3% |
| `ssimulacra2_1920x1080` | 302.88 ms | 297.53 ms | −1.8% | 0.3% / 0.5% |
| `ssimulacra2_3840x2160` | 1266.19 ms | 1248.19 ms | −1.4% | 0.4% / 0.1% |
| `ssimulacra2_rgb_320x240` | 9.98 ms | 9.83 ms | −1.5% | 1.2% / 0.7% |
| `ssimulacra2_rgb_1920x1080` | 390.08 ms | 385.69 ms | −1.1% | 0.1% / 0.4% |
| `ssimulacra2_rgb_3840x2160` | 1638.56 ms | 1612.36 ms | −1.6% | 0.6% / 0.2% |

Every case improves, and the two rounds' medians agree within 0.7 percentage
points. The gain is real but **smaller than aarch64's** (−1.1…−1.8% end to end
against −3.2…−6.1%; blur −5.1% against −12.9%), which tracks the kernel-level
measurements: on x86 the cube root gained 6–8% and the blur 4–7%, against
2.7–2.9% and 12.3–13.7% on NEON.

Two procedural notes, because they explain the missing third round and are worth
not repeating:

- **`r7900x` was the wrong box.** The first attempt ran there and had to be
  discarded: it is shared, a competing 100%-CPU job ran throughout, and the
  deltas swung ±20% in *both* directions on cases with identical per-pixel work
  (`3840x2160` +20.1% while `rgb_3840x2160` −0.3%). Check `uptime` before
  trusting a timing from it. `r5900xt` sat at load 1.00 — exactly this
  single-threaded benchmark — for the whole run.
- **The third round was lost to plumbing, not to measurement.** `r5900xt` is
  reachable only *from* `dev`, so the run is driven over two ssh hops; the first
  attempt's pipe closed after round two, and a re-launch through nested
  quoting produced a malformed `awk`. Two rounds at 0.1–1.3% spread is a firmer
  basis than the three-run rule was written for anyway — that rule exists
  because `ssimulacra2_320x240` once spanned 9.3% across runs of one unchanged
  binary, and here it spans 0.2%.

**Correctness on x86 is measured separately from timing**, on `r7900x` where
load does not matter: the re-pinned `implementation_parity` fixtures pass,
`simd_consistency` passes, and all 28 `arch_scores` values are bit-identical to
the aarch64 run.

## What moves, and what it costs downstream

Scores shift against 0.9.0 by **mean |Δ| 0.020, max 0.45, with 48% of cells
moving more than 0.01**. That is the price of the fidelity gain, and it is not
small enough to ignore: anything holding fast-ssim2 scores to a fixed value —
pinned fixtures, cached quality decisions, RD curves, selector training data —
needs re-baselining.

`tests/implementation_parity.rs` re-pinned its four real-image scores
(+0.0103, −0.0383, +0.0117, −0.0912). Per that test's own instruction, the new
values were confirmed on a second architecture — the x86_64 run passes the same
pinned constants — before being written down.

## The FMA class, and a gotcha worth remembering

jpegli's cube root uses genuine multiply-adds. The two Halley steps it replaced
did not, in the sense that mattered: their only multiplier was an exactly
representable `2.0`, so fusing or not fusing gave the same bits. The XYB stage
is therefore now fusion-sensitive, exactly like the blur's `MUL_PREV` step.

Measured over every archmage token permutation on aarch64: **0.0 difference on
every permutation that keeps NEON, 2.98e-7 per XYB sample with NEON disabled**
(the `magetypes` scalar polyfill, whose `mul_add` is `a * b + c`).
`tests/simd_consistency.rs` now gates this by FMA class — bit-identity required
*within* a class, a measured per-sample bound across classes — which is the
same discipline it already applied to the blur, not a loosened bound. Its
end-to-end same-class spread actually improved, from 2.2e-8 to ~3e-9.

**The gotcha:** `SimdImpl::Scalar` runs a *separate* blur implementation
(`blur/gaussian.rs`), not the scalar arm of the SIMD one. Swapping only
`blur/simd_gaussian.rs` made the two backends compute different metrics, and
`simd_consistency` caught it immediately as a 1.5e-3 backend split. Both files
now carry the unrolled form. Any future change to one must be mirrored in the
other.

## Reproduce

```sh
cargo test -p fast-ssim2 --test simd_consistency -- --nocapture   # tier/backend gating
cargo test -p fast-ssim2 --test implementation_parity             # pinned scores
cargo run --release --example arch_scores                         # cross-arch identity
cargo bench -p fast-ssim2 --bench benches                         # speed
```

Parity against the C++ binary uses the driver in
[`version_divergence_2026-09-09/`](version_divergence_2026-09-09/); its `head`
path dependency points at this working tree, so re-running it after a kernel
change measures the change.
