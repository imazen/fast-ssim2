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

### x86_64 (Ryzen 9 7900X) — **not measured**

The paired run was attempted and is being discarded rather than reported: that
box is shared, and a competing 100%-CPU job (`drv_r48`) ran throughout, with
load average 2.2–5.5 and seven logged-in sessions. Under `nice -n 19` the bench
was descheduled unpredictably, and the resulting deltas swung ±20% in *both*
directions on cases whose per-pixel work is identical — `3840x2160` +20.1% while
`rgb_3840x2160` −0.3%, `rgb_1920x1080` +16.0% while `1920x1080` −1.7%. That is
the box's other work, not this change.

The kernel-level x86 measurements it rests on were taken earlier on the same
host and *are* reproducible (three runs each, ≤1% spread): the cube root 6–8%
faster at ≥64K px ([`cbrt_perf`](cbrt_perf_2026-09-09.md)) and the horizontal
blur 4–7% faster ([`blur_stride`](blur_stride_2026-09-09.md)). An end-to-end x86
figure needs a quiet box; until then, treat the aarch64 table as the measured
one and x86 end-to-end as unquantified.

**Correctness on x86 is measured**, and separately from timing: the re-pinned
`implementation_parity` fixtures pass there, `simd_consistency` passes there,
and all 28 `arch_scores` values are bit-identical to the aarch64 run.

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
