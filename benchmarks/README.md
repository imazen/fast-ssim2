# fast-ssim2 benchmarks — methodology & reproduction

How to measure fast-ssim2 fairly, and how to read the committed result files in
this directory. fast-ssim2's reason to exist is being fast — runtime SIMD
dispatch instead of a scalar reference path — so the numbers that matter are
(a) SIMD vs its own scalar path, (b) the warm batch path (`compare_with`) vs a
cold `compare()`, and (c) **score agreement** with the upstream SSIMULACRA2
implementations (the speed is only worth anything if the score is right).

## Fairness guarantees

- **No `-C target-cpu=native`.** Builds use runtime SIMD dispatch (archmage),
  which is what ships. Native builds bake in ISA extensions and give misleading
  numbers. The canonical figures use runtime dispatch; the ARM baseline below
  was cross-checked with `RUSTFLAGS` empty vs `target-cpu=neoverse-n1` and the
  two were within noise (the NEON kernel is selected at runtime either way).
- **No I/O in the timed region.** The criterion benches and the
  `precompute_benchmark` example synthesize / clone inputs into memory *before*
  timing; only the metric call is measured, and the result is `black_box`-ed so
  it isn't optimized away. (`compare_tool` is a *correctness* check, not a timer,
  and does read files — see below.)
- **Single-thread vs single-thread.** The default build is single-threaded
  (`rayon` is off by default); compare like-for-like. Don't pit a `rayon` build
  against a scalar one.
- **Apples-to-apples inputs.** Same images, same dimensions, same pixel format,
  same code path across the contenders being compared.

## Reproduce (this repo)

```sh
git clone https://github.com/imazen/fast-ssim2 && cd fast-ssim2
git checkout <commit>          # the commit named in the result file you're reproducing

# self timings (criterion-compat via zenbench), 320×240 / 1920×1080 / 3840×2160:
cargo bench -p fast-ssim2

# scalar vs SIMD, per kernel and full metric:
cargo run --release --example benchmark_simd

# one-shot vs warm compare vs warm compare_with (zero-alloc batch path):
cargo run --release --example precompute_benchmark

# peak-memory shape of the strip path vs the full-image path:
cargo run --release --example strip_memory
```

Build **without** `-C target-cpu=native`.

## Reproduce (score parity vs upstream)

The `compare_tool` binary (workspace member `compare-ssim`, excluded from the
default workspace build) scores one image pair with three implementations and
prints the deltas. It pins the upstream Rust port for a fixed reference:

| Reference implementation | Version | Role |
|--------------------------|---------|------|
| [`ssimulacra2`](https://crates.io/crates/ssimulacra2) (rust-av) | 0.5.1 | upstream Rust port, score reference |
| fast-ssim2 (SIMD)        | this repo | default path |
| fast-ssim2 (scalar)      | this repo | `Ssimulacra2Config::scalar()` |

```sh
cd compare_tool
cargo run --release -- source.png distorted.png
# prints: rust-av v0.5.1 score, fast-ssim2 SIMD score (+Δ), fast-ssim2 scalar score (+Δ)
```

This is a correctness check (it does file I/O and is not timed). Ground-truth
parity against the C++ SSIMULACRA2 binary from
[libjxl](https://github.com/libjxl/libjxl) is exercised by
`fast-ssim2/tests/reference_parity.rs` and `jpeg_quality_reference.rs`
(tolerance 1.5 on the 0–100 scale); see `REFERENCE_TESTING.md` at the repo root.

## Committed result files

Each committed run states its host, toolchain, commit, and exact command in its
header. Don't commit numbers you didn't generate, and don't extrapolate one size
or one CPU to another — measure each. Memory claims need heaptrack / `time -v`,
not estimates.

- [`arm_neoverse_n1_baseline_2026-06-01.md`](arm_neoverse_n1_baseline_2026-06-01.md)
  — Ampere Altra (Neoverse-N1) baseline + the horizontal-blur vectorisation, with
  the x86 7950X scalar-vs-SIMD cross-check (full metric ~3.5×, blur ~7×) and the
  reverted Newton-Raphson reciprocal (committed as a falsified finding).
- [`ssim2_perf/2026-05-21_kanetaka_lossless.md`](ssim2_perf/2026-05-21_kanetaka_lossless.md)
  — Ryzen 9 7950X absolute timings for the one-shot / `compare` / `compare_with`
  paths (256² … 1920×1080) across the three lossless Kanetaka-IWAIT-2026 changes
  (state hoisting, zero-weight skip-map, `CompareContext`); all bit-identical to
  the prior path.
- [`cpp_parity_2026-08-31.md`](cpp_parity_2026-08-31.md) — M4 Pro (aarch64)
  agreement with the C++ SSIMULACRA2 binary (jpeg-xl 0.12.0): 576 real
  photographic pairs at mean |delta| 0.024, why the synthetic `uniform_shift`
  cases disagree one-directionally, per-archmage-tier maxima, and the
  0.7.1-vs-0.8.2 comparison. Raw per-case data in the two sibling `.tsv` files.
- [`version_divergence_2026-09-09.md`](version_divergence_2026-09-09.md)
  — why 0.8.2 scores differently from 0.7.1 (it is the cube root, and *only* the
  cube root: HEAD with that one function reverted reproduces 0.7.1 to 6.1e-6),
  which of the two is closer to the C++ binary (neither — the paired CI on the
  difference straddles zero at 2016 cells), and the first attribution of the
  residual C++ gap: every shipped version is +0.0067 high, and adopting jpegli's
  own `CubeRootAndAdd` + 4-unrolled `FastGaussian1D` horizontal pass removes the
  bias and cuts mean |Δ| 21%. Reproduction sources in the sibling
  [`version_divergence_2026-09-09/`](version_divergence_2026-09-09/) directory.
- [`vs_cpp_and_mt_2026-09-10.md`](vs_cpp_and_mt_2026-09-10.md) — speed against
  the C++ binary, CLI to CLI: **1.9× per pixel** (54.8 vs 104.5 ms/MP) and
  **3–5× on small images** (fixed cost ~1.4 ms against ~7 ms). The C++ tool has
  no MT path at all (its `ThreadPool*` is null). Ours reaches 1.6–1.8× on 12
  cores, up from 1.29×, after parallelising XYB / multiply / ssim_map and fixing
  two overhead bugs — and the record shows why the limiter is Amdahl rather than
  cache locality, including the measurement that the strip walker (this crate's
  own locality tool) is *slower* than the full-image path at 4K.
- [`jpegli_kernels_2026-09-09.md`](jpegli_kernels_2026-09-09.md) — landing
  jpegli's cube root and horizontal Gaussian, the two kernels the C++
  SSIMULACRA2 actually evaluates. Removes the +0.0067 bias every released
  version carried (mean(ours − C++) now +0.00012, mean |Δ| 0.0206 → 0.0166, max
  0.52 → 0.17) *and* runs 4–6% faster end to end. Records what moved (mean |Δ|
  0.020 vs 0.9.0), why the opsin matmul stays unfused, and the FMA-class gating
  the fusion-sensitive cube root now needs.
- [`arch_consistency_2026-09-09.md`](arch_consistency_2026-09-09.md) — 28
  score pairs computed on an M4 Pro (NEON) and a Ryzen 9 7900X (AVX2) agree
  **bit for bit**, so the aarch64-measured C++ parity transfers to x86.
  Reproduce with `cargo run --release --example arch_scores` on two hosts and
  `diff`. i686/wasm128 are expected to differ (non-FMA polyfill) and are not
  covered.
- [`blur_stride_2026-09-09.md`](blur_stride_2026-09-09.md) — the horizontal
  blur fell off a 7.6× cliff at power-of-two widths (4 KiB congruence between
  the source and destination planes, since it gathers eight rows at
  `width * 4` bytes). Fixed by placing the temp plane 256 B off the source's
  page position; scores bit-identical, −14.7% end-to-end at 2048×1024 on x86 and
  −35.1% at 4096×512 on aarch64. `benches/blur_stride.rs` guards it.
- [`cbrt_perf_2026-09-09.md`](cbrt_perf_2026-09-09.md) — the companion perf
  answer: jpegli's `CubeRootAndAdd` is not just bit-exact with the C++ reference,
  it is **faster** than the cube root fast-ssim2 ships (2.7-2.9% on NEON, 6-8% on
  AVX2 at >=64K px, medians of three runs per host), because it iterates the
  reciprocal cube root — no divides, and a seed that vectorises.
  `magetypes::f32x8::cbrt_midp` is the same algorithm as ours and measures
  slower. Also records that `cloudinary/ssimulacra2`, `libjxl` and `jpegli` carry
  byte-identical `CubeRootAndAdd` and `FastGaussian1D`.
- [`ssim2_perf/2026-08-31_x86_0.9.0.md`](ssim2_perf/2026-08-31_x86_0.9.0.md)
  — Ryzen 9 7900X check that the 0.9.0 API change costs nothing. Its real
  finding is about the harness, not the code: `ssimulacra2_320x240` spans
  7.5–8.2 ms across **three runs of one unchanged binary**, so no sub-1 ms
  delta at that size means anything. Every other case reproduces to ≤0.3%.

`BENCHMARKS.md` at the repo root is **historical** (pre-archmage rewrite: it
describes removed `simd` / `unsafe-simd` feature flags and the `wide` crate) and
is kept only for provenance — it does not describe the current build.

## Charts (what to plot for which decision)

| Question | Chart |
|----------|-------|
| "How much does SIMD buy over scalar?" | horizontal **bar**, sorted by throughput (MP/s); one bar per backend per size |
| "How does cost scale with image size?" | **line**, x = pixels (log); fit `total = α + β·pixels` and report the fixed overhead AND the per-pixel slope |
| "Is the warm-vs-cold delta real / how noisy?" | **violin** or PDF of per-call times, or a paired 95% CI |

For new comparison charts prefer [zenbench](https://github.com/imazen/zenbench)
— it interleaves A/B runs to cancel thermal/scheduler drift and emits a sorted
throughput bar chart plus a self-contained SVG report. The current benches use
zenbench's `criterion-compat` shim; porting them to native zenbench is the way to
get publishable paired-CI charts.
