## Previous engine development record (through 2026-09-23)

Historical implementation notes, superseded by the config-driven pipeline above.
The measurements and links describe the commits that produced them.


### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next minor (0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->
- `Ssimulacra2Config` gains a public `tuning: Tuning` field, so struct-literal
  construction (`Ssimulacra2Config { impl_type }`) needs the field or
  `..Default::default()`. Constructor callers (`::new`, `::simd`, `::scalar`,
  `Default`) are unaffected; `with_tuning(t)` is the builder-style setter.

### Added

- **`Tuning`: machine-adaptive scheduling constants.** A small struct of
  knobs that decide *when* parallel execution is worth its overhead and how
  work is divided — never *what* is computed, so scores cannot depend on
  them. Resolved by `Tuning::detect()`: a fixed per-ISA table of measured
  defaults, then `FAST_SSIM2_*` environment overrides (`FAST_SSIM2_VBAND_MIN_GROUPS`,
  `FAST_SSIM2_PAR_MIN_SAMPLES`). Reachable through
  `Ssimulacra2Config::with_tuning` / the `tuning` field, `Blur::with_tuning`
  and `Blur::set_tuning`; `Tuning::serial()` is the deterministic all-off
  value for benchmarking. Knobs: `min_groups_per_band` (vertical-blur
  banding, `0` = off) and `par_min_samples` (the `rayon` engagement floor,
  previously the fixed `PAR_MIN_SAMPLES` constant).
- **The vertical blur pass is parallelised — over pre-sliced column bands, on
  the machines where that measured a win.** Each band is whole `LANES`-wide
  column groups of a per-column IIR, so the split is bit-identical however
  the columns are divided. `Tuning::detect()` turns it **on for aarch64**
  (`min_groups_per_band = 64`: M4 Pro −21% at 4K, flat across band counts)
  and **off everywhere else** — the fleet measurement showed the optimal
  band count is machine-dependent, not portable (helps WSL2 −13%, hurts four
  other x86 boxes +11–25%). x86 users on a machine where it pays can opt in
  per-process via `FAST_SSIM2_VBAND_MIN_GROUPS` or `Ssimulacra2Config::with_tuning`;
  4–8 bands measured −7 to −12% on Zen 3, so `64`–`120` is the range to
  sweep. Full record: [`benchmarks/vertical_band_parallel_2026-09-10.md`](../../benchmarks/vertical_band_parallel_2026-09-10.md)

- **Parallel paths for XYB conversion, `image_multiply` and `ssim_map` under the `rayon` feature**, all bit-identical to the serial path (pixels and channels are independent, so no reduction order changes; the pinned `implementation_parity` scores and `simd_consistency` pass with the feature on and off). With the overhead fixes below, multi-threaded throughput on 12 cores goes from 1.29× to **1.62×** at 4K and 1.35× to **1.81×** on the RGB path. The remaining serial block is the vertical blur pass (~26% of runtime); its columns are independent but each worker would write a strided region, which safe Rust cannot express as disjoint `&mut` slices without a staging buffer — see [`benchmarks/vs_cpp_and_mt_2026-09-10.md`](../../benchmarks/vs_cpp_and_mt_2026-09-10.md). (That claim proved wrong — per-row `split_at_mut` grouped by band does it with no staging buffer; the pass is now parallelised behind `Tuning`, above.)

### Changed

- **BEHAVIOUR: scores move. The opsin cube root and the horizontal Gaussian are now jpegli's own, which is what the C++ SSIMULACRA2 evaluates.** Both were adopted because they are simultaneously *faster* and *closer to the reference* — there was no trade-off to weigh.

  Against the C++ binary (`/opt/homebrew/bin/ssimulacra2`, jpeg-xl 0.12.0), 96 references × 3 sizes × 7 distortions = 2016 cells:

  | | 0.9.0 | now |
  |---|--:|--:|
  | mean(ours − C++) | +0.00740 | **+0.00012** |
  | mean \|Δ\| | 0.02056 | **0.01658** |
  | max \|Δ\| | 0.5223 | **0.1726** |

  The systematic positive bias every released version has carried is gone, and the worst case is a third of what it was. Speed, paired A/B over interleaved rounds: on an M4 Pro (3 rounds) `ssimulacra2_1920x1080` −5.3%, `3840x2160` −4.3%, the `blur` kernel −12.9%; on a Ryzen 9 5900XT (Zen 3, 2 rounds, run-to-run spread 0.1–1.3%) −1.8%, −1.4% and −5.1%. Every case improves on both.

  **What moves:** scores shift by mean \|Δ\| 0.020 against 0.9.0 (max 0.45; 48% of cells move more than 0.01). Anything holding fast-ssim2 scores to a fixed value — pinned fixtures, cached quality decisions, RD curves — needs re-baselining. `tests/implementation_parity.rs` re-pinned its four real-image scores accordingly (verified identical on aarch64 and x86_64 before re-pinning).

  Two mechanical notes: the cube root is jpegli's `CubeRootAndAdd`, which iterates the *reciprocal* cube root (no divisions, vectorisable seed); the horizontal Gaussian is jpegli's `FastGaussian1D`, four outputs per iteration from the closed forms for 2/3/4 recurrence steps, replacing the eight-rows-per-lane-group form. The opsin matrix multiply stays **unfused** — measured indistinguishable from the fused form in C++ agreement (0.01953 vs 0.01968) while keeping the scalar/wasm arms bit-identical, so 0.9.0's arch-consistency decision stands.

  The cube root is FMA-shaped, so like the blur it now differs on targets without hardware FMA (the `magetypes` scalar polyfill: measured 2.98e-7 per XYB sample, 0 on every FMA-capable tier). `tests/simd_consistency.rs` gates that the same way it already gated the blur: bit-identity required within an FMA class, a measured per-sample bound across classes. Both `SimdImpl` backends and both architectures were re-verified bit-identical.

### Fixed

- **`rayon` made small images *slower*, and barely helped large ones.** The horizontal blur split work per *row* — about a microsecond each, so a 320-wide plane meant one `rayon` join per row — and no stage had a minimum size, while the pyramid shrinks 4× per scale. 320×240 measured **2× slower** with the feature on than off (5.29 ms vs 2.99 ms). Blur tasks are now groups of rows (`rows / (threads * 4)`), and every parallel path takes a `PAR_MIN_SAMPLES` (2¹⁸) floor. 320×240 with `rayon` now matches the single-threaded time exactly (2.99 ms) instead of doubling it
- **Horizontal blur fell off a 4 KiB-aliasing cliff at power-of-two widths.** The pass runs the IIR over eight rows at once, one row per SIMD lane, so each column access is eight loads at stride `width * 4` bytes; at a power-of-two width those are congruent modulo 4096, and when the destination plane is page-aligned as well — deterministic for planes of a few MB — the loads, the stores and each other all collide in one cache set. Measured 5.34 ns/px at width 1024 against 0.70 at 1032 on a Ryzen 9 7900X (7.6×), and 3.89 against 0.72 at width 4096 on an Apple M4 Pro. `SimdGaussian` now keeps 1024 spare floats and places the temp plane 256 bytes past the source plane's own position within a page, computed from the actual pointers. **Scores are bit-identical** — this moves where the intermediate lives, not what it contains. End-to-end `compute_ssimulacra2`, paired A/B over three interleaved rounds: **−14.7% at 2048×1024** and −11.8% at 1024×1024 on x86, **−35.1% at 4096×512** on aarch64, every non-power-of-two width within ±1.2%. Regression guard: the new `blur_stride` bench. See [`benchmarks/blur_stride_2026-09-09.md`](../../benchmarks/blur_stride_2026-09-09.md)
- **Test-only:** `cpp_parity_diag::cpp_cube_root_and_add` grouped its FMAs the wrong way round. Highway's `NegMulAdd(a, b, c)` is the fused `c - a*b`, so `NegMulAdd(xa_3, Mul(r2, r2), Mul(k4_3, r))` fuses `xa_3 * r4` and rounds `k4_3 * r` first; the old spelling fused the other product. The two agree on 89.7% of the opsin domain and are up to 3.6e-7 apart on the rest, so jpegli's cube root measures 2.62 ulp max error there, not the 3.34 ulp previously recorded. No shipped code path is affected — the module is `#[cfg(test)]`.

### Documentation

- **New cross-architecture record: [`benchmarks/arch_consistency_2026-09-09.md`](../../benchmarks/arch_consistency_2026-09-09.md), and the `arch_scores` example that produces it.** 28 score pairs (four sizes × seven distortions, sources synthesised in-process so no corpus is needed) computed on an Apple M4 Pro (`neon`) and a Ryzen 9 7900X (`v3`/AVX2+FMA) agree **bit for bit** at full `f64` precision. Every parity number in the earlier records was aarch64-only; this establishes that they transfer to x86. i686 and wasm128 take the non-FMA `magetypes` polyfill and are expected to differ, so these scores must not be pinned as a CI fixture until that is measured
- **New perf record: [`benchmarks/cbrt_perf_2026-09-09.md`](../../benchmarks/cbrt_perf_2026-09-09.md).** jpegli's `CubeRootAndAdd` — the one the C++ SSIMULACRA2 evaluates — is *faster* than the cube root fast-ssim2 ships: 2.7–2.9% on NEON (M4 Pro) and 6–8% on AVX2 (Ryzen 9 7900X, ≥64K px) in the XYB kernel, medians of three runs per host with the memcpy floor subtracted. It iterates the reciprocal cube root, so it has zero divides where ours has two, and its seed vectorises. `magetypes::f32x8::cbrt_midp` is the same algorithm we hand-roll and measures 12% slower on NEON. The record also documents that `cloudinary/ssimulacra2` (the official repo), `libjxl` and `jpegli` carry byte-identical `CubeRootAndAdd` and `FastGaussian1D`, and corrects the 2026-08-31 claim that bit-exactness needs matching the reference's vector width (its horizontal Gaussian is capped at four lanes on every target).
- **New parity record: [`benchmarks/version_divergence_2026-09-09.md`](../../benchmarks/version_divergence_2026-09-09.md).** Attributes the 0.7.1 → 0.8.2 score divergence to the cube root alone (HEAD with `cbrtf_fast` and the fused opsin matmul restored reproduces 0.7.1 to mean |Δ| 6.1e-6 over 2016 cells), confirms with a paired bootstrap that neither version is closer to the C++ binary (0.7.1 − 0.8.2 = +0.00036, 95% CI [−0.00050, +0.00124]), and measures that jpegli's own `CubeRootAndAdd` plus its 4-unrolled `FastGaussian1D` horizontal pass would remove the +0.0067 positive bias every shipped version carries, cutting mean |Δ| 21% and max |Δ| 51%. Reproduction sources committed alongside.
