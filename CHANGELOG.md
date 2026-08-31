# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next minor (0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->
_(none — both previously queued items shipped in 0.9.0.)_

## [0.9.0] - 2026-08-31

`fast-ssim2-cli` goes to **0.6.1** in the same commit. Its own surface — the `image <source> <distorted>` command and the score it prints — is unchanged, so it takes a patch bump, not a leading-digit one; only its `fast-ssim2` requirement moved (`0.8` → `0.9.0`).

### Removed

- **`compute_frame_ssimulacra2` and `compute_frame_ssimulacra2_with_config`** (deprecated since 0.8.0). Migration: `compute_ssimulacra2` / `compute_ssimulacra2_with_config`, which take `ToLinearRgb` inputs — including `Yuv`, which is new in this release (see below) (68c22ee)

  One consumer in the zen workspace calls them: `zenjpeg/tests/bundled/edge_tile_ssim2_comparison.rs`, in a `#[ignore]`d test behind `#[allow(deprecated)]`. It is **not broken by this release** — zenjpeg requires `fast-ssim2 = "0.8.0"`, which does not resolve to 0.9.0 — but it will need one line changed whenever that requirement is widened: it passes two `LinearRgbImage`s, which implement `ToLinearRgb`, so `compute_frame_ssimulacra2(a, b)` becomes `compute_ssimulacra2(a, b)` and the `#[allow(deprecated)]` goes. (An earlier ablation report said no workspace consumer used them; that was wrong.)

  Three workspace crates depend on this one by **path**, so they compile against 0.9.0 with no version bump: `codec-eval/crates/codec-iter`, `zenmetrics/crates/zenmetrics-cli`, and `glassa`. None of them call the removed functions or implement `ToLinearRgb` — a `grep` over `~/work` finds no `ToLinearRgb` implementor outside this repo at all — so the break reaches nothing.

### Changed

- **BREAKING: `ToLinearRgb` conversion is fallible.** The required method is now `try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error>`, and the provided buffer-reusing method is `try_into_linear_rgb(self)`. `to_linear_rgb` / `into_linear_rgb` are gone; **no infallible convenience method replaces them**, because a panicking provided method on a trait whose purpose is to stop panicking would defeat the change (68c22ee)

  Migration — implementors:
  ```rust
  // before
  fn to_linear_rgb(&self) -> LinearRgbImage { LinearRgbImage::new(data, w, h) }
  // after
  fn try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error> {
      LinearRgbImage::try_new(data, w, h).map_err(|_| Ssimulacra2Error::InvalidImageSize)
  }
  ```
  Migration — direct callers: `x.to_linear_rgb()` → `x.try_to_linear_rgb()?`. Callers who only go through `compute_ssimulacra2` / `Ssimulacra2Reference` / the strip APIs need no change; those already returned `Result<f64, Ssimulacra2Error>` and now propagate conversion failures through it.

  Why: the deleted `compute_frame_*` pair was bounded on `LinearRgb: TryFrom<T>`, which accepts `Yuv<u8>`, while its documented replacement was bounded on `ToLinearRgb`, for which no `Yuv` impl existed. The deprecated pair was therefore the only public YUV entry point and its migration note was false. YUV→linear RGB fails on ordinary caller-supplied colour signalling, so a `ToLinearRgb for Yuv` impl built on an infallible trait would have had to panic — turning a recoverable `Err` into an abort. Making fallibility the primitive removes that temptation everywhere.

- **BREAKING: `Ssimulacra2Error` is `#[non_exhaustive]` and gained `Cancelled(enough::StopReason)`.** Downstream `match` arms need a wildcard `_ =>`. Shipped with the cooperative-cancellation API below (742ea99)
- **Zero-sized inputs are `Err(Ssimulacra2Error::InvalidImageSize)` instead of a panic.** Every `ToLinearRgb` impl now builds through `LinearRgbImage::try_new`; previously a 0-dimension `ImgRef` panicked inside `LinearRgbImage::new`. `InvalidImageSize`'s documentation now covers both this and the sub-8×8 strip case (68c22ee)

### Added

- **`ToLinearRgb` for `yuvxyb::Yuv<T>` and `&yuvxyb::Yuv<T>`** (`T: yuvxyb::Pixel`, i.e. `u8` and `u16` — its only implementors), so planar YUV straight off a decoder is a first-class input to `compute_ssimulacra2`, `Ssimulacra2Reference` and the strip APIs. Unconvertible colour signalling returns `Err(Ssimulacra2Error::LinearRgbConversionFailed)`; verified reachable in yuvxyb 0.5.0 for H.273 MC=3 (`Reserved`) and MC=12 (`ChromaticityDerivedNonConstantLuminance`). `MatrixCoefficients::Unspecified` is *not* one of them — `Yuv::new` rewrites it (68c22ee)
- **`ToLinearRgb` for `yuvxyb::Xyb`** — infallible, and the input type the crate's own `test_ssimulacra2` uses (68c22ee)
- `input.rs::yuv_tests` — six tests covering the YUV path: bit-for-bit agreement with `yuvxyb`'s own conversion, owned-vs-borrowed equivalence through a generic bound, unsupported matrix coefficients as `Err` on both impls, propagation through `compute_ssimulacra2`, identical frames scoring 100, and the `Rgb` ST428 regression below (68c22ee)
- Cooperative cancellation across every slow path: `compute_ssimulacra2_with_stop` / `compute_ssimulacra2_strip_with_stop` (one-shot), and on `Ssimulacra2Reference` the warm-reference batch paths `compare_with_stop` / `compare_with_and_stop` (zero-alloc, reuses a `CompareContext`) and the cached-ref strip paths `compare_strip_with_stop` / `compare_strip_with_config_and_stop`. All take a `&dyn enough::Stop` token and return `Err(Ssimulacra2Error::Cancelled)` if cancelled; the token is checked at the per-scale / per-strip outer-loop boundary — never per-pixel. The existing non-`_stop` methods delegate with `enough::Unstoppable` (unchanged behavior) (742ea99)
- `examples/photo_parity.rs` — C++ parity on real photographic content (codec-corpus: CID22, KADID-10k, gb82, CLIC2025) across four sizes and six distortion families, 576 pairs. The compiled-in reference table is entirely synthetic and 40 of its 66 cases compare an image against *itself*, which every implementation scores exactly 100.0; this measures the thing that table cannot (dccb0e9)
- `examples/parity_report.rs` — **signed** per-case delta against the C++ binary for both backends. `tests/reference_parity.rs` asserts only `|error| <= tolerance`, which cannot distinguish symmetric FP noise from a one-directional bias (dccb0e9)
- `examples/flat_field_probe.rs` — sweeps texture amplitude to show how much of a `uniform_shift` disagreement is flat-field degeneracy (dccb0e9)
- `src/cpp_parity_diag.rs` (test-only) — per-stage diagnostics with swappable cube root (ours-f32 / ours-f64 / a verbatim transliteration of jpegli's `CubeRootAndAdd`) and swappable horizontal Gaussian (sequential / jpegli's 4-unrolled / f64 reference), so a divergence can be localised to a stage instead of guessed at from the final score (dccb0e9)
- `tests/simd_consistency.rs`: `dispatched_kernels_are_bit_identical_across_tiers` (bit-identity, not a tolerance) and `blur_tier_divergence_is_bounded` (1e-5 per blurred sample). The end-to-end tier check now partitions combinations by blur-equivalence class, so combinations with an identical blur are held to 1e-4 instead of the old blanket 0.5 (dccb0e9)
- `fast-ssim2/dev/testcases.rs` — one copy of the synthetic-case generators, `include!`d by `capture_cpp_reference`, `parity_report` and `reference_parity`. Those three previously carried hand-synchronised copies, with a comment in the test reading "must match capture_cpp_reference.rs exactly" (dccb0e9)

### Fixed
- **`SimdImpl::Scalar` and `SimdImpl::Simd` computed different metrics, not the same metric at different speeds.** `SimdImpl::Scalar` routed the XYB conversion through `yuvxyb`, whose cube root is two Newton steps in f64 (0.50 ulp), while `SimdImpl::Simd` used two Halley steps in f32 (1.75 ulp). SSIMULACRA2 divides a blur residual of ~5e-7 by `kC2 = 9e-4` and rectifies the result with `max(d, 0)`, so it amplifies a ~1e-7 input difference by ~1e6: measured over 576 real photographic pairs the two backends disagreed by up to **0.879** on the 0..100 scale. `tests/simd_consistency.rs` carried a 0.5 tolerance and synthetic images that never provoked it. Both backends now share one cube root and one operation order (`xyb_simd::linear_rgb_to_xyb_scalar`); the same corpus now measures a maximum disagreement of **2.6e-7**. Agreement with the C++ binary is unchanged (mean |delta| 0.0244 → 0.0239) (82d9da8)
- **Within `linear_rgb_to_xyb_simd`, a plane's last seven pixels were converted with different arithmetic than the rest of it** — the `len % 8` remainder used the f64 cube root while the vectorised body used the f32 one. Both now use `cbrtf_halley_f32` (82d9da8)
- **`edge_diff_map`'s SIMD kernel lost the whole signal on smooth content.** It computed `d1 = (1 + diff2) / (1 + diff1) - 1` in f32, where the subtraction carries ~1 ulp of 1.0 (6e-8) of *absolute* error however small the true value is; on smooth content both diffs are ~1e-7, so `d1` was noise, and `max(d1, 0)` rectified it into a one-directional bias. Replaced with the algebraically identical, cancellation-free `(diff2 - diff1) / (1 + diff1)` in the vector body, its scalar remainder (which previously used a *third* formulation, in f64) and the scalar kernel (82d9da8)
- **Fusion-sensitive expressions removed from the dispatched kernels, which were not arch-consistent.** `magetypes` lowers `mul_add` to a real FMA on NEON/AVX2/AVX-512 but to `a * b + c` on wasm128 and on its scalar polyfill, so the opsin matrix and the SSIM' `num_m` term returned different values on any target without AVX2/NEON — 1.79e-7 in the XYB output, amplified to **0.085** end-to-end. Both are now written unfused (which also matches the C++ reference, whose `SSIMMap` writes `num_m` unfused). All four dispatched kernels are now bit-identical across every archmage tier permutation. The recursive Gaussian's `MUL_PREV` step is deliberately left fused because the reference fuses it; unfusing it was implemented, measured and rejected — see `benchmarks/cpp_parity_2026-08-31.md` (82d9da8, db1873d)
- **`ToLinearRgb for yuvxyb::Rgb` could abort the process on a valid image.** Its non-sRGB arm unwrapped `LinearRgb::try_from` with `.expect("Rgb to LinearRgb conversion should not fail")`. It does fail: H.273 TC=17 (`ST428`, digital cinema) — and TC=0/3 `Reserved0`/`Reserved` and TC=12 `BT1361E` — have no `to_linear` in yuvxyb, so an ST428-tagged image panicked instead of returning an error. Now `Err(Ssimulacra2Error::LinearRgbConversionFailed)`, with a regression test (68c22ee)
- `Blur`'s doc claimed the scalar path was an "f64 IIR baseline (most accurate)". It has always been f32 (82d9da8)
- **Pushes to `main` now cancel their superseded CI runs.** `ci.yml` keyed its concurrency group on `${{ github.head_ref || github.run_id }}`. `github.head_ref` is populated only for `pull_request` events, so on a push it was empty and the group fell through to `github.run_id` — unique per run, so no two pushes ever shared a group and `cancel-in-progress` could never fire. Every push started a full matrix that ran to completion even when several commits landed seconds apart. Now keyed on `${{ github.ref }}`, which is set for both event types (`refs/heads/main` on push, `refs/pull/N/merge` on a PR), so PR cancellation is unchanged and consecutive pushes supersede each other (db1c7dc)

### Performance

Measured on x86_64 (Ryzen 9 7900X, Zen 4), `nice -n 19 ionice -c 3 cargo bench --bench benches`, full suite, no `target-cpu=native`. Full record with every run: `benchmarks/ssim2_perf/2026-08-31_x86_0.9.0.md`.

**No measurable regression.** Three consecutive runs of the 0.9.0 build:

| Benchmark | run 1 | run 2 | run 3 | v0.8.2 (2 runs) |
|---|---|---|---|---|
| `ssimulacra2_320x240` (YUV) | 7.5 ms | 8.2 ms | 7.5 ms | 7.5 / 7.8 ms |
| `ssimulacra2_1920x1080` (YUV) | 231.5 ms | 231.5 ms | 231.2 ms | 231.2 / 230.2 ms |
| `ssimulacra2_3840x2160` (YUV) | 965.9 ms | 965.0 ms | 962.6 ms | 960.3 ms |
| `ssimulacra2_rgb_320x240` | 8.4 ms | 8.4 ms | 8.4 ms | flat |
| `ssimulacra2_rgb_1920x1080` | 312.2 ms | 312.1 ms | 311.7 ms | flat |
| `ssimulacra2_rgb_3840x2160` | 1.31 s | 1.31 s | 1.31 s | flat |
| `blur` | 5.4 ms | 5.5 ms | — | flat |

An earlier v0.8.2-vs-`f56991e` comparison on this host read `ssimulacra2_320x240` at 8.5 / 8.1 ms and looked like a +8–9% regression at the smallest YUV size. **It does not reproduce, and it is not established by the data**: the 0.9.0 build alone spans 7.5–8.2 ms across three runs of the identical binary, so a 0.7 ms band is run-to-run spread on one build, not an effect of a code change. zenbench reports `0 rounds` on every case here — it never completes its round budget — and 320×240 is the only benchmark short enough (7–8 ms/iter) for that to dominate the mean. Every other case reproduces to ≤0.3% across runs *and* across builds; 4K sits ~0.5% above v0.8.2's single sample, inside its own spread.

No per-call fixed-overhead figure is claimed. Establishing one needs an `α + β·pixels` fit over ≥4 sizes on a build where zenbench completes its rounds; that has not been done.

Not measured on aarch64.

### Semver

`cargo semver-checks check-release -p fast-ssim2 --baseline-version 0.8.2` reports **4 major-category failures, 6 items** (run with the version temporarily set to 0.8.3, since a 0.9.0 bump makes every check "unnecessary" and reports nothing):

| Lint | Item |
|---|---|
| `function_missing` | `compute_frame_ssimulacra2` |
| `function_missing` | `compute_frame_ssimulacra2_with_config` |
| `trait_method_missing` | `ToLinearRgb::to_linear_rgb` |
| `trait_method_missing` | `ToLinearRgb::into_linear_rgb` |
| `trait_method_added` | `ToLinearRgb::try_to_linear_rgb` (new required method) |
| `enum_marked_non_exhaustive` | `Ssimulacra2Error` |

Two further source-compatible-but-visible changes that no lint fires on: `Ssimulacra2Error::Cancelled` was added (covered by the `#[non_exhaustive]` failure above), and `yuvxyb::Hsl` — reachable as an input through the old `LinearRgb: TryFrom<T>` bound — is **no longer accepted**. Nothing in the workspace used it; convert with `yuvxyb::LinearRgb::from(hsl)` and pass the result.

### Documentation
- `benchmarks/cpp_parity_2026-08-31.md` — full C++ parity and cross-tier measurement record (aarch64/M4 Pro, jpeg-xl 0.12.0 reference binary): why the `uniform_shift` reference cases disagree, why that disagreement is one-directional, real-photo agreement, tier maxima, and the 0.7.1-vs-0.8.2 version comparison (dccb0e9, db1873d)
- `tests/reference_parity.rs`: replaced the `uniform_shift` tolerance justification. It read "different FP rounding in SIMD paths", which was wrong twice — the two backends agree there to ~1e-8, and the disagreement is with the reference and is one-directional (18 of 20 cases positive, mean +0.408). The real cause, and the measurements behind it, are now in the comment (dccb0e9, f7220c9)
- README: documented the cooperative-cancellation API (the `*_with_stop` variants were shipped but never appeared in the README — found via an insulated external-developer usability test), the flat-`Vec<u8>` → `ImgVec` on-ramp in the Quick Start, the `f64` score type, the no-`[u8; 4]`/alpha note, and the strip API signatures + strip-height semantics (093aa6f)
- README overhaul to the zen-family conventions: canonical badge row (dropped `branch=` and the codecov badge, `license` → in-page anchor), rendered crosslink footer, credit to the SSIMULACRA2 authors (Cloudinary / libjxl) alongside the rust-av port, and a split crates.io README (`README.crates.md`, generated; `readme` now points at it). Replaced the unverifiable "vs upstream crate" speedup table with measured, committed scalar-vs-SIMD / batch figures + repro, and added `benchmarks/README.md` methodology (c6fd462, 585006c, f6af3af)

## [0.8.2] - 2026-06-10

### Added
- Sub-8px inputs are reflect(mirror)-padded up to the metric's 8px pyramid floor instead of returning `InvalidImageSize`: `compute_ssimulacra2` / `compute_ssimulacra2_with_config` now score images down to 1×1 (identical pairs still score 100) (480df7e)
- `Ssimulacra2Reference` applies the same sub-8px reflect-padding, so the batch path accepts the same inputs as the one-shot path and produces identical scores; `width()`/`height()` report the caller-supplied (pre-padding) dimensions and mismatched pre-padding dimensions are still rejected. Strip APIs intentionally keep the ≥8×8 requirement (54df4683)
- Experimental `hdr-pu` feature: `compute_ssimulacra2_pu_nits` scores HDR content using the PU21 (banding_glare) encoding in place of the cube-root opsin nonlinearity; input is absolute-luminance linear RGB in cd/m². Validated on UPIQ HDR (380 pairs, SROCC 0.7044; see imazen/zenmetrics#25) (35f198af)
- CI now lints (`clippy -D warnings`) and tests the non-default `hdr-pu` feature on the Linux clippy job, so the feature-gated path is exercised on every push (f987fc1c)

### Changed
- Vectorised the recursive-Gaussian blur **horizontal pass** across rows (8 rows per SIMD lane group, one row per lane), mirroring the across-columns trick the vertical pass already uses. Bit-identical output to the scalar path — the SIMD IIR replicates the scalar op order exactly. Measured on Neoverse-N1 (where the scalar per-row recurrence was ~50% of the blur kernel): blur-only +9–12%, full SSIMULACRA2 +3.6–9.5% (`ssimulacra2_1920x1080` 540.9 → 503.0 ms). x86 (AVX2) unaffected — blur was already well-vectorised there (87e06d5)

## [0.8.1] - 2026-05-27

### Added
- `compute_ssimulacra2_strip` and `Ssimulacra2Reference::compare_strip` — strip-wise SSIMULACRA2 with bounded peak memory for very large images. Processes the image in horizontal strips (default 32-aligned, halo of 96 rows for IIR Gaussian convergence) and accumulates per-scale SSIM and edge-diff sums across strips. Scores match the full-image path to within ~1e-5 on the 0..100 scale at 1024² and larger; identical-image inputs round-trip to 100 in both modes. Bounds dist-side peak memory to ~`24 * strip_h * width * 4 B` instead of ~`24 * height * width * 4 B`.
- `Ssimulacra2StripConfig` — configurable halo size and underlying SIMD backend for strip processing. Defaults are tuned for atomic-tolerance parity with the full path.
- `HALO_ROWS_DEFAULT` (96) and `MIN_STRIP_HEIGHT` (8) public constants for callers that need the strip walker's tuning constants
- Hidden `Ssimulacra2Reference::scale_planes` accessor returning a `ScalePlanesView` of the cached per-scale XYB-planar data; required by the strip walker to derive ref-side strip slices, marked `#[doc(hidden)]` because the representation is an implementation detail

### Added (from prior unreleased work)
- `CompareContext` and `Ssimulacra2Reference::compare_with(&mut ctx, distorted)` — zero-allocation batch-comparison API. Pair with `reference.compare_context()`; subsequent calls reuse the working buffers (`mul`, `mu2`, `sigma2_sq`, `sigma12`, `img2_planar`, blur state) instead of allocating ~13 image-sized `Vec<f32>` planes per call. Measured 1.10–1.25× faster than `compare()` on the precompute benchmark at 256x256 / 512x512 / 1024x1024 / 1920x1080 (c419b3d)
- `LinearRgbImage::try_new` fallible constructor returning `LinearRgbImageError` for invalid dimensions or data length
- `Ssimulacra2Error::ImageTooLarge` variant and public `MAX_IMAGE_PIXELS` constant (16384*16384) capping caller-supplied image size to prevent unbounded working-buffer allocation

### Changed
- Skip per-channel SSIM and edge-difference work whose final-score weight is zero. Bit-identical to the prior path (the dropped contributions multiplied by zero downstream); reference-parity test passes across the C++ corpus including 64x64 cases where `scales_n < NUM_SCALES` makes `score()`'s linear WEIGHT walk shift in the layout. Lossless variant of Technique 2 from Kanetaka et al. IWAIT 2026, DOI 10.1117/12.3100969 (75e3234)
- Hoist the 6 IIR-state vectors used by the SIMD vertical blur pass out of the per-call inner function and onto `SimdGaussian`, eliminating ~180 small `Vec<f32>` allocations per ssim2 frame (bc9d011)

### Fixed
- `LinearRgbImage::new` now validates dimensions and data length at runtime (was `debug_assert_eq!` only) so release-mode misuse no longer constructs malformed images that panic deep in `From<LinearRgbImage> for yuvxyb::LinearRgb`
- `SimdGaussian::new` no longer eagerly allocates `max_width * 4096` floats; the temp buffer grows on demand. Also guards against `usize` overflow on 32-bit targets when `width * height` would wrap

## Version 0.7.3

- Add proper CI workflow with full platform matrix (Linux, macOS, Windows on x64/ARM64), i686 cross testing, WASM testing, MSRV verification, and code coverage
- Fix unused import lint on i686 from archmage `#[autoversion]` dispatch

## Version 0.7.0

- Update all dependencies to latest versions
- criterion 0.5 → 0.8, rand 0.8 → 0.10, png 0.17 → 0.18, which 7 → 8
- crossterm 0.27 → 0.29, indicatif 0.17 → 0.18, statrs 0.17 → 0.18
- safe_unaligned_simd 0.2.3 → 0.2.4, thiserror 2.0.9 → 2.0.18

## Version 0.6.0

- Rename crate from `ssimulacra2` to `fast-ssim2`
- Add `imgref` support and simplified input API
- Add precomputed reference API (`Ssimulacra2Reference`) for batch comparisons
- Add runtime SIMD backend selection via `Ssimulacra2Config`
- Add unsafe SIMD backend with x86 intrinsics for best performance
- Reduce memory allocations by 77% and memory usage by 36%
- Add C++ reference parity tests and JPEG quality regression tests
- Update multiversion to 0.8
- Improve API documentation and README

## Version 0.5.1

- Remove nalgebra-macros and update criterion
- Use yuvxyb-math to calculate float constants
- Cleanup way too verbose Clippy settings
- Update thiserror to 2.0

## Version 0.5.0

- Return a concrete `Ssimulacra2Error` error type instead of a freeform `anyhow::Result`
- Precalculate float consts for RecursiveGaussian at build time (performance)
- Update `yuvxyb` dependency to 0.4

## Version 0.4.0

- Update to [version 2.1 of the metric](https://github.com/cloudinary/ssimulacra2/compare/v2.0...v2.1)

## Version 0.3.1

- Minor optimizations
- Bump `nalgebra` dependency to 0.32

## Version 0.3.0

- [Breaking] Reexported structs from yuvxyb have had `From<&T>` impls removed
- Considerably speedups and optimizations

## Version 0.2.0

- [Breaking] Implement updates to the algorithm from upstream (https://github.com/libjxl/libjxl/pull/1848)
- Bump yuvxyb version
- Speed improvements

## Version 0.1.0

- Initial release
