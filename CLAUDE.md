# fast-ssim2 Project Notes

## C++ parity and arch consistency (2026-08-31, aarch64 M4 Pro)

Full record with every measurement: `benchmarks/cpp_parity_2026-08-31.md`.
Reference binary: `/opt/homebrew/bin/ssimulacra2` (jpeg-xl 0.12.0). Read the
C++ at `~/work/jpegli/tools/ssimulacra2.cc` + `tools/gauss_blur.cc` +
`lib/extras/xyb_transform.cc` + `lib/base/fast_math-inl.h` — **jpegli is a
different repo, read-only.**

- **Does fast-ssim2 match C? On real content, yes.** 576 photographic pairs:
  mean |delta| **0.024**, 283/576 positive (no bias), max 0.523.
- **The synthetic reference table proves much less than it looks like.** 40 of
  its 66 cases compare an image against itself and score exactly 100.0 in any
  implementation. Do not quote "40 of 66 bit-exact" as parity evidence.
- **`uniform_shift` is a degenerate family, not a bug.** SSIMULACRA2 divides a
  ~5e-7 blur residual by `kC2 = 9e-4` and rectifies with `max(d, 0)`, giving a
  ~1e6 amplification on flat fields. Rounding contributes 3x the true signal at
  `uniform_shift_1`; the reference's own values are non-monotonic in the shift.
  The 10.0 tolerance stays; the "different FP rounding in SIMD paths" comment
  that justified it was wrong and has been replaced.
- **The C++ reference is itself not arch-consistent** — `FastGaussian1D` uses a
  4-unrolled recurrence on vector Highway targets and a sequential one on
  `HWY_SCALAR`. Transliterating its form into our pipeline cuts the
  uniform_shift mean error 0.381 -> 0.258. Bit-exact agreement with "the C++
  tool" is not achievable without also fixing which vector width it was built
  for.
- **Fixed: the two `SimdImpl` backends were computing different metrics** (up to
  0.879 apart on real photos, hidden by a 0.5 tolerance and synthetic-only test
  images). Now 2.6e-7 apart. See CHANGELOG for the four defects.
- **The `magetypes` FMA difference is DOCUMENTED POLICY, not an upstream bug —
  do not "fix" it.** An earlier version of this file called it a "known upstream
  gap ... belongs in archmage." That was wrong, and acting on it would make
  things worse. Verified in source:
  - `magetypes/src/simd/impls/wasm128.rs:112` is `f32x4_add(f32x4_mul(a,b), c)`
    because **WASM SIMD128 has no FMA instruction** — a spec limitation, not an
    omission. (`relaxed_madd` exists in relaxed-simd but is *implementation-
    defined* as to whether it fuses, so it would destroy bit-identity, not
    provide it.)
  - `magetypes/src/simd/scalar.rs:125` calls `nostd_math::fmaf`, which is
    documented at its definition as *"non-fused fallback: `a * b + c` ... no
    hardware FMA instruction to use."* A correct software FMA is possible but
    **slow** — that is the tradeoff being made, deliberately.
  - archmage's own `CLAUDE.md` carries a "Known Cross-Architecture Behavioral
    Differences" table headed *"they are not bugs to fix"*, whose `mul_add` row
    prescribes: *"Accept <=1 ULP difference; **avoid near-zero cancellation**."*
    Our flat-field path is exactly the near-zero cancellation it warns about, so
    fast-ssim2 walked into a documented hazard rather than hitting a defect.
  Every fusion-sensitive expression has been removed from our dispatched kernels
  *except* the blur, which stays fused because the reference fuses it (unfusing
  was measured: it costs 2.8x worse C++ agreement plus a -0.058 bias). So
  fast-ssim2 is bit-identical on all FMA-capable targets and up to 0.497 apart
  on wasm128 / no-SIMD builds. **That gap is inherent and the fix is not
  upstream.** The only real options are: accept it (current choice), unfuse
  everywhere and take 2.8x worse C++ agreement, or special-case the blur for
  unfused targets — which needs a measurement, not a patch to archmage.
- **Version question settled: 0.7.1 and 0.8.2 are equally faithful to C** (mean
  |delta| 0.0129 vs 0.0140 over 360 cells; 0.7.1 closer on 189/360 — a coin
  flip). They differ from each other by up to 0.143, so the workspace split
  (jxl-encoder/zengif/zenwebp on `^0.7.1`, which really resolves to 0.7.1
  because 0.7.2/0.7.3 are yanked; everything else on 0.8.2) has a cost and no
  benefit. **Unify on 0.8.2+**, on recency, not accuracy. Re-measured
  2026-09-09 at 2016 cells (`benchmarks/version_divergence_2026-09-09.md`):
  0.7.1 − 0.8.2 in agreement with C++ is +0.00036, 95% CI [−0.00050, +0.00124]
  — still a coin flip, now with a paired CI behind it.
- **Perf answer (2026-09-09, `benchmarks/cbrt_perf_2026-09-09.md`): jpegli's
  cube root is FASTER than ours, on both arches.** 2.7-2.9% on NEON (M4 Pro),
  6-8% on AVX2 (7900X, >=64K px), medians of 3 runs/host, memcpy floor
  subtracted. It iterates the *reciprocal* cube root (`r*r*x` at the end), so it
  has **zero divides** where ours has two, and its seed is pure integer
  shift/multiply so it vectorises instead of round-tripping through
  `to_array`/`from_array`. Fidelity and speed point the same way here — there is
  no trade-off to weigh.
- **`magetypes::f32x8::cbrt_midp` (0.9.29) is NOT an upgrade.** It is the same
  algorithm we hand-roll (Kahan seed pulled into a scalar array, 2 Halley steps,
  2 divides) plus sign/zero handling we do not need, with a different magic
  constant. Measured 12% slower on NEON, 1-2% slower on AVX2. Don't "just use
  the library one" without re-measuring.
- **All three C++ implementations are the same code.**
  `cloudinary/ssimulacra2` (the official standalone), `libjxl` and `jpegli` carry
  byte-identical `CubeRootAndAdd` and `FastGaussian1D` — only include paths,
  macro spellings and signatures differ, and `kC2`/the 108 weights/`SSIMMap`/
  `EdgeDiffMap`/`Downsample` match too. There is no separate "official approach"
  to chase.
- **Correction to the arch caveat above:** the C++ horizontal Gaussian is capped
  at four lanes on every target (`JPEGLI_GAUSS_MAX_LANES 4`,
  `HWY_CAPPED(float, 4)`), so its result does not depend on the build's vector
  width; only `HWY_SCALAR` differs. "Matching the vector width it was built for"
  is not a prerequisite for bit-exactness — matching *scalar vs vector* is.
- **The blur half is measured too: jpegli's 4-unrolled horizontal pass is
  faster, and finding that out uncovered a real perf bug in ours.** Port is
  bit-exact with the scalar transliteration and 12.3-13.7% faster on NEON,
  ~4-7% on x86. The bug: our horizontal pass gathers 8 rows at stride
  `width * 4` bytes, so at a power-of-two width the 8 addresses are 4 KiB
  congruent — and when the destination plane is page-aligned too (deterministic
  for multi-MB planes) everything lands in one cache set. Measured **5.34 vs
  0.70 ns/px at width 1024 on Zen 4** (7.6x), and 3.89 vs 0.72 at 4096 on M4
  Pro. End-to-end that was `compute_ssimulacra2` at 2048x1024 costing 22.5% more
  than 2040x1024. **Fixed** (`SimdGaussian::temp_offset`, scores bit-identical):
  -14.7% at 2048x1024 and -11.8% at 1024x1024 on x86, **-35.1% at 4096x512 on
  aarch64**, other widths within +/-1.2%. Guard: `benches/blur_stride.rs`.
  Record: `benchmarks/blur_stride_2026-09-09.md`. **If you touch the blur's
  buffers, keep the de-aliasing** — and note the same trick has NOT been applied
  to the metric's own plane pairs in `lib.rs`, where a ~3% residual remains at
  width 1024.
- **A faithful port of jpegli's blur is still blocked on magetypes.** A faithful
  port of jpegli's 4-unrolled horizontal pass wants `Broadcast<N>` and
  `ShiftLeftLanes<N>`; `magetypes` 0.9.29 `f32x4` has neither (only
  `interleave_lo/hi`, `transpose_4x4`, `blend`). The dodge — precompute the four
  shifted coefficient vectors as constants, splat each input sum from a scalar
  load — is possible with today's API.
- **Tier coverage: no AVX-512 arm anywhere.** Every kernel is
  `#[magetypes(v3, neon, wasm128, scalar)]`; archmage's `v3` IS AVX2+FMA, so x86
  is not stuck on SSE, but `v4`/`v4x` are absent. Adding them alone would mostly
  re-encode the same 8-lane code — a real win needs `f32x16` bodies. Untested.
- **x86_64 and aarch64 compute the SAME score, bit for bit** (2026-09-09,
  `benchmarks/arch_consistency_2026-09-09.md`). 28 pairs, four sizes x seven
  distortions, `examples/arch_scores.rs` on an M4 Pro (`neon`) and a 7900X
  (`v3`/AVX2): zero differences at full f64 precision. So the aarch64-measured
  C++ agreement transfers to x86, and the NEON/AVX2 arms of every
  `#[magetypes]` kernel agree. **Still not measured:** fast-ssim2 vs the C++
  binary *on* x86 (r7900x has no `ssimulacra2` binary and no corpus — its
  codec-corpus checkout is 3 MB, LFS not pulled), and i686/wasm128, which take
  the non-FMA polyfill and are *expected* to differ. Do NOT pin those 28 scores
  as a CI fixture before measuring i686/wasm — it would fail by design there.
- The 3 "ignored" tests are 3 ```ignore doctest fences (`src/lib.rs` lines 10 and
  367, `src/strip.rs` line 72), not `#[ignore]` attributes. They are pseudo-code
  snippets (`load_image(...)`, `/* ... */`) and two of them need the `imgref`
  feature, which doctests do not build with. There are zero `#[ignore]`s.

## What actually makes the versions differ, and what the C++ gap is (2026-09-09)

Full record: `benchmarks/version_divergence_2026-09-09.md`. Same aarch64 M4 Pro
host and `/opt/homebrew/bin/ssimulacra2` reference; 96 references x 3 sizes x 7
distortions = 2016 cells.

- **The 0.7.1 -> 0.8.2 score divergence is the cube root, and nothing else.**
  0.8.2 swapped the f64 Newton `cbrtf_fast` in the vector XYB body for two f32
  Halley steps (`844605f`, part of the `magetypes` port). HEAD with only that
  and the opsin-matmul association reverted reproduces 0.7.1 to **mean |delta|
  6.1e-6, max 7.4e-5** — so the blur row-vectorisation, the 4->8 NEON lane
  widths, the zero-weight cell skips and the rest of the refactor are
  numerically inert. Do NOT go looking for the divergence in the blur.
  `844605f` knew it moved scores ("< 0.05 absolute delta", and it re-pinned the
  `implementation_parity` expectations); on real content the worst case is 0.20.
- **Cube-root accuracy and C++ fidelity are different axes.** Over the whole
  opsin domain: 0.7.1's f64 Newton is exact (0 ulp), our f32 Halley is 0.49 mean
  / 3 max ulp, and **jpegli's own `CubeRootAndAdd` is the least accurate of the
  three** (0.72 mean / 5 max ulp). So "our cube root is better than C's" and
  "our score is closer to C's" cannot both be optimised.
- **Every shipped version is biased +0.0067 +/- 0.0016 high vs the C++ binary**,
  and it is not the cube root. Adopting jpegli's *own* two approximations — its
  `CubeRootAndAdd` (bias fused into the last multiply-add) plus its 4-unrolled
  `FastGaussian1D` horizontal pass — removes the bias (+0.0010, CI includes 0),
  cuts mean |delta| 0.0209 -> **0.0166** and max |delta| 0.49 -> **0.24**, all
  significant under a paired bootstrap. The horizontal-blur form is the larger
  half. **Not landed:** it is a metric change (needs the user's call), the
  transliteration is FMA-shaped so it re-opens the wasm128/scalar
  arch-consistency question 0.9.0's unfused matmul closed, and jpegli's
  4-unrolled recurrence would have to be re-vectorised to be shippable.
- **Ruled out as sources of the residual:** the sRGB transfer function (the C++
  takes the `ExtraTF::kSRGB` fast path = `TF_SRGB().DisplayFromEncoded`, whose
  ten rational-polynomial coefficients are bit-identical as f32 to
  `input.rs::srgb_to_linear`'s) and `intensity_target` (255 for 8-bit sRGB PNG,
  so the opsin matrix scale is exactly 1.0). Still open: `Downsample` and the
  skcms linear->linear leg.
- **`cpp_parity_diag::cpp_cube_root_and_add` had its FMAs grouped the wrong way
  round** (Highway's `NegMulAdd(a,b,c)` fuses `a*b`, not the other product).
  Fixed 2026-09-09; jpegli's cube root measures **2.62 ulp** max on the opsin
  domain, not the 3.34 ulp this file used to quote.
## Current state (2026-06-10)

- **v0.8.2 RELEASED 2026-06-10**: tag v0.8.2 = b7c2b4b3, GH release with
  changelog notes, published to crates.io (verified). CI was 12/12 green
  (incl. windows-11-arm, macos-26-intel, i686, WASM, MSRV 1.89.0). Ships
  sub-8px reflect-pad unification, `hdr-pu` feature
  (`compute_ssimulacra2_pu_nits`, UPIQ SROCC 0.7044), N1 blur
  horizontal-pass vectorization. Downstream unblock: zenmetrics CPU-ssim2
  HDR routing can now depend on the published `hdr-pu` feature.
- `CompareContext` + `Ssimulacra2Reference::compare_with` (zero-alloc batch
  comparisons) shipped in 0.8.1, including the SIMD blur-state hoisting
  (bc9d011). The old TODO describing that design is done and was removed.
- Sub-8px inputs reflect-pad up to the 8px pyramid floor on the one-shot
  (480df7e) and `Ssimulacra2Reference` (54df4683) paths; strip APIs
  intentionally require ≥8×8.
- Test philosophy (user ruling 2026-06-10): no tests that assert transcribed
  constants against copies of the same constants. Tests must exercise
  behavior (parity vs the C++ SSIMULACRA2 implementation, SIMD-tier
  consistency, strip-vs-full parity, monotonicity, white-point landing).
- Deprecated since 0.8.0, removal queued for 0.9.0 (see CHANGELOG QUEUED
  BREAKING CHANGES when added): `compute_frame_ssimulacra2`,
  `compute_frame_ssimulacra2_with_config`. **DONE — removed in 0.9.0.**

## 0.9.0 API break (2026-08-31)

- **`ToLinearRgb` is fallible.** Required method is now
  `try_to_linear_rgb(&self) -> Result<LinearRgbImage, Ssimulacra2Error>`; the
  provided buffer-reusing method is `try_into_linear_rgb(self)`. There is **no
  infallible convenience method** — a panicking provided method on a trait
  whose point is to stop panicking would defeat the change.
- **Why:** the removed `compute_frame_ssimulacra2` pair was the only public
  YUV entry point (bounded on `LinearRgb: TryFrom<T>`, which accepts
  `Yuv<u8>`), and there was no `ToLinearRgb for Yuv` impl, so its "use
  `compute_ssimulacra2` instead" note was false. YUV→linear fails on real
  metadata, so the replacement impl had to be able to return `Err`.
- **New impls:** `Yuv<T>` and `&Yuv<T>` for `T: yuvxyb::Pixel` (= `u8`, `u16`
  — the only `Pixel` impls), and `Xyb` (which `src/lib.rs`'s own
  `test_ssimulacra2` feeds). **`Hsl` was reachable through the old
  `TryFrom` bound and is NOT covered** — nothing uses it; the workaround is
  `yuvxyb::LinearRgb::from(hsl)`.
- **Not a metric change.** `Yuv` routes through `yuvxyb::LinearRgb::try_from`,
  exactly as the deleted function did — *not* through our `Rgb` impl, which
  substitutes our own sRGB linearization for C++ parity. So `Yuv` and `Rgb`
  inputs of the same picture still linearize differently; that is carried over
  deliberately, not fixed here. `input.rs::yuv_tests::
  yuv_conversion_matches_yuvxyb_bit_for_bit` pins it.
- **The old infallible trait hid a reachable panic in the `Rgb` impl.** Its
  non-sRGB arm did `.expect("Rgb to LinearRgb conversion should not fail")`,
  and it does fail — H.273 TC=17 (`ST428`, digital cinema) has no `to_linear`
  in yuvxyb. Regression test:
  `input.rs::yuv_tests::unsupported_transfer_on_rgb_is_an_error_not_a_panic`.
- Reachable `Err` values, read out of yuvxyb 0.5.0's `yuv_rgb/color.rs` +
  `yuv_rgb/transfer/mod.rs` with the discriminants checked against
  `av-data 0.4.4`'s enums: **MC=3** `Reserved` and **MC=12**
  `ChromaticityDerivedNonConstantLuminance` (`get_yuv_constants` has no KR/KB
  for it); **TC=0/3** `Reserved0`/`Reserved`, **TC=12** `BT1361E`, **TC=17**
  `ST428`; plus unsupported `ColorPrimaries` under MC=0/10/11/13/14, which are
  the values routed to `ncl_rgb_to_yuv_matrix_from_primaries`.
  `MatrixCoefficients::Unspecified` (MC=2) is **not** one of them — `Yuv::new`
  rewrites it via `fix_unspecified_data`. Note MC=14 `ICtCp` and MC=8 `YCgCo`
  do *not* fail with BT.709/BT.2020 primaries, despite reading like they
  should; the first test written for this used ICtCp and passed.

## `ssimulacra2_320x240` is too noisy to draw conclusions from

Full record: `benchmarks/ssim2_perf/2026-08-31_x86_0.9.0.md`. On r7900x,
`cargo bench --bench benches` gives 7.5, 8.2, 7.5 ms for `ssimulacra2_320x240`
across **three runs of one unchanged binary** — a 9.3% spread. Every other case
(1080p, 4K, all three RGB sizes, `blur`) reproduces to ≤0.3% across runs *and*
across builds. So a 0.5–1.0 ms move at 320×240 between two builds is not
evidence of anything; a v0.8.2-vs-`f56991e` pair that read as "+8–9% regression"
did not reproduce on 0.9.0.

Cause is likely that zenbench prints `0 rounds ⚠ only 0 rounds` on every case —
it never completes its round budget — and 320×240 is the only benchmark whose
per-iteration time (7–8 ms) is short enough for that to dominate the mean. **Do
not quote a 320×240 delta without ≥3 runs per build.** To measure per-call fixed
overhead properly, fit `α + β·pixels` over ≥4 sizes on a build where zenbench
reports a completed round count.

## Who consumes fast-ssim2 in `~/work` (audited 2026-08-31)

- **Path deps — these compile against whatever is in this working tree, with no
  version bump:** `codec-eval/crates/codec-iter`,
  `zenmetrics/crates/zenmetrics-cli`, `glassa`. A break here reaches them
  immediately. None of the three call the removed `compute_frame_*` pair or
  implement `ToLinearRgb`, so 0.9.0 does not touch them.
- **Registry deps** (unaffected until someone widens the requirement):
  zenjpeg / codec-eval / zensim-bench / zenavif / heic `0.8.0`; zenmetrics-
  orchestrator `0.8.1`; ravif / zensr-bench / zencodecs / imageflow's jpeg-q
  harness `0.8.2`; zengif `0.7.1`; zenimage `0.6`.
- **The only workspace caller of the removed pair** is
  `zenjpeg/tests/bundled/edge_tile_ssim2_comparison.rs` (a `#[ignore]`d test
  behind `#[allow(deprecated)]`), on the `0.8.0` requirement. Migration is one
  line — it passes `LinearRgbImage`s, which implement `ToLinearRgb`, so
  `compute_frame_ssimulacra2(a, b)` → `compute_ssimulacra2(a, b)`.
  **The 2026-06-11 ablation report's "no external org consumers found" was
  wrong** — re-grep before repeating it.
- **Zero `ToLinearRgb` implementors exist outside this repo**, so the trait's
  fallibility break has no downstream implementor to migrate. Every other
  `compute_frame_ssimulacra2` hit in `~/work` (zensim-bench, ssim2-gpu,
  jpegli-rs) is the *external* rust-av `ssimulacra2` crate, not this one.

## `compare_tool/` does not build either (pre-existing, unrelated)

`cargo build --manifest-path compare_tool/Cargo.toml` fails with 3 errors, none
of them in code this repo controls:

- `src/main.rs:17` — `yuvxyb::Rgb::new` takes `NonZeroUsize` since yuvxyb 0.5.0;
  `compare_tool` still passes `usize`. It has been stale since that bump.
- `src/main.rs:38` — the *external* `ssimulacra2 0.5.1` crate (rust-av) pins
  `yuvxyb 0.4.2`, so `ssimulacra2::LinearRgb: TryFrom<yuvxyb::Rgb>` does not
  hold against our `yuvxyb 0.5.0`. Two versions of yuvxyb in one graph.

`compare_tool` is `exclude`d from the workspace and no CI job builds it. Its
0.9.0 call sites were updated anyway (`fast_ssim2::compute_frame_ssimulacra2` →
`compute_ssimulacra2`); the `ssimulacra2::compute_frame_ssimulacra2` call on
line 38 is the upstream crate's function and stays. Fixing the build needs a
yuvxyb-version decision, not an edit here.

## The `video` feature of `fast-ssim2-cli` does not build (pre-existing)

`cargo build -p fast-ssim2-cli --features video` fails with 8 errors, all in
`src/video.rs`: `av-metrics-decoders 0.3.2` pulls `v_frame 0.3.9` while
`yuvxyb 0.5.0` uses `v_frame 0.5.2`, so `Frame<S>` and the `Pixel` bound come
from two different crate versions. **Verified pre-existing at `f56991e`** (the
parent commit fails identically). No CI job builds it — CI only ever runs
`cargo clippy -p fast-ssim2-cli --all-targets` and `cargo test -p
fast-ssim2-cli` with default features. Do not assume `--all-features` works on
this workspace; fixing it needs a dependency bump, not a source edit.
