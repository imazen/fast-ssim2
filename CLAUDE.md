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
  benefit. **Unify on 0.8.2+**, on recency, not accuracy.
- **x86_64 and wasm128 are NOT MEASURED** — this was an aarch64 host. Deferred.
- The 3 "ignored" tests are 3 ```ignore doctest fences (`src/lib.rs` lines 10 and
  367, `src/strip.rs` line 72), not `#[ignore]` attributes. They are pseudo-code
  snippets (`load_image(...)`, `/* ... */`) and two of them need the `imgref`
  feature, which doctests do not build with. There are zero `#[ignore]`s.

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
- Reachable `Err` values, verified in yuvxyb 0.5.0 source: MC=3 `Reserved`,
  MC=12 `ChromaticityDerivedNonConstantLuminance`, TC=17 `ST428`, TC=0/3
  `Reserved0`/`Reserved`, TC=22 `BT1361E`, plus unsupported primaries under
  MC=0/9/13/14/15. `MatrixCoefficients::Unspecified` is **not** one of them —
  `Yuv::new` rewrites it via `fix_unspecified_data`.

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

## The `video` feature of `fast-ssim2-cli` does not build (pre-existing)

`cargo build -p fast-ssim2-cli --features video` fails with 8 errors, all in
`src/video.rs`: `av-metrics-decoders 0.3.2` pulls `v_frame 0.3.9` while
`yuvxyb 0.5.0` uses `v_frame 0.5.2`, so `Frame<S>` and the `Pixel` bound come
from two different crate versions. **Verified pre-existing at `f56991e`** (the
parent commit fails identically). No CI job builds it — CI only ever runs
`cargo clippy -p fast-ssim2-cli --all-targets` and `cargo test -p
fast-ssim2-cli` with default features. Do not assume `--all-features` works on
this workspace; fixing it needs a dependency bump, not a source edit.
