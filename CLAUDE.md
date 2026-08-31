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
- **Known upstream gap: `magetypes` does not fuse `mul_add` on wasm128 or on its
  scalar polyfill** (`impls/wasm128.rs`, `impls/scalar.rs`), while NEON/AVX2/
  AVX-512 emit a real FMA — and its own `f32x1::mul_add` uses `fmaf`. Every
  fusion-sensitive expression has been removed from our dispatched kernels
  *except* the blur, which stays fused because the reference fuses it (unfusing
  was measured: it costs 2.8x worse C++ agreement plus a -0.058 bias). So
  fast-ssim2 is bit-identical on all FMA-capable targets and up to 0.497 apart
  on wasm128 / no-SIMD builds until magetypes is fixed. **This is the one open
  item; it belongs in archmage, not here.**
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
  `compute_frame_ssimulacra2_with_config`.
