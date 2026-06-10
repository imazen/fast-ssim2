# fast-ssim2 Project Notes

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
