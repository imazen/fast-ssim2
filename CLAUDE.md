# fast-ssim2 Project Notes

## Current state (2026-06-10)

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
