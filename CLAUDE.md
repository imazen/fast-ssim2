# fast-ssim2 Project Notes

## TODO

### CompareContext buffer pool for Ssimulacra2Reference

Design a `CompareContext` struct that holds reusable buffers for `Ssimulacra2Reference::compare()`. Goal: zero allocations after first use when comparing many distorted images against the same reference (simulated annealing, encoder tuning).

**Buffers to pool:**
- `Blur` struct (includes SIMD temp buffer)
- `mul: [Vec<f32>; 3]` working buffer
- `mu2`, `sigma2_sq`, `sigma12`: each `[Vec<f32>; 3]` — switch from `blur.blur()` to `blur_into()`
- `img2_planar: [Vec<f32>; 3]` — switch from `xyb_to_planar()` to `xyb_to_planar_into()`

**API shape:**
- `reference.compare_context() -> CompareContext` — factory, pre-sizes to reference dimensions
- `reference.compare_with(&self, ctx: &mut CompareContext, distorted) -> Result<f64>` — zero-alloc path
- `reference.compare(&self, distorted)` — convenience wrapper, allocates internally (unchanged)
- `CompareContext` is `Send` but not `Sync` — each thread owns one

**Also consider:**
- Hoisting the 6 SIMD blur state vectors (`prev_1..prev2_5` in `simd_gaussian.rs:247-252`) onto `SimdGaussian` to eliminate ~108 small allocations per frame
- Pooling downscale buffers (2 per scale > 0, 10 total) — trickier since they shrink each scale
- Whether `CompareContext` should also work with the non-precompute `compute_frame_ssimulacra2` path
