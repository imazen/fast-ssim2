# Libblur Backend Evaluation

## Summary

**Libblur is NOT compatible with the C++ SSIMULACRA2 reference implementation.**

While PR #28 (gembleman) showed 5-6x performance improvement with 0.001-1.1% accuracy loss, those measurements compared against the **old Rust implementation**, not the **C++ reference**. When tested against our 66 C++ reference test cases, libblur produces catastrophic errors.

## Test Results

### Baseline (Default: f64 IIR - Accurate)
- **Max error**: 0.954936
- **Tests passing**: 66/66 (100%)
- **Textured images**: 0.000 error (perfect match with C++)
- **Only errors**: Synthetic uniform_shift tests (acceptable)

### Libblur Backend
- **Max error**: 1041.641814 (1092x worse than baseline)
- **Tests failing**: 24/66 (36% failure rate)
- **WORST case**: `gradient_vs_uniform_64x64` → 1041.64 error
- **Uniform shift errors**: 2.13-34.77 (vs. baseline 0.18-0.95)

## Detailed Comparison

| Test Pattern | Baseline Max Error | Libblur Max Error | Degradation |
|--------------|-------------------|-------------------|-------------|
| uniform_shift | 0.954936 | 34.768670 | **36x worse** |
| gradients | 0.000000 | 0.000000 | Same |
| checkerboard | 0.000000 | 0.000000 | Same |
| noise | 0.000000 | 0.000000 | Same |
| edges | 0.000000 | 0.000000 | Same |
| synthetic_vs | 0.001332 | **1041.641814** | **781,000x worse!** |
| distortions | 0.120631 | 1.431356 | 12x worse |

## Why Libblur Fails

**ROOT CAUSE: Libblur uses completely different blur parameters than the C++ reference.**

| Parameter | C++ Reference | Libblur | Difference |
|-----------|---------------|---------|------------|
| Sigma | **1.5** | **2.2943** | +53% (!) |
| Algorithm | Charalampidis 2016 recursive | Standard Gaussian | Different math |
| Radius | 5 | kernel_size=11 | Different support |

**Verification**: Simple 8x8 gradient test shows libblur produces systematically wrong values:
- C++ reference: top-left corner = 0.730 (correct blur from value 0)
- Libblur: top-left corner = 1.729 (138% too high!)

The different sigma (1.5 vs 2.2943) combined with different algorithms means libblur is computing a **fundamentally different blur**, not an approximation of the C++ reference.

## PR #28 Measurement Baseline

PR #28 reported "0.001% to 1.1% accuracy change", but those measurements used the
prior Rust implementation as the reference rather than the C++ implementation:

1. The prior Rust implementation used f32 accumulators in the IIR filter, which caused
   divergence from the C++ reference
2. PR #28 tuned sigma=2.2943 to match that prior Rust implementation
3. After the f64 IIR fix brought the Rust implementation to 0.955 max error vs C++,
   it became clear that libblur's sigma was calibrated to a baseline that had already
   diverged from C++

The 5-6x speedup from libblur is real, but the sigma calibration targets the prior
Rust implementation rather than the C++ reference.

## Recommendation

**DO NOT use libblur backend** if you need:
- Compatibility with C++ SSIMULACRA2 implementation
- Reproducible scores across implementations
- Accurate perceptual quality measurement

**Libblur might be acceptable** if:
- You only care about relative comparisons within your own system
- You never need to compare scores with other implementations
- You understand scores will be completely different from reference

## Alternative: Transpose-Based Gaussian

The gembleman PR also included a transpose-based f32 IIR implementation. This should be evaluated separately as it:
- Uses the same recursive Gaussian algorithm as C++ (just f32 instead of f64)
- Likely has much smaller errors than libblur
- Still offers performance improvements via transpose trick + rayon

**TODO**: Evaluate transpose-based gaussian implementation against reference tests.

## Performance vs. Accuracy Tradeoff

```
┌────────────────────────────────────────────────────────────┐
│ Blur Backend Performance vs. Accuracy                      │
├────────────────────┬───────────────┬───────────────────────┤
│ Backend            │ Speed         │ Max Error vs. C++     │
├────────────────────┼───────────────┼───────────────────────┤
│ f64 IIR (default)  │ 1x (baseline) │ 0.955 ✅ ACCURATE    │
│ Transpose f32 IIR  │ ~4x faster    │ ??? (needs testing)   │
│ Libblur            │ ~6x faster    │ 1041.6 ❌ UNUSABLE   │
└────────────────────┴───────────────┴───────────────────────┘
```

## Conclusion

Libblur provides significant performance improvements but **fundamentally changes the SSIMULACRA2 metric**. Scores are not comparable to the C++ reference implementation or any other standard implementation.

For production use requiring accuracy and compatibility: **Use default f64 IIR backend.**

---

**Test Environment**:
- Date: 2026-01-03
- Branch: evaluate-libblur
- Libblur version: 0.17.5
- Test suite: 66 C++ reference cases
- Comparison: Against libjxl ssimulacra2 v2.1
