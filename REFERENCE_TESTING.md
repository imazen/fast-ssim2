# SSIMULACRA2 Reference Testing

This document explains how to verify the Rust implementation against the C++ reference.

## Quick Reference: Updating Reference Numbers

**TL;DR** - To regenerate reference data from C++:

```bash
# 1. Ensure C++ binary is built (see "Build C++ SSIMULACRA2" below if needed)
export SSIMULACRA2_BIN=/path/to/libjxl/build/tools/ssimulacra2

# 2. Navigate to ssimulacra2 crate
cd ssimulacra2

# 3. Generate reference data (overwrites src/reference_data.rs)
cargo run --release --example capture_cpp_reference

# 4. Verify tests pass
cargo test --release --test reference_parity -- --nocapture

# 5. Review the detailed variance report, then commit
git add src/reference_data.rs
git commit -m "chore: update reference data from C++ ssimulacra2"
```

**When to update**:
- ✅ C++ ssimulacra2 version updated
- ✅ Algorithm intentionally changed in Rust port
- ❌ **NOT** when tests are failing (fix the bug first!)

## Overview

The Rust ssimulacra2 implementation is tested against reference values from the C++ implementation to ensure correctness. This uses a two-step process:

1. **Capture**: Generate test images and capture scores from C++ ssimulacra2
2. **Test**: Run tests that compare Rust scores against captured C++ scores

## Prerequisites

### Build C++ SSIMULACRA2

The ssimulacra2 tool is available in the libjxl repository:

```bash
# If you don't have libjxl cloned:
git clone https://github.com/libjxl/libjxl.git
cd libjxl

# Build with DEVTOOLS enabled to get ssimulacra2
cmake -B build -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_DEVTOOLS=ON
cmake --build build --target ssimulacra2 -j$(nproc)

# Verify binary works
./build/tools/ssimulacra2 --help
```

The binary will be at `./build/tools/ssimulacra2`.

**Note**: The cloudinary/ssimulacra2 repository requires lcms2 >= 2.13, which may not be available on all systems. Using libjxl's version avoids this dependency issue.

## Capturing Reference Data

Run the capture tool to generate reference data:

```bash
cd ssimulacra2  # Assumes you're in the ssimulacra2 crate directory

# Set path to C++ binary (adjust path to your libjxl build)
export SSIMULACRA2_BIN=/path/to/libjxl/build/tools/ssimulacra2

# Capture reference data (generates src/reference_data.rs)
cargo run --release --example capture_cpp_reference
```

This will:
1. Generate 66 synthetic test images (gradients, noise, patterns, distortions)
2. Save them as PNGs in `/tmp/ssimulacra2_reference/`
3. Call the C++ binary to get reference scores
4. Compute SHA256 hashes of source images (detects generation changes)
5. Generate `src/reference_data.rs` with all reference values and hashes

### Test Patterns Generated

| Pattern | Variations | Purpose |
|---------|-----------|---------|
| Perfect match | 4 sizes | Should score 100.0 |
| Uniform + shift | 5 shifts × 4 sizes | Test uniform color distortion (FP precision) |
| Gradients | H/V/Diag × 4 sizes | Test smooth transitions |
| Checkerboard | 3 cell sizes × 4 sizes | Test high-frequency patterns |
| Random noise | 3 seeds × 4 sizes | Test noise handling |
| Edges | Vertical × 4 sizes | Test sharp transitions |
| Synthetic pairs | 2 pairs | Test non-identical synthetic images |
| **Distortions** | 4 realistic | **Box blur, sharpen, YUV roundtrip** |

**Total**: 66 test cases (62 original + 4 distortions)

## Running Reference Tests

Once reference data is captured:

```bash
# Run reference parity tests (with detailed variance report)
cargo test --release --test reference_parity -- --nocapture

# Expected output:
# All 66 reference tests passed! Max error: 0.954936
```

### What this table can and cannot prove

**40 of the 66 cases compare an image against itself.** Every implementation
returns exactly `100.0` for an identical pair, whatever its arithmetic, so the
zero error on `perfect_match`, `gradients`, `checkerboard`, `noise` and `edges`
is a constant, not evidence of parity. The informative cases are the other 26,
and 20 of those are `uniform_shift`, which is numerically degenerate (below).

For parity evidence on content where the metric is well conditioned, run
`examples/photo_parity.rs` against a real corpus. Measured 2026-08-31 on 576
photographic pairs (CID22, KADID-10k, gb82, CLIC2025 x 4 sizes x 6 distortions):
mean |fast-ssim2 - C++| = **0.024**, 283/576 deltas positive (no bias).

### Per-Pattern Tolerances

| Pattern | Tolerance | Max error measured 2026-08-31 |
|---|--:|--:|
| `uniform_shift` | 10.0 | 1.997 |
| distortions (`boxblur8x8`, `sharpen`, `yuv_roundtrip`) | 0.2 | 0.182 |
| `_vs_`, `perfect_match`, `gradient_h`/`gradient_v`, `checkerboard`, `noise_seed`, `edge_*` | 0.01 | 0.002 |
| fallback | 0.05 | — |

### Why `uniform_shift` gets 10.0

Not because the SIMD paths round differently — they agree with each other to
~1e-8. The disagreement is with the C++ reference, and it is one-directional:
**18 of the 20 cases score higher than the reference, mean +0.408.** FP noise
does not do that.

On a flat field the SSIM' term is degenerate. `sigma11 - mu1^2`,
`sigma22 - mu2^2` and `sigma12 - mu1*mu2` are analytically zero away from the
border, so `num_s` and `denom_s` both collapse onto `kC2 = 9e-4` plus whatever
the f32 recursive Gaussian left behind — a residual of ~5e-7, about 1/1600 of
the term it perturbs. Measured at the centre of `uniform_shift_1_32x32`: the
true per-pixel error is 1.11e-5 and rounding contributes 3.31e-5, three times
more; 60% of pixels have `|d - d_exact| > d_exact`; at 256x256 the centre
pixel's `d` clamps to exactly 0. `d = max(..., 0)` rectifies the residual so it
cannot cancel, and whichever implementation has the noisier blur reports the
lower score.

Three independent confirmations:

1. The reference's own values are **not monotonic in the shift** — at 32x32,
   shift 1 -> 97.749 but shift 5 -> 98.808.
2. Perturbing any single stage by ~1e-7 moves the score by ~0.5. Swapping only
   the cube root (ours, 1.75 ulp, for jpegli's own `CubeRootAndAdd`, 3.34 ulp)
   moves `uniform_shift_1_32x32` and `uniform_shift_5_32x32` in *opposite*
   directions relative to the reference.
3. Transliterating jpegli's own 4-unrolled horizontal Gaussian into our
   pipeline cuts the mean error on this family from 0.381 to 0.258, and at
   64x64 brings three of five cases within 0.008. That also means **the C++
   reference is not arch-consistent with itself** — its horizontal blur differs
   between `HWY_SCALAR` and its vector targets.

So this family cannot discriminate implementations, and the wide tolerance
reflects that the *reference* is uninformative here. Do not tighten it into a
pass/fail on a noise measurement; add well-conditioned cases instead.

Full record, with every number and how it was obtained:
`benchmarks/cpp_parity_2026-08-31.md`. Stage-level diagnostics:
`cargo test --release --lib cpp_parity_diag -- --nocapture`.

### Detailed Variance Report

The test output includes:
- **Top 10 largest errors** with actual vs expected scores
- **Error breakdown by pattern type** (count, max, mean, P95)
- **Error percentiles** (p50, p90, p95, p99)

Example output (2026-08-31, aarch64, reference binary jpeg-xl 0.12.0):
```
================================== REFERENCE PARITY TEST RESULTS ===================================
All 66 reference tests passed! Max error: 1.996955

Error percentiles: p50=0.0000, p90=0.4444, p95=0.7762, p99=1.9970
Errors >0.1: 15, >0.5: 6, >1.0: 1

--------------------------------- Error Breakdown by Pattern Type ----------------------------------
Pattern                   Count       Max Error      Mean Error       P95 Error
--------------------------------------------------------------------------------
checkerboard                 12        0.000000        0.000000        0.000000
distortions                   4        0.181726        0.066305        0.181726
edges                         4        0.000000        0.000000        0.000000
gradients                     8        0.000000        0.000000        0.000000
noise                        12        0.000000        0.000000        0.000000
perfect_match                 4        0.000000        0.000000        0.000000
synthetic_vs                  2        0.001659        0.001099        0.001659
uniform_shift                20        1.996955        0.407777        1.996955
```

For the **signed** version of this table, which is what distinguishes noise
from bias, run `cargo run --release --example parity_report` (set
`SSIMULACRA2_BIN` to re-verify the captured table against a live binary at the
same time).

### SHA256 Hash Verification

Before testing scores, the test verifies that image generation hasn't changed by comparing SHA256 hashes of generated images against captured hashes. This catches:
- Changes in RNG libraries (e.g., LCG implementation updates)
- Accidental modifications to image generation functions
- Platform-specific image generation differences

If hashes mismatch, the test will fail with:
```
ERROR: Source image hash mismatch for gradient_h_64x64!
Expected: abc123...
Got:      def456...
This indicates the image generation algorithm changed.
```

### Troubleshooting Failed Tests

If tests fail:
1. **Check error magnitude**: Is it outside tolerance for that pattern type?
2. **Review recent changes**: Did algorithm changes affect scores?
3. **Verify hash matches**: If hashes fail, image generation changed (need to re-capture)
4. **Check C++ binary version**: Ensure using same C++ version as capture

**Do NOT** loosen tolerances to make tests pass - fix the underlying bug instead.

## Continuous Integration

### GitHub Actions Example

```yaml
name: Reference Parity Tests

on: [push, pull_request]

jobs:
  parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install C++ dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake build-essential libhwy-dev libjpeg-dev libpng-dev

      - name: Build C++ ssimulacra2
        run: |
          git clone https://github.com/libjxl/libjxl.git /tmp/libjxl
          cd /tmp/libjxl
          cmake -B build -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_DEVTOOLS=ON
          cmake --build build --target ssimulacra2 -j$(nproc)

      - name: Capture reference data
        run: |
          export SSIMULACRA2_BIN=/tmp/libjxl/build/tools/ssimulacra2
          cargo run --release --example capture_cpp_reference

      - name: Run parity tests
        run: cargo test --release --test reference_parity
```

## Comparison to Current Tests

### Before (Current)

```rust
#[test]
fn test_ssimulacra2() {
    let result = compute_frame_ssimulacra2(source, distorted).unwrap();
    let expected = 17.398_505_f64;
    assert!(
        // SOMETHING is WEIRD with Github CI where it gives different results
        (result - expected).abs() < 0.25f64,  // ❌ Loose tolerance!
        "Result {result:.6} not equal to expected {expected:.6}",
    );
}
```

**Problems**:
- Single test case
- Value not verified against C++ (self-generated)
- Loose tolerance (0.25 = 1.4% error)
- CI instability noted

### After (With Reference Data)

```rust
#[test]
fn test_reference_parity() {
    for case in REFERENCE_CASES {
        let score = compute_frame_ssimulacra2(source, distorted).unwrap();

        // Per-pattern tolerance based on observed error characteristics
        let tolerance = if case.name.contains("uniform_shift") {
            1.2  // IIR filter FP precision
        } else if case.name.contains("boxblur8x8") || ... {
            0.15 // Distortion operations
        } else if case.name.contains("_vs_") {
            0.002 // Synthetic non-identical
        } else {
            0.001 // Identical patterns
        };

        assert!(
            (score - case.expected_score).abs() < tolerance,
            "{}: expected {}, got {}, error {}",
            case.name,
            case.expected_score,
            score,
            (score - case.expected_score).abs()
        );
    }
}
```

**Benefits**:
- **66 test cases** covering diverse patterns + realistic distortions
- All verified against C++ reference implementation
- **Per-pattern tolerances** matched to error characteristics
- **SHA256 hash verification** catches image generation changes
- **Detailed variance report** shows actual vs expected for all cases
- Clear test names and error messages
- Auto-generated and reproducible

## Troubleshooting

### "ssimulacra2 binary not found"

Set `SSIMULACRA2_BIN`:
```bash
export SSIMULACRA2_BIN=/path/to/ssimulacra2
```

Or ensure binary is in `PATH`:
```bash
export PATH=/path/to/build:$PATH
cargo run --example capture_cpp_reference
```

### "Could not parse score from output"

The C++ binary output format might have changed. Check:
```bash
$SSIMULACRA2_BIN source.png distorted.png
```

Expected format: Last number on each line is the score.

### Test failures after capturing

If tests fail immediately after capturing:
1. This indicates Rust implementation differs from C++
2. Check recent code changes
3. Compare specific failing cases to isolate issue

### Platform differences

Floating-point operations may differ slightly across platforms. If you see small differences (< 1e-5):
- Acceptable for different architectures (x86 vs ARM)
- Re-capture on target platform for CI

## Maintenance

### When to Re-capture

Re-capture reference data when:
- ✅ C++ ssimulacra2 updates to new version
- ✅ Algorithm intentionally changes
- ✅ Platform changes (x86 → ARM, etc.)

Do NOT re-capture when:
- ❌ Tests are failing (fix the bug first!)
- ❌ Just to make tests pass (indicates regression)

### Version Tracking

Document reference data version in `reference_data.rs` header:
```rust
//! Generated from: cloudinary/ssimulacra2 v2.1
//! Date: 2026-01-03
//! Platform: x86_64-unknown-linux-gnu
```

## Investigation Findings: What Didn't Work

This section documents attempted fixes that **did not improve** parity with C++. This prevents wasted effort re-investigating these approaches.

### Summary: Error Reduction Journey

| Attempt | Max Error | Result | Reason |
|---------|-----------|--------|--------|
| Baseline (all f32) | 1.16 | - | Starting point |
| Downscaling f64 normalization | 1.16 | ❌ No effect | Not the source of error |
| **Horizontal IIR f64** | **0.955** | ✅ **-18%** | **Found the main issue!** |
| SSIM computation f64 | 0.955 | ❌ No effect | Error is in blur, not SSIM |
| Vertical IIR f64 | 1.984 | ❌ **Worse!** | Precision mismatch between passes |

### Failed Attempt #1: Downscaling with f64 Normalization

**Hypothesis**: Downscaling by 2x accumulates rounding errors when averaging 4 pixels.

**What we tried** (`src/lib.rs:175-192`):
```rust
pub(crate) fn downscale_by_2(in_data: &LinearRgb) -> LinearRgb {
    // Use f64 accumulator to reduce rounding errors
    let mut sum = 0f64;
    for iy in 0..SCALE {
        for ix in 0..SCALE {
            sum += f64::from(in_pix[c]);
        }
    }
    out_pix[c] = (sum / (SCALE * SCALE) as f64) as f32;
}
```

**Result**: No change in error (still 1.16)

**Why it didn't help**: The downscaling is already numerically stable. Averaging 4 values doesn't accumulate significant error. The issue was elsewhere in the pipeline.

### Failed Attempt #2: SSIM Computation with f64

**Hypothesis**: SSIM computation has rounding errors when dividing small differences.

**What we tried** (`src/lib.rs:242-248`):
```rust
// Use f64 for SSIM computation to reduce rounding errors
let num_m = f64::from(mu_diff).mul_add(-f64::from(mu_diff), 1.0f64);
let num_s = 2f64.mul_add(f64::from(row_s12[x] - mu12), f64::from(C2));
let denom_s = f64::from(row_s11[x] - mu11) + f64::from(row_s22[x] - mu22) + f64::from(C2);
let mut d = 1.0f64 - (num_m * num_s) / denom_s;
```

**Result**: No change in error (still 0.955)

**Why it didn't help**: By this point in the pipeline, the blur has already introduced the errors. Computing SSIM in higher precision doesn't undo errors from earlier stages.

### Failed Attempt #3: Vertical IIR Filter with f64

**Hypothesis**: Both horizontal and vertical IIR filters should use f64 for consistency.

**What we tried** (`src/blur/gaussian.rs:125-187`):
```rust
pub fn vertical_pass<const COLUMNS: usize>(...) {
    // Use f64 accumulators to reduce rounding error accumulation
    let mut prev = vec![0f64; 3 * COLUMNS];
    let mut prev2 = vec![0f64; 3 * COLUMNS];
    let mut out = vec![0f64; 3 * COLUMNS];

    // Convert to f64 for accumulation
    let sum = f64::from(top_row[i]) + f64::from(bottom_row[i]);
    // ... rest of computation in f64
}
```

**Result**: Error **increased** from 0.955 to **1.984** (worse than baseline!)

**Why it made things worse**:
- Creates precision **mismatch** between horizontal (f64) and vertical (f64) passes
- The horizontal pass outputs f32, which gets read by vertical pass
- Mixing f32 outputs with f64 accumulators causes different rounding than C++
- Multi-scale processing (6 scales) compounds these differences
- The horizontal and vertical filters need **precision consistency**

**Lesson**: When fixing numerical issues, changing one part of a pipeline can make overall results worse if other parts aren't changed compatibly.

### Why Only Horizontal f64 Works

The successful fix was **horizontal IIR f64 only**, which:
1. Reduces accumulation errors in the primary scan direction
2. Still outputs f32, maintaining compatibility with vertical pass
3. Keeps vertical pass at f32 to match C++ behavior
4. Achieves consistency: horizontal uses f64 internally but interfaces in f32

This creates a "best of both worlds" - better precision where it matters, but maintains interface compatibility.

### What We Learned

**Error pattern analysis is crucial**:
- The fact that **all textured patterns had 0.000 error** told us:
  - The blur algorithm is fundamentally correct
  - Errors are specific to uniform images (no texture to mask rounding)
  - The IIR filter accumulation was the likely culprit

**Precision changes can backfire**:
- Changing one component to f64 doesn't always help
- Pipeline stages need precision **consistency**
- Mixed precision can introduce worse errors than consistent f32

**C++ uses f32 throughout**:
- C++ implementation uses `float` for all blur operations
- C++'s HWY SIMD might have different rounding behavior
- The 0.955 remaining error might be unavoidable without SIMD

### Current State: Production Ready

**Remaining 0.955 error is acceptable because**:
- Only affects synthetic uniform color images (no real-world impact)
- All textured patterns match exactly (0.000 error)
- Error is within tolerance (1.2 for uniform_shift)
- Further improvements require SIMD or major architectural changes

**To pursue exact parity, future work could**:
1. Port C++ HWY SIMD implementation for bit-exact matching
2. Investigate compiler-specific FMA (fused multiply-add) behavior
3. Test on different platforms (ARM, RISC-V) to isolate x86-specific behavior
4. Compare C++ builds on different platforms to see if C++ also varies

But for a pure-Rust scalar implementation, **0.955 max error is excellent**.
