> **Historical (pre-archmage rewrite).** The API, feature flags, and benchmarks below are from the pre-archmage version and no longer apply. See README.md for current API and performance data.

# SSIMULACRA2 Rust Performance Benchmarks

Captured: 2026-01-04
Hardware: (add your CPU info)
Rust: (add rustc version)

## Runtime Implementations

Three runtime-selectable implementations via `Ssimulacra2Config`:

| Config | Blur | XYB | Compute | Notes |
|--------|------|-----|---------|-------|
| `scalar()` | f64 IIR | yuvxyb | scalar | Baseline, most accurate |
| `simd()` | wide crate | wide crate | wide crate | Default, portable SIMD |
| `unsafe_simd()` | raw AVX2 | raw AVX2 | raw AVX2 | Fastest, requires `unsafe-simd` feature |

## Feature Flags

```toml
[features]
default = ["simd", "unsafe-simd"]
simd = []           # Enable safe SIMD via wide crate
unsafe-simd = []    # Enable raw x86 intrinsics (AVX2/FMA)
rayon = ["dep:rayon"]  # Optional parallelism
```

## Full Pipeline Performance

Time in milliseconds for complete SSIMULACRA2 computation:

| Size | Scalar | SIMD | Unsafe-SIMD | Speedup |
|------|--------|------|-------------|---------|
| 512x512 | 130 ms | 38 ms | 30 ms | 4.3x |
| 1024x1024 | 530 ms | 185 ms | 155 ms | 3.4x |
| 1920x1080 (FHD) | 1050 ms | 410 ms | 360 ms | 2.9x |
| 2560x1440 (QHD) | 1870 ms | 675 ms | 610 ms | 3.1x |
| 3840x2160 (4K) | 4350 ms | 1570 ms | 1490 ms | 2.9x |

## Memory Usage

After optimization (commit 519a72e):

| Size | Scalar | SIMD | Unsafe-SIMD |
|------|--------|------|-------------|
| **Allocations** |
| 512x512 | 1618 | 58 | 58 |
| 4K | 6478 | 58 | 58 |
| **Memory (bytes/pixel)** |
| 512x512 | 230 | 228 | 228 |
| 4K | 180 | 179 | 179 |
| **Total Memory** |
| 4K | 1.39 GB | 1.38 GB | 1.38 GB |

Memory optimization achieved:
- Allocations: 252 → 58 (77% reduction)
- Memory: 2.16 GB → 1.38 GB at 4K (36% reduction)

## Accuracy vs C++ Reference

Tested against libjxl ssimulacra2 binary on JPEG-compressed images:

### JPEG Quality Test Images (256x256 crop, 15k unique colors)

| Quality | C++ Score | Scalar | SIMD | Unsafe-SIMD |
|---------|-----------|--------|------|-------------|
| Q20 | 57.146 | 57.121 | 57.068 | 57.050 |
| Q45 | 68.627 | 68.639 | 68.676 | 68.667 |
| Q70 | 79.388 | 79.518 | 79.507 | 79.472 |
| Q90 | 90.852 | 90.935 | 90.670 | 90.750 |

### Error vs C++ Reference

| Quality | Scalar | SIMD | Unsafe-SIMD |
|---------|--------|------|-------------|
| Q20 | -0.025 | -0.077 | -0.095 |
| Q45 | +0.011 | +0.048 | +0.040 |
| Q70 | +0.130 | +0.119 | +0.084 |
| Q90 | +0.083 | -0.182 | -0.101 |
| **Max Error** | **0.130** | **0.182** | **0.101** |

All implementations within 0.2 of C++ reference (tolerance: 1.5).

### 66 Synthetic Pattern Tests

From `tests/reference_parity.rs`:
- Max error vs C++: 0.955 (scalar baseline)
- All textured images: 0.000 error
- Only uniform color shifts show small FP differences

## API Usage

```rust
use ssimulacra2::{compute_frame_ssimulacra2_with_config, Ssimulacra2Config};

// Default (safe SIMD)
let score = compute_frame_ssimulacra2(source, distorted)?;

// Explicit configuration
let score = compute_frame_ssimulacra2_with_config(
    source,
    distorted,
    Ssimulacra2Config::unsafe_simd()  // or ::simd() or ::scalar()
)?;
```

## Blur-Only Performance

Gaussian blur is the dominant operation (~60% of runtime):

| Size | Scalar | SIMD | Unsafe-SIMD |
|------|--------|------|-------------|
| 512x512 | 8.5 ms | 2.2 ms | 1.8 ms |
| 1024x1024 | 35 ms | 8.5 ms | 7.2 ms |
| 4K | 290 ms | 95 ms | 85 ms |

## Parallelization: Not Beneficial

We evaluated scale-level parallelization (processing all 6 SSIM scales in parallel
with rayon). Results with unsafe-simd configuration:

| Size | Sequential | Parallel | Speedup | Allocations | Memory |
|------|------------|----------|---------|-------------|--------|
| 512x512 | 38.7 ms | 34.5 ms | 1.12x | 58 → 639 | +40% |
| 1024x1024 | 161.6 ms | 139.4 ms | 1.16x | 58 → 237 | +30% |
| FHD | 327.0 ms | 280.7 ms | 1.16x | 58 → 237 | +30% |
| QHD | 547.7 ms | 531.9 ms | 1.03x | 58 → 237 | +30% |
| 4K | 1706.1 ms | 1380.8 ms | 1.24x | 58 → 237 | +30% |

**Conclusion**: Scale-level parallelism provides only 3-24% speedup while increasing
memory usage by 30-40% (each parallel task needs its own buffers). The sequential
unsafe-SIMD implementation is already highly optimized; memory bandwidth becomes
the bottleneck at larger sizes, limiting parallel gains.

The `rayon` feature remains available for blur row parallelization, but scale-level
parallelism is not implemented as the trade-off is unfavorable.

## Build Commands

```bash
# Default (simd + unsafe-simd)
cargo build --release

# SIMD only (no unsafe)
cargo build --release --no-default-features --features simd

# With Rayon parallelism
cargo build --release --features rayon

# Run benchmarks
cargo run --release --example benchmark_allocations --features unsafe-simd
cargo run --release --example benchmark_unsafe_simd --features unsafe-simd
cargo run --release --example feature_benchmark --features unsafe-simd
```
