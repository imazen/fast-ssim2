# fast-ssim2

[![Build Status](https://img.shields.io/github/actions/workflow/status/imazen/fast-ssim2/rust.yml?branch=main&style=for-the-badge)](https://github.com/imazen/fast-ssim2/actions/workflows/rust.yml)
[![docs.rs](https://img.shields.io/docsrs/fast-ssim2?style=for-the-badge)](https://docs.rs/fast-ssim2)
[![Crates.io](https://img.shields.io/crates/v/fast-ssim2?style=for-the-badge)](https://crates.io/crates/fast-ssim2)
[![codecov](https://img.shields.io/codecov/c/github/imazen/fast-ssim2?style=for-the-badge)](https://codecov.io/gh/imazen/fast-ssim2)
[![LICENSE](https://img.shields.io/crates/l/fast-ssim2?style=for-the-badge)](https://github.com/imazen/fast-ssim2/blob/main/LICENSE)

Fast SIMD-accelerated Rust implementation of [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2), a perceptual image quality metric.

## Quick Start

```toml
[dependencies]
fast-ssim2 = { version = "0.7", features = ["imgref"] }
```

```rust
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;

let source: ImgVec<[u8; 3]> = /* your source image */;
let distorted: ImgVec<[u8; 3]> = /* compressed/modified version */;

let score = compute_ssimulacra2(source.as_ref(), distorted.as_ref())?;
// 100 = identical, 90+ = imperceptible, <50 = significant degradation
```

## Score Interpretation

| Score | Quality |
|-------|---------|
| **100** | Identical |
| **90+** | Imperceptible difference |
| **70-90** | Minor, subtle difference |
| **50-70** | Noticeable difference |
| **<50** | Significant degradation |

## API Overview

### Primary Functions

| Function | Use Case |
|----------|----------|
| [`compute_ssimulacra2`](https://docs.rs/fast-ssim2/latest/fast_ssim2/fn.compute_ssimulacra2.html) | Compare two images (recommended) |
| [`Ssimulacra2Reference::new`](https://docs.rs/fast-ssim2/latest/fast_ssim2/struct.Ssimulacra2Reference.html) | Precompute for batch comparisons (~2x faster) |

### Input Types

With the `imgref` feature:

| Type | Color Space |
|------|-------------|
| `ImgRef<[u8; 3]>` | sRGB (8-bit) |
| `ImgRef<[u16; 3]>` | sRGB (16-bit) |
| `ImgRef<[f32; 3]>` | Linear RGB |
| `ImgRef<u8>`, `ImgRef<f32>` | Grayscale |

**Convention:** Integer types = sRGB gamma. Float types = linear RGB.

Without features, use `yuvxyb::Rgb` or `yuvxyb::LinearRgb`, or implement [`ToLinearRgb`](https://docs.rs/fast-ssim2/latest/fast_ssim2/trait.ToLinearRgb.html) for custom types.

## Batch Comparisons

When comparing multiple images against the same reference (e.g., testing compression levels), precompute the reference:

```rust
use fast_ssim2::Ssimulacra2Reference;

let reference = Ssimulacra2Reference::new(source.as_ref())?;

for distorted in compressed_variants {
    let score = reference.compare(distorted.as_ref())?;
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `imgref` | No | Support for `imgref` image types |
| `rayon` | No | Parallel computation |

SIMD is always available — runtime CPU detection via [archmage](https://crates.io/crates/archmage) selects the best backend automatically (AVX2+FMA on x86_64, NEON on aarch64, SIMD128 on wasm32, scalar fallback elsewhere).

## Performance

Benchmarked on AMD Ryzen 9 7950X (x86_64, AVX2+FMA), full SSIMULACRA2 computation vs upstream [ssimulacra2](https://crates.io/crates/ssimulacra2) crate:

| Resolution | ssimulacra2 (scalar) | fast-ssim2 (SIMD) | Speedup |
|------------|---------------------|-------------------|---------|
| 320x240 | 139ms | 8.7ms | **16x** |
| 1920x1080 | 1,006ms | 316ms | **3.2x** |
| 3840x2160 | 3,615ms | 1,317ms | **2.7x** |

Run your own benchmarks:
```bash
cargo bench -p fast-ssim2
```

## Advanced Usage

### Custom Input Types

```rust
use fast_ssim2::{ToLinearRgb, LinearRgbImage, srgb_u8_to_linear};

struct MyImage { /* ... */ }

impl ToLinearRgb for MyImage {
    fn to_linear_rgb(&self) -> LinearRgbImage {
        let data: Vec<[f32; 3]> = self.pixels.iter()
            .map(|[r, g, b]| [
                srgb_u8_to_linear(*r),
                srgb_u8_to_linear(*g),
                srgb_u8_to_linear(*b),
            ])
            .collect();
        LinearRgbImage::new(data, self.width, self.height)
    }
}
```

### Explicit SIMD Backend

```rust
use fast_ssim2::{compute_ssimulacra2_with_config, Ssimulacra2Config};

// Force scalar (for comparison/debugging)
let score = compute_ssimulacra2_with_config(source, distorted, Ssimulacra2Config::scalar())?;

// Use SIMD (default — auto-detects AVX2/NEON/WASM128)
let score = compute_ssimulacra2_with_config(source, distorted, Ssimulacra2Config::simd())?;
```

### Using yuvxyb Types Directly

```rust
use fast_ssim2::{compute_ssimulacra2, Rgb, TransferCharacteristic, ColorPrimaries};

let source = Rgb::new(
    pixel_data,
    width,
    height,
    TransferCharacteristic::SRGB,
    ColorPrimaries::BT709,
)?;
let score = compute_ssimulacra2(source, distorted)?;
```

## Requirements

- **Minimum image size:** 8x8 pixels
- **MSRV:** 1.89.0

## Attribution

Fork of [rust-av/ssimulacra2](https://github.com/rust-av/ssimulacra2). Thank you to the rust-av team for the original implementation.

**What's different:** Cross-platform SIMD acceleration (x86_64/aarch64/wasm32 via [archmage](https://crates.io/crates/archmage)), precomputed reference API, `imgref` support, `#![forbid(unsafe_code)]`.

## License

BSD-2-Clause (same as upstream)

---

Developed with assistance from Claude (Anthropic). Tested against the C++ reference implementation.
