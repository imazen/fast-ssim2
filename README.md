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
fast-ssim2 = { version = "0.6", features = ["imgref"] }
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
| `simd` | Yes | Safe SIMD via `wide` crate |
| `unsafe-simd` | Yes | x86_64 AVX2 intrinsics (fastest) |
| `imgref` | No | Support for `imgref` image types |
| `rayon` | No | Parallel computation |

## Performance

Benchmarked on AMD Ryzen (x86_64), full SSIMULACRA2 computation:

| Resolution | Scalar | SIMD | Unsafe SIMD |
|------------|--------|------|-------------|
| 1920x1080 | 1083ms | 434ms (2.5x) | 370ms (2.9x) |
| 3840x2160 | 4256ms | 1612ms (2.6x) | 1422ms (3.0x) |

Run your own benchmarks:
```bash
cargo run --release --features "simd unsafe-simd" --example benchmark_unsafe_simd
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

// Force scalar (most portable)
let score = compute_ssimulacra2_with_config(source, distorted, Ssimulacra2Config::scalar())?;

// Force unsafe SIMD (fastest on x86_64)
#[cfg(feature = "unsafe-simd")]
let score = compute_ssimulacra2_with_config(source, distorted, Ssimulacra2Config::unsafe_simd())?;
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

**What's different:** SIMD acceleration (via `wide` crate and x86 intrinsics), precomputed reference API, `imgref` support. These come with trade-offs: higher MSRV and more complex code.

## License

BSD-2-Clause (same as upstream)

---

Developed with assistance from Claude (Anthropic). Tested against the C++ reference implementation.
