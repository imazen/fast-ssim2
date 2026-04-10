# fast-ssim2 [![CI](https://img.shields.io/github/actions/workflow/status/imazen/fast-ssim2/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/imazen/fast-ssim2/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/fast-ssim2?style=flat-square)](https://crates.io/crates/fast-ssim2) [![lib.rs](https://img.shields.io/crates/v/fast-ssim2?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/fast-ssim2) [![docs.rs](https://img.shields.io/docsrs/fast-ssim2?style=flat-square)](https://docs.rs/fast-ssim2) [![codecov](https://img.shields.io/codecov/c/github/imazen/fast-ssim2?style=flat-square)](https://codecov.io/gh/imazen/fast-ssim2) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/fast-ssim2?style=flat-square)](https://github.com/imazen/fast-ssim2#license)

Fast SIMD-accelerated Rust implementation of [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2), a perceptual image quality metric.

## Quick Start

```toml
[dependencies]
fast-ssim2 = { version = "0.8", features = ["imgref"] }
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

Without `imgref`, use `yuvxyb::Rgb` or `yuvxyb::LinearRgb` (add `yuvxyb` to your own dependencies), or implement [`ToLinearRgb`](https://docs.rs/fast-ssim2/latest/fast_ssim2/trait.ToLinearRgb.html) for custom types.

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
use fast_ssim2::compute_ssimulacra2;
use yuvxyb::{Rgb, TransferCharacteristic, ColorPrimaries};

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

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · **fast-ssim2** · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

BSD-2-Clause (same as upstream)



### Upstream Contribution

This is a fork of [rust-av/ssimulacra2](https://github.com/rust-av/ssimulacra2) (BSD-2-Clause).
We are willing to release our improvements under the original BSD-2-Clause
license if upstream takes over maintenance of those improvements. We'd rather
contribute back than maintain a parallel codebase. Open an issue or reach out.

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
