# fast-ssim2 [![CI](https://img.shields.io/github/actions/workflow/status/imazen/fast-ssim2/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/imazen/fast-ssim2/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/fast-ssim2?style=flat-square)](https://crates.io/crates/fast-ssim2) [![lib.rs](https://img.shields.io/crates/v/fast-ssim2?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/fast-ssim2) [![docs.rs](https://img.shields.io/docsrs/fast-ssim2?style=flat-square)](https://docs.rs/fast-ssim2) [![codecov](https://img.shields.io/codecov/c/github/imazen/fast-ssim2?style=flat-square)](https://codecov.io/gh/imazen/fast-ssim2) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/fast-ssim2?style=flat-square)](https://github.com/imazen/fast-ssim2#license)

Fast SIMD-accelerated Rust implementation of [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2), a perceptual image quality metric.

## Quick Start

```toml
[dependencies]
fast-ssim2 = { version = "0.8", features = ["imgref"] }
imgref = "1.12"   # for ImgVec/ImgRef — the input container fast-ssim2 takes
```

Most callers start from a flat, interleaved `Vec<u8>` (RGB8, row-major, no
padding). Wrap it in an `ImgVec<[u8; 3]>` and pass `.as_ref()`:

```rust
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;

// Your decoded pixels: width * height * 3 bytes, R,G,B,R,G,B, ...
let source_bytes: Vec<u8> = /* decoded RGB8 of the original */;
let distorted_bytes: Vec<u8> = /* decoded RGB8 of the compressed/modified version */;
let (width, height) = (1920, 1080);

// Group the flat byte stream into [R, G, B] pixels, then wrap with dimensions.
let to_img = |bytes: &[u8]| -> ImgVec<[u8; 3]> {
    let pixels: Vec<[u8; 3]> = bytes
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    ImgVec::new(pixels, width, height)
};
let source = to_img(&source_bytes);
let distorted = to_img(&distorted_bytes);

let score: f64 = compute_ssimulacra2(source.as_ref(), distorted.as_ref())?;
// 100 = identical, 90+ = imperceptible, <50 = significant degradation
```

The score is an `f64` on a **fixed 0–100 scale where higher is better** and
100 is a pixel-identical match (negative scores are possible for severe
distortion). It is *not* normalized to your inputs — the same number means the
same perceptual quality across every image pair. `u8`/`u16` pixels are treated
as **sRGB (gamma-encoded)**; `f32` pixels as **linear RGB** (see
[Input Types](#input-types)).

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

All comparison functions return `Result<f64, `[`Ssimulacra2Error`](https://docs.rs/fast-ssim2/latest/fast_ssim2/enum.Ssimulacra2Error.html)`>` — the score is an `f64` on the 0–100 scale above.

| Function | Use Case |
|----------|----------|
| [`compute_ssimulacra2`](https://docs.rs/fast-ssim2/latest/fast_ssim2/fn.compute_ssimulacra2.html) | Compare two images (recommended) |
| [`Ssimulacra2Reference::new`](https://docs.rs/fast-ssim2/latest/fast_ssim2/struct.Ssimulacra2Reference.html) | Precompute for batch comparisons (~2x faster) |
| [`Ssimulacra2Reference::compare_with`](https://docs.rs/fast-ssim2/latest/fast_ssim2/struct.Ssimulacra2Reference.html#method.compare_with) | Batch comparisons with a reusable [`CompareContext`](https://docs.rs/fast-ssim2/latest/fast_ssim2/struct.CompareContext.html) — zero allocations after the first call |
| [`compute_ssimulacra2_strip`](https://docs.rs/fast-ssim2/latest/fast_ssim2/fn.compute_ssimulacra2_strip.html) | Very large images with bounded peak memory (horizontal strips) — see [Bounded-Memory Strips](#bounded-memory-strips-very-large-images) |
| [`compute_ssimulacra2_with_stop`](https://docs.rs/fast-ssim2/latest/fast_ssim2/fn.compute_ssimulacra2_with_stop.html) | Cancellable comparison for servers (and the `*_strip_with_stop` / `compare_with_stop` variants) — see [Cooperative Cancellation](#cooperative-cancellation) |

### Input Types

With the `imgref` feature:

| Type | Color Space |
|------|-------------|
| `ImgRef<[u8; 3]>` | sRGB (8-bit) |
| `ImgRef<[u16; 3]>` | sRGB (16-bit) |
| `ImgRef<[f32; 3]>` | Linear RGB |
| `ImgRef<u8>`, `ImgRef<f32>` | Grayscale |

**Convention:** Integer types = sRGB gamma. Float types = linear RGB.

**RGBA / alpha:** there is no `[u8; 4]` (or `[u16; 4]` / `[f32; 4]`) input —
SSIMULACRA2 scores three color channels only. Drop the alpha channel to RGB
before wrapping:

```rust
// rgba: flat Vec<u8> of R,G,B,A,R,G,B,A, ...
let rgb: Vec<[u8; 3]> = rgba.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect();
let img = imgref::ImgVec::new(rgb, width, height);
```

If alpha is meaningful to your comparison (e.g. transparent regions), composite
both images over the same opaque background first, then drop alpha — comparing
straight (un-premultiplied) RGB ignores how transparency would actually render.

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

## Cooperative Cancellation

A server scoring untrusted or large images needs to abort an in-flight
comparison (request timeout, client disconnect, shutdown). Every slow path has
a `*_with_stop` variant that takes a cancellation token and returns
[`Ssimulacra2Error::Cancelled`] if it fires. The token is polled at the
per-scale (one-shot) / per-strip (strip) **outer-loop boundary — never
per-pixel** — so cancellation is responsive without adding any cost to the hot
path.

```toml
[dependencies]
enough = "0.4.4"          # the Stop trait + Unstoppable no-op
almost-enough = "0.4.4"   # a concrete, thread-safe Stopper you can cancel
```

The token is `&dyn enough::Stop`. Pass [`enough::Unstoppable`] for the
never-cancel path — it is indistinguishable in cost from the plain function:

```rust
use fast_ssim2::compute_ssimulacra2_with_stop;
use enough::Unstoppable;

let score: f64 = compute_ssimulacra2_with_stop(
    source.as_ref(),
    distorted.as_ref(),
    &Unstoppable, // never cancels — same result as compute_ssimulacra2
)?;
```

For a real cancellation, use an `almost_enough::Stopper` — it is `Clone` (an
8-byte `Arc<AtomicBool>` handle), so you score on one thread and cancel from
another (a timeout task, a signal handler, the request's drop guard):

```rust
use fast_ssim2::{compute_ssimulacra2_with_stop, Ssimulacra2Error};
use almost_enough::Stopper;

let stopper = Stopper::new();          // live; not yet cancelled
let cancel_handle = stopper.clone();   // hand this to your timeout/abort logic

// ... on timeout / client disconnect, from any thread:
// cancel_handle.cancel();

match compute_ssimulacra2_with_stop(source.as_ref(), distorted.as_ref(), &stopper) {
    Ok(score)                          => { /* use score: f64 */ }
    Err(Ssimulacra2Error::Cancelled(_)) => { /* aborted early */ }
    Err(e)                             => return Err(e),
}
```

> `Stopper::cancelled()` builds an already-fired token (handy for tests).
> For stronger cross-thread ordering guarantees use `almost_enough::SyncStopper`
> (same `new()` / `cancel()` shape, Acquire/Release instead of Relaxed).

The `*_with_stop` variants mirror the whole API surface:

| Cancellable function | Non-cancellable equivalent |
|----------------------|-----------------------------|
| `compute_ssimulacra2_with_stop(source, distorted, &stop)` | `compute_ssimulacra2` |
| `compute_ssimulacra2_strip_with_stop(source, distorted, strip_height, &stop)` | `compute_ssimulacra2_strip` |
| `Ssimulacra2Reference::compare_with_stop(&self, distorted, &stop)` | `compare` |
| `Ssimulacra2Reference::compare_strip_with_stop(&self, distorted, strip_height, &stop)` | `compare_strip` |

All return `Result<f64, Ssimulacra2Error>`. Because `Ssimulacra2Error` is
`#[non_exhaustive]`, `match` arms over it need a wildcard `_ =>`.

## Bounded-Memory Strips (very large images)

The full-image path allocates roughly `24 × width × height × 4` bytes of
working memory (~7 GiB at 40 MP). For very large images, the strip API
processes the image in horizontal strips and bounds peak memory to
`~24 × width × (strip_height + halo) × 4` bytes (~220 MiB at 40 MP with
`strip_height = 256`):

```rust
use fast_ssim2::compute_ssimulacra2_strip;

let strip_height: u32 = 256; // rows per strip's interior at scale 0
let score: f64 = compute_ssimulacra2_strip(source.as_ref(), distorted.as_ref(), strip_height)?;
```

Signatures:

```rust
pub fn compute_ssimulacra2_strip<S, D>(source: S, distorted: D, strip_height: u32)
    -> Result<f64, Ssimulacra2Error>
where S: ToLinearRgb, D: ToLinearRgb;

// On a precomputed reference (batch):
impl Ssimulacra2Reference {
    pub fn compare_strip<T: ToLinearRgb>(&self, distorted: T, strip_height: u32)
        -> Result<f64, Ssimulacra2Error>;
}
```

`strip_height` is the interior row count at scale 0; the working strip is
`strip_height + 2 * halo_rows` tall (`halo_rows` defaults to `HALO_ROWS_DEFAULT`,
configurable via `Ssimulacra2StripConfig`). Strip scores match the full-image
path to within ~1e-5 on the 0–100 scale. **Unlike the one-shot path, the strip
APIs do not reflect-pad** — they target very large images and return
`InvalidImageSize` for inputs below 8×8 or `strip_height < 8`; use
[`compute_ssimulacra2`](https://docs.rs/fast-ssim2/latest/fast_ssim2/fn.compute_ssimulacra2.html)
for tiny inputs.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `imgref` | No | Support for `imgref` image types |
| `rayon` | No | Parallel computation |
| `hdr-pu` | No | Experimental: HDR scoring via the PU21 (banding_glare) encoding; input is absolute-luminance linear RGB in cd/m² |

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

- **Image size:** 1x1 up to 16384x16384-equivalent pixels (`MAX_IMAGE_PIXELS`); inputs below the metric's 8x8 pyramid floor are reflect(mirror)-padded. The strip APIs (`compute_ssimulacra2_strip`, `compare_strip`) target very large images and require at least 8x8.
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
[`enough::Unstoppable`]: https://docs.rs/enough/latest/enough/struct.Unstoppable.html
[`Ssimulacra2Error::Cancelled`]: https://docs.rs/fast-ssim2/latest/fast_ssim2/enum.Ssimulacra2Error.html#variant.Cancelled
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
