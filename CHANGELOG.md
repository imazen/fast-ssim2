## [Unreleased]

### Added
- `LinearRgbImage::try_new` fallible constructor returning `LinearRgbImageError` for invalid dimensions or data length
- `Ssimulacra2Error::ImageTooLarge` variant and public `MAX_IMAGE_PIXELS` constant (16384*16384) capping caller-supplied image size to prevent unbounded working-buffer allocation

### Fixed
- `LinearRgbImage::new` now validates dimensions and data length at runtime (was `debug_assert_eq!` only) so release-mode misuse no longer constructs malformed images that panic deep in `From<LinearRgbImage> for yuvxyb::LinearRgb`
- `SimdGaussian::new` no longer eagerly allocates `max_width * 4096` floats; the temp buffer grows on demand. Also guards against `usize` overflow on 32-bit targets when `width * height` would wrap

## Version 0.7.3

- Add proper CI workflow with full platform matrix (Linux, macOS, Windows on x64/ARM64), i686 cross testing, WASM testing, MSRV verification, and code coverage
- Fix unused import lint on i686 from archmage `#[autoversion]` dispatch

## Version 0.7.0

- Update all dependencies to latest versions
- criterion 0.5 → 0.8, rand 0.8 → 0.10, png 0.17 → 0.18, which 7 → 8
- crossterm 0.27 → 0.29, indicatif 0.17 → 0.18, statrs 0.17 → 0.18
- safe_unaligned_simd 0.2.3 → 0.2.4, thiserror 2.0.9 → 2.0.18

## Version 0.6.0

- Rename crate from `ssimulacra2` to `fast-ssim2`
- Add `imgref` support and simplified input API
- Add precomputed reference API (`Ssimulacra2Reference`) for batch comparisons
- Add runtime SIMD backend selection via `Ssimulacra2Config`
- Add unsafe SIMD backend with x86 intrinsics for best performance
- Reduce memory allocations by 77% and memory usage by 36%
- Add C++ reference parity tests and JPEG quality regression tests
- Update multiversion to 0.8
- Improve API documentation and README

## Version 0.5.1

- Remove nalgebra-macros and update criterion
- Use yuvxyb-math to calculate float constants
- Cleanup way too verbose Clippy settings
- Update thiserror to 2.0

## Version 0.5.0

- Return a concrete `Ssimulacra2Error` error type instead of a freeform `anyhow::Result`
- Precalculate float consts for RecursiveGaussian at build time (performance)
- Update `yuvxyb` dependency to 0.4

## Version 0.4.0

- Update to [version 2.1 of the metric](https://github.com/cloudinary/ssimulacra2/compare/v2.0...v2.1)

## Version 0.3.1

- Minor optimizations
- Bump `nalgebra` dependency to 0.32

## Version 0.3.0

- [Breaking] Reexported structs from yuvxyb have had `From<&T>` impls removed
- Considerably speedups and optimizations

## Version 0.2.0

- [Breaking] Implement updates to the algorithm from upstream (https://github.com/libjxl/libjxl/pull/1848)
- Bump yuvxyb version
- Speed improvements

## Version 0.1.0

- Initial release
