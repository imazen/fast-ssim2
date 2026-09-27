# fast-ssim2 Project Notes

## Current development state (2026-09-27)

The working API migration targets the manifest's 0.9.0 surface. This session
has not published or tagged a release. See [README.md](README.md) for executable
examples and [CHANGELOG.md](CHANGELOG.md) for migration details.

- Public comparisons take `&zenpixels::PixelSlice`; descriptors declare layout,
  transfer, primaries, alpha, and signal range. Rows support byte strides.
- `Ssimulacra2Config` combines backend, strip, and cancellation options.
  Reference construction rejects strips; cached comparisons accept them.
  HDR rejects strips. Parallel strips without `rayon` return an explicit error.
- `Ssimulacra2Reference` owns a full SDR reference pyramid. Scalar/SIMD selection
  and cancellation apply to construction and comparison. There is no reusable
  zero-allocation comparison context and no cached HDR reference.
- Original dimensions must match before sub-8px reflect padding. Original and
  padded pixel counts are capped by `MAX_IMAGE_PIXELS`.
- Strip processing bounds intermediate kernel buffers, not total memory:
  whole-image input buffers remain, plus the reference pyramid for cached use.
- `pipeline` is private by default and exposed only by `unstable-internals`
  for development tools. CI's all-feature pass covers those tools and HDR.
- JPEG parity tests use a pinned upstream zenjpeg revision; no sibling checkout
  is required. Do not replace the decoder without checking reference scores.

## Regression coverage

`fast-ssim2/tests/api_contracts.rs` exercises dimension validation, cancelled
construction, unsupported options, alpha on only the distorted side, grayscale
expansion, HDR compositing in nits, and extreme strip settings.
`fast-ssim2/src/source/tests.rs` checks decoding across HDR layouts and padded
rows, including BGR channel order and premultiplied float grayscale.

`just check-library`, `just clippy-lib`, and `just check-doc-cli` record the local
checks. Run heavy commands through `~/work/zen/scripts/run-heavy` with a
host-appropriate memory cap and at most eight build jobs; set `TMPDIR=~/tmp`.
`README.md` is canonical: `just docs-sync` updates the package/rustdoc copies;
CI checks equality. `just api-doc` regenerates API snapshots.

Tests must exercise behavior, not copies of implementation constants. Keep
reference parity, SIMD consistency, strip/full comparisons, and color-layout
regressions. Do not relax assertions to accommodate a changed implementation.

## Known Bugs

The API review's channel offsets, grayscale expansion, alpha backgrounds,
dimension checks, and ignored-configuration bugs have regression coverage and
fixes in this migration. No unresolved instance of those review findings is
being carried as a limitation. Strip/full numerical differences remain an
explicit property of the strip algorithm, not a bit-equality guarantee.
