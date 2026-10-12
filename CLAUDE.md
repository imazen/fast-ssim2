# fast-ssim2 Project Notes

## Current development state (2026-09-27)

The manifest still reads 0.9.0, which is already a released version with the
older trait-based API. This is an unreleased breaking migration; a future
release must pass semver review and receive version approval. This session
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

## Dependency compatibility (2026-10-12)

The explicit-token SIMD constructors and slice partitioning use magetypes
0.9.30 `_t` spellings. The previous names are deprecated in 0.9.30 and
failed CI under `-D warnings`; their replacements delegate to the same
operations. Keep the manifest minimum at the version providing these methods.

## Known Bugs: portable fused arithmetic (2026-10-12)

Fixed: the pipeline's vector multiply-add used non-fused arithmetic on WASM
and scalar SIMD emulation while the scalar backend used fused `f32::mul_add`.
The unchanged exact cached-reference assertion failed on WASM and i686 CI.
Using magetypes 0.9.30 `mul_add_portable` preserves native FMA and supplies
fused software arithmetic on these targets. Native, WASM SIMD128 and cross
i686 library tests pass with the original assertions.

## Release and performance audit (2026-10-12)

Current release/API/configuration findings and committed measurements live in
`docs/release-audit-2026-10-12.md`. RGB8 agrees much more closely with the stored
C++ fixtures; normalized f32 input does not have the same residual. Small-image
and removed scratch-reuse performance regressions are measured there.
