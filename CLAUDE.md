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
  The native HDR frontend and PU21 reject strips. Parallel strips without
  `rayon` return an explicit error.
- `Ssimulacra2Reference` owns a full SDR or native HDR reference pyramid. Scalar/SIMD selection
  and cancellation apply to construction and comparison. There is no reusable
  zero-allocation comparison context; PU21 has no cached reference.
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

## Registry release: 0.9.0 (2026-10-12)

Published to crates.io from `76131b44350aac1a1a68e2c9493945f9cd9ea9eb`,
tagged `v0.9.0-release`. GitHub release was created before publication; all
12 CI jobs passed, including Intel macOS. A fresh registry download's VCS
metadata and SIMD source match the release commit. The older August `v0.9.0`
GitHub tag is preserved and was never the registry release source.

## Known Bugs: final score transform portability (2026-10-26)

Native HDR frontend fixture bits match on x86-64 Linux and WASM SIMD128.
Final scores need not match the Linux C++ oracle across libm implementations.
For weighted sum `f64::from_bits(0x407076193bdfd43b)`, the unchanged final
transform returns `0xc06c6fb83762f952` on x86-64 glibc 2.35 and
`0xc06c6fb83762f950` on wasm32-wasip1. The difference is reproduced by calling
only the score transform, without any image conversion or blur. Reproduce
with `just native-hdr-transform 407076193bdfd43b` and
`just native-hdr-transform-wasm 407076193bdfd43b`.

Captured C++ score bits gate x86-64 Linux GNU; all platforms retain exact
scalar/SIMD, cached/one-shot and map/score consistency assertions. See
`docs/native-hdr-parity.md` for the pinned C++ oracle and frontend coverage.

## Known Bugs: native frontend edge cases (2026-10-26)

Fixed: native HLG now returns positive zero for either encoded signed zero,
matching libjxl's early return. The ICC matrix accumulator starts at positive
zero, preserving the reference CMS's result when every product is negative
zero. Independent captures cover all eight RGB sign patterns across transfers,
gamuts and intensity targets; the fixtures pass native, WASM SIMD128 and i686.

Fixed: straight/premultiplied alpha metadata on RGB/Gray layouts without a
stored alpha channel is rejected with `UnsupportedInput` before ingestion.
The shared check covers standard, native HDR and PU21 frontends; these inputs
previously ignored alpha metadata or could read past a pixel's stored channels.

## Known Bugs: HLG grayscale power portability (2026-10-26)

Fixed: macOS ARM's `powf` differed by one bit from the pinned Linux C++ oracle
for HLG grayscale on the 0.1 alpha background at 203 nits. Using f64 power
and casting back also differed for some inputs (HLG code 271/65535 at 1000
nits). A private Arm-derived power polynomial retains the oracle's operation
order for positive finite bases. All 648 frontend fixtures and an expanded
5,505,024-probe RGB/grayscale differential test pass on x86-64 Linux.
The code-271 case also runs as a normal unit regression. See the parity record
for platform verification and source provenance.

Grayscale HLG now evaluates the inverse OETF once before expanding to RGB,
matching the CMS's single-channel processing. The previous path evaluated
identical channel samples three times. The 5,505,024-probe oracle, 648 frontend
fixtures and full native score/cache/map audit pass with unchanged expectations.
