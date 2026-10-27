# Native HDR inputs

Use the ordinary `compute_ssimulacra2*`, `Ssimulacra2Reference` and spatial-map
APIs. No feature flag is required. They use SSIMULACRA2's cube-root XYB encoding
and weights, with the pinned reference's HDR color conversion.

The deprecated `hdr-pu` feature selects separate PU21 entry points. Those
scores and luminance units have a different contract; they are not a
substitute for native HDR scoring.
Local/path builds warn when the feature is enabled, and the PU21 functions
warn at call sites. Cargo normally hides dependency build-script warnings
for registry/git packages. Explicitly allowing deprecated calls retains the
PU21 score definition while callers migrate; changing to native HDR requires
separate score baselines.

## Descriptors and samples

Every image supplies a `zenpixels::PixelSlice`. Its descriptor determines
transfer, primaries, channel order, alpha and signal range.

| Transfer | Samples | Supported layouts |
|---|---|---|
| Linear | f32 | RGB, RGBA, Gray, GrayA |
| PQ / HLG | u8 | RGB, RGBA, RGBX, BGRA, BGRX, Gray |
| PQ / HLG | u16 / f32 | RGB, RGBA, Gray, GrayA |

Integer PQ/HLG samples cover the complete encoded range: 0–255 or 0–65535.
Float PQ/HLG samples use the same encoded scale with 1.0 at its upper endpoint.
Signal range must be `Full`; limited-range samples must be expanded first.
Supported RGB primaries are BT.709, BT.2020 and Display P3, with D65 white.
Grayscale has no RGB gamut conversion. RGB and grayscale images can be compared.

Rows may have padding: set the byte stride in `PixelSlice::new`. Native HDR
supports that stride directly. Native HDR requires whole-image processing;
`StripConfig` options return `InvalidConfiguration`, including cached
comparisons with a native HDR reference or distorted image.
Unanchored BT.709 linear f32 uses the existing linear path and retains its
strip support; a luminance anchor or wide RGB gamut selects the native path.

## Luminance anchors

`PixelSlice::with_diffuse_white(DiffuseWhite::new(nits))` supplies the
reference's intensity target. This is per image, including when source and
distorted images use different descriptors.

| Input | Default | Meaning of a custom anchor |
|---|---|---|
| Linear f32 | 255 nits | Luminance represented by linear 1.0 |
| PQ | 10000 nits | Normalization target; PQ still decodes its absolute luminance |
| HLG | 1000 nits | Reference display peak used by the HLG system gamma |

For absolute linear samples already expressed in nits, use an anchor of 1.0.
An anchor must be finite and greater than zero. The native frontend preserves
linear headroom and negative gamut-conversion results before the reference's
XYB processing; it does not tone-map or clip them into SDR.

For example, these linear BT.2020 samples represent fractions of 1000 nits:

```rust
use fast_ssim2::{compute_ssimulacra2, PixelDescriptor, PixelSlice};
use zenpixels::{ColorPrimaries, DiffuseWhite};

let source = [0.25_f32, 0.5, 0.75];
let distorted = [0.26_f32, 0.5, 0.74];
let descriptor = PixelDescriptor::RGBF32_LINEAR
    .with_primaries(ColorPrimaries::Bt2020);
let a = PixelSlice::new(bytemuck::cast_slice(&source), 1, 1, 12, descriptor)?
    .with_diffuse_white(DiffuseWhite::new(1000.0));
let b = PixelSlice::new(bytemuck::cast_slice(&distorted), 1, 1, 12, descriptor)?
    .with_diffuse_white(DiffuseWhite::new(1000.0));
let score = compute_ssimulacra2(&a, &b)?;
assert!(score.is_finite());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Original dimensions must match. Sub-8-pixel images use the library's existing
reflect padding, an extension beyond the C++ reference's minimum dimensions.
The original and padded images remain subject to `MAX_IMAGE_PIXELS`.

## Alpha, caching and maps

Premultiplied inputs are unpremultiplied on ingest. If either image has alpha,
evaluate both 0.1 and 0.9 backgrounds in each image's **input transfer domain**,
then take the lower score. Transfer decoding and gamut conversion follow
compositing. These backgrounds are encoded values for PQ/HLG and relative
values for linear inputs. Integer unpremultiplication retains the existing
integer-format rounding. Alpha metadata requires a stored alpha channel.

`Ssimulacra2Reference` stores the source pyramid and reuses its transfer and
intensity conversion. Its comparisons and maps agree exactly with one-shot
scoring on the same platform and configuration. Alpha references retain both
background pyramids. Maps report the background selected for their score.

Cached scoring still allocates comparison buffers. Native conversion and the
metric use whole-image planes; this is not a bounded-memory or zero-allocation
API. Scalar/SIMD selection and cooperative cancellation apply to construction
and comparison. Row stride support does not imply strip processing support.

## Reference compatibility

The target is [Cloudinary SSIMULACRA2 2.1](https://github.com/cloudinary/ssimulacra2/tree/81feacf6a5b0bbd7ec11b4a881a4615033ab0b10)
and its bundled libjxl CMS with the captured ICC profile matrices. PQ uses
the reference's rational polynomial, HLG keeps its operation order and
intensity scaling modifies the opsin coefficients before RGB mixing.

The [parity record](native-hdr-parity.md) gives the oracle, probe coverage,
reproduction commands and platform scope. Pixel conversion equality does not
guarantee identical final f64 scores across platform math libraries. No claim
is made about every newer libjxl version or arbitrary external ICC profiles.

The [performance and implementation review](native-hdr-performance-2026-10-26.md)
records pinned before/after timings, profiling, the grayscale cleanup and
remaining measurement limits. Run `just check-native-hdr-guide` to compile
and execute this guide's Rust example; CI runs it too.
