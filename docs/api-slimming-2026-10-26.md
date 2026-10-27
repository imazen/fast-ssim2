# Public API slimming review — 2026-10-26

The default API already concentrates on scoring, cached references, spatial
maps and configuration. Most kernel functions live behind `unstable-internals`
and are hidden from ordinary rustdoc. The strongest additional cleanup
candidates are the exported SIMD macro and a never-emitted error variant.
Constructor aliases are a lower-priority style choice.

## Implemented: PU21 deprecation

`hdr-pu` and its two public entry points are deprecated, with the existing
feature, signatures and scoring behavior retained. Both wrappers call one
private implementation, so internal calls do not trigger their own warnings.
Native HDR uses the standard comparison functions without an HDR feature.
Switching a PU21 caller to native HDR changes the metric and its luminance
contract; existing score baselines cannot be silently relabeled.

A build script emits a Cargo warning when the feature is enabled. Cargo
normally shows these only for local/path packages; registry/git dependency
build warnings require verbose output or a failing build. Function-level
`#[deprecated]` attributes warn in the caller even when Cargo suppresses the
dependency's build warning. See the
[Cargo build-script warning reference](https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargowarningmessage).
Stable Cargo has no supported feature-deprecation metadata at the crate's
MSRV. The warning does not require nightly or an MSRV increase.

`just check-deprecations` checks actual downstream compilation: default build
without the warning, feature-only enablement with the warning, both function
warnings, both errors under `-Ddeprecated`, and successful explicit legacy
opt-in. CI runs this check. Existing PU21 regression tests explicitly allow
deprecation only where they exercise retained compatibility; score assertions
and the workspace's warning-as-error lint gates are unchanged.

The [zenmetrics CPU dispatcher](https://github.com/imazen/zenmetrics/blob/d4948aa7f66dc83a1ffabf4ba049692275d6c474/crates/zenmetrics-api/src/cpu_dispatch.rs#L589)
is a verified caller, and its
[manifest](https://github.com/imazen/zenmetrics/blob/d4948aa7f66dc83a1ffabf4ba049692275d6c474/crates/zenmetrics-api/Cargo.toml#L183)
enables `hdr-pu`. Actual removal needs a consumer migration and an explicitly
approved breaking release. No removal version is assigned here.

## Further candidates — proposed, not deprecated

| Item | Priority | Finding and migration |
|---|---|---|
| `cube_root_and_add_x8!` | High | `#[macro_export]` leaks an internal SIMD helper at the crate root even with default features. Its expansion requires caller-scope `GenericF32x8` / `GenericI32x8` aliases. Internal calls can use a module-local macro. Audit external kernel users before removing the export; there is no promised drop-in public replacement for that low-level operation. |
| `Ssimulacra2Error::GaussianBlurError` | High | The variant is declared but no library path constructs or returns it. Deprecate it before removal. Match-only callers can remove its dedicated arm while retaining a catch-all for the non-exhaustive error enum. Applications constructing the variant need their own error handling. |
| `Ssimulacra2Config::simd()` | Optional | It currently produces the same configuration as `Ssimulacra2Config::default()`. Deprecating it would establish one spelling for defaults. |
| `Ssimulacra2Config::new(implementation)` | Optional | Equivalent to `Ssimulacra2Config::default().with_impl(implementation)`. A builder-only convention could remove this duplicate constructor. |

Source checks: the macro and its internal uses are in
`fast-ssim2/src/pipeline/simd.rs`; the error declaration and constructors are
in `fast-ssim2/src/lib.rs`. Searches across the library found no construction
of `GaussianBlurError`. An indexed GitHub organization search returned no
external matches for the macro/error names; that is not proof of absence.
The constructor search found uses in the public `enough` and `how-far`
development tools, so those conveniences are not assumed unused.

## Keep

- Plain and `_with_config` scoring functions: useful default and explicit
  configuration paths, with the same pattern for cached comparisons and maps.
- Cached-reference construction/comparison and spatial-map result types:
  separate capabilities that cannot be replaced by dropping an alias.
- `PixelSlice` / `PixelBuffer` / descriptor re-exports: keep the ordinary
  caller path usable without importing a separate crate for every type.
- Backend selection, cancellation, strip settings and validation constants:
  these express behavior or bounds, not redundant names.
- Public config fields for now: builders cover writes, but removing field
  access would also require a reviewed read-access replacement.

The optional constructor cleanup saves few names and creates migration work.
I would handle the SIMD macro and unused error variant first, and retain
`scalar()` / `strips()` unless a consistent builder-only API is chosen.
No additional candidate is changed by this review.
