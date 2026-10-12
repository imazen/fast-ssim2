# Release audit: 2026-10-12

**Published:** [fast-ssim2 0.9.0](https://crates.io/crates/fast-ssim2/0.9.0),
with the [GitHub release](https://github.com/imazen/fast-ssim2/releases/tag/v0.9.0-release).
The findings below include observations taken before publication.

Issue [8](https://github.com/imazen/fast-ssim2/issues/8) was already fixed by
96ad1ab: the HDR path supplies an `Unstoppable` when no stop token is configured.
The HDR feature compiles and its library tests pass. The issue is closed.

## Published baseline and compatibility

Live registry inspection found 0.8.2 as the latest published crate. GitHub has a
v0.9.0 tag and release, but crates.io has no 0.9.0. Resolve that discrepancy
before choosing the next version; do not reuse a historical tag.

`cargo semver-checks` against 0.8.2 reports 12 major-category and one minor
compatibility checks failing. Run with Rust 1.98.1: semver-checks 0.49 does not
understand Rust 1.99's rustdoc JSON format yet.

The break is substantial: generic input traits and `LinearRgbImage` disappear,
entry points now borrow `PixelSlice`, and configuration gains a lifetime and
becomes non-exhaustive. `CompareContext`, `compare_context`, `compare_with`,
strip entry-point variants and the old HDR nits entry point disappear. The
error enum becomes non-exhaustive and loses its conversion variant.
The existing [migration guide](../CHANGELOG.md#migration-guide) describes the
replacement calls. A breaking 0.x release needs version approval.

## Configuration and reused references

Both published 0.8.2 and current defaults select SIMD. Current config defaults
to whole-image computation with no stop token; there is no scratch-reuse flag.
The audit explicitly uses `Ssimulacra2Config::simd()` for both current reference
construction and comparison. No rayon feature, strips, decoding, or
`target-cpu=native` are used. Reference construction is outside cached timing.

0.8.2 `compare()` creates a fresh `CompareContext` for every comparison;
`compare_with()` reuses caller-owned scratch. Current `compare_with_config()`
selects options but does not restore that scratch-reuse API. These are distinct
baselines and must be reported separately.

## Performance

Measurements use identical deterministic RGB8 buffers in a single process,
interleaved by zenbench on a Ryzen 9 7900X. Each version sees the same pixels.
The algorithms differ, so these compare observable APIs rather than isolate a
single kernel. Full commands, revisions, confidence intervals and resource
records are in the linked outputs.

| Comparison with 0.8.2 | Measured current result |
|---|---|
| One-shot 320×240, three process runs | Slower; paired 95% intervals +6.6–12.5%, +7.6–9.8%, +15.8–17.4% |
| One-shot 1920×1080, first run | Faster; interval −7.5% to −1.5% |
| One-shot 3840×2160, first run | Faster; interval −8.1% to −5.6% |
| Cached 1920×1080, explicit SIMD | 113.04 ms current; 121.57 ms old `compare`; 78.99 ms old `compare_with` |

The last comparison is +39.5–46.4% slower than old scratch reuse. It reproduces
an earlier default-config run (+39.6–46.4%). This is a release-relevant
regression for consumers of the removed reuse API. No new public replacement
was added during this audit.

Outputs: [first run](../benchmarks/version_audit_2026-10-12.txt),
[second run](../benchmarks/version_audit_2026-10-12_repeat2.txt),
[third small-image run](../benchmarks/version_audit_2026-10-12_repeat3.txt),
[cached defaults](../benchmarks/version_audit_2026-10-12_cached.txt),
[cached explicit SIMD](../benchmarks/version_audit_2026-10-12_cached_explicit.txt).
The second run's original command-line group filter was ineffective; it ran
all sizes. The harness now filters through `AUDIT_GROUP`.

Resource records: first run `peak-RSS 1.22GiB | min-avail 19781MiB | peak-load 5.51`;
second `peak-RSS 1.22GiB | min-avail 22488MiB | peak-load 2.21`;
third `peak-RSS 1.21GiB | min-avail 23023MiB | peak-load 0.81`;
explicit cached `peak-RSS 1.61GiB | min-avail 22134MiB | peak-load 1.93`.
These are wrapper observations, not isolated library allocation measurements.

No real-photo performance comparison, rayon comparison, ARM comparison, or
portable-software-FMA performance comparison was measured in this audit.

## Scoring fidelity

[The RGB8 parity run](../benchmarks/version_audit_2026-10-12_rgb8_parity.txt)
checks source and distorted SHA-256 hashes against the captured C++ table and
excludes 40 identity pairs from the accuracy summary.

| Version | Mean absolute error, 26 nonidentity pairs | Maximum |
|---|---:|---:|
| 0.7.1 | 0.339807752488 | 1.996953279363 |
| 0.8.2 | 0.326533066838 | 1.996953279363 |
| Current | 0.000000046766 | 0.000000609235 |

This verifies much closer RGB8 agreement with the stored C++ scores, whose
capture date is 2026-01-04. It is not a fresh photographic C++ evaluation or
proof of bit identity on arbitrary images. The existing `reference_parity`
test uses normalized encoded float pixels: its uniform-shift maximum error is
0.485317. Do not extend the RGB8 LUT fidelity claim to every input descriptor.

## Fixes and release gates

Strict CI initially failed on deprecated magetypes constructors. 4d66e3a
updates constructors and raises the minimum to 0.9.30. dfd9bc1 uses
`mul_add_portable` to preserve fused arithmetic on WASM and scalar SIMD
emulation; unchanged exact assertions now pass on WASM SIMD128 and cross i686.
Native strict library/CLI clippy and library tests also pass.

Before publishing:

1. Review README and migration guide, approve the version and give release authorization.
2. Resolve the GitHub/registry version discrepancy and decide how to handle the
   small-image and scratch-reuse regressions above.
3. Keep `cargo test --all-targets`, `cargo test --doc`, feature checks, API
   snapshots and semver review green on the chosen release commit.
4. Wait for every CI platform, including Windows ARM, macOS Intel and cross
   i686. This audit does not authorize tagging while CI is pending.
5. Push release commit, wait for green CI, tag, push tag, create GitHub release,
   then publish to crates.io. No crate was published by this audit.

The optional CLI video feature remains a separate release check: it requires
system decoder libraries and is not exercised by default CLI CI. Historical
build-break notes are not evidence that the current feature builds.

## Final preparation checks

At 52d8ed30, `cargo test --all-targets` and `cargo test --doc` pass. The full
[gate output](../benchmarks/release_gate_2026-10-12.txt) is retained; its
development-profile benchmark samples are not used as regression evidence.
Resource record: `peak-RSS 1.29GiB | min-avail 19213MiB | peak-load 5.46`.
The public API snapshots pass `ZEN_API_DOC=check`. Library publication dry-run
packages and verifies successfully; no upload occurred.

The approved release is 0.9.0, relative to the published 0.8.2 baseline. The
existing August `v0.9.0` GitHub tag is preserved; `v0.9.0-release` identifies
the registry release source. The user approved README review, documentation
corrections and publication, and explicitly waived the macOS Intel CI gate.

## Publication verification

Release source: `76131b44350aac1a1a68e2c9493945f9cd9ea9eb`.
All 12 jobs in [release-commit CI](https://github.com/imazen/fast-ssim2/actions/runs/38230095777)
passed, including macOS Intel, so the authorized waiver was not needed.

`cargo publish -p fast-ssim2` completed successfully. A fresh
`cargo read fast-ssim2==0.9.0 --path-only` download records the same commit in
`.cargo_vcs_info.json`; its SIMD source matches the released tree byte for byte.
`cargo search fast-ssim2 --limit 1` reports 0.9.0. The GitHub release existed
before the registry upload. The August `v0.9.0` tag remains unchanged.
