# fast-ssim2 0.9.1 release validation — 2026-10-31

Dates use America/Denver; the registry timestamp below is UTC.

- Release/tag source: `b02184f76193973ea28df8fb6a8584e821810d60`.
- [GitHub release](https://github.com/imazen/fast-ssim2/releases/tag/v0.9.1)
  and [crates.io package](https://crates.io/crates/fast-ssim2/0.9.1) are public.
- [CI run](https://github.com/imazen/fast-ssim2/actions/runs/41253595063): all
  11 jobs passed, including Windows x64/ARM64, macOS ARM64, Linux x64/ARM64,
  cross i686 and WASM. Logs confirm `aarch64-apple-darwin` and
  `aarch64-pc-windows-msvc`; i686 runs through `cross`.
- [Binary workflow](https://github.com/imazen/fast-ssim2/actions/runs/41255060933):
  all five platform builds and asset publication passed. The release contains
  Linux x64/ARM64, macOS ARM64 and Windows x64/ARM64 archives. The legacy
  Intel macOS entry was removed before tagging.

## Local gates

Commands ran through `run-heavy --mem 3G --jobs 2` with disk-backed scratch.
The test environment was x86-64 GNU Linux on AMD Ryzen 9 7950X.

| Command | Result |
|---|---|
| `just release-check` (`cargo test --all-targets --release`, then `cargo test --doc --release`) | Passed, including all benchmark executables and seven doc tests |
| `just semver-release` (baseline 0.9.0, inferred compatible 0.x update) | Passed without lint overrides |
| `just check-deprecations` | Feature/caller warnings, deny-deprecated errors and explicit legacy allowance passed |
| `cargo clippy -p fast-ssim2 --example reference_spot -- -D warnings` | Passed |
| `just package-check` (`cargo publish -p fast-ssim2 --dry-run`) | Passed |
| `cargo publish -p fast-ssim2` | Published 0.9.1 |

The all-target/doc run compiled the 0.9.1 metric/CLI source at `c7bfa386`;
subsequent release commits changed README URLs, copied the base LICENSE and
removed the Intel macOS release job. Their Rust source is identical. Full
CI and package verification run against the tagged commit.

The successful all-target/doc wrapper reported:
`done rc=0 1906s | peak-RSS 1.29GiB | min-avail 4255MiB | peak-load 11.94`.
This is a resource observation for the whole validation command, not a library
memory benchmark. A preliminary 0.9.0 all-target run was intentionally
interrupted to prioritize the metadata audit. Its transcript is preserved as
incomplete; it is not a release gate or usable timing comparison.

Package commands warned about yanked versions in the existing development
lock graph (`chacha20 0.10.0`, `core2 0.4.0`, `zenbench 0.1.5`). Package build
verification and publication succeeded. No scoring expectation was relaxed.

## Registry verification

Registry publication timestamp: `2026-11-01T00:09:17.122310Z`.

Package SHA256: `e1f3216821c49c3988daa5f35451152559093e7b55c065ec3240ec60a11cc61b`.

The fresh registry download matches the verified pre-publication archive.
Its VCS metadata points to the release commit. The README, build script,
public API source and native HDR conversion/power sources match the checkout
byte-for-byte. All three license notices are included; the workspace's base
BSD-2-Clause notice was copied into the package before publication.

## Scope and CLI artifact check

Native HDR compatibility targets pinned Cloudinary SSIMULACRA2 2.1 through
explicit pixel descriptors. The [reference audit](../docs/reference-spot-2026-10-26.md)
records both legacy ICC inference and explicit cICP controls. It does not
claim universal Cloudinary/libjxl CLI equality.

The file CLI retains its existing `to_rgb32f` / float-sRGB frontend and does
not infer HDR from ICC/cICP. Its output need not equal the raw-u8 LUT path.
The downloaded Linux release binary and a local build of the tagged checkout
produce identical printed scores on both checked opaque RGB8 PNG pairs:

| Pair | Downloaded and checkout CLI | Raw-u8 library API |
|---|---:|---:|
| rgb8-srgb-17x13 | 97.16300457 | 97.163072278759472 |
| rgb8-srgb-513x257 | 93.47096261 | 93.529528546307915 |

Linux CLI executable SHA256: `3e94e2fa2e0602af8f69f59032ab2e3175dcf03159d892277599905994338387`.

The first smoke expectation incorrectly compared the float CLI to raw-u8
reference scores. The verified check compares the same frontend across the
built artifact and checkout, and preserves the raw-u8 differences above.
The CLI frontend is unchanged from 0.9.0; this check is artifact
reproducibility evidence, not a claim of CLI bit equality with Cloudinary.
The GitHub release notes state these limits.

## Supporting artifacts

Full transcripts, registry verification and CLI comparison data are preserved
under `/mnt/v/output/fast-ssim2/release-0.9.1-2026-10-31/` in the test
environment above. Resolve the machine with `hostname` and `lscpu`; the path
was checked there on this date. `manifest.json` records every hash and size.
The primary completed all-target/doc transcript SHA256 is
`4bedd74f09d781f68d069ab551906446522d5b78594cb9535fafaa87303ff5ce`.
