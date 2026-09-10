# x86_64 and aarch64 compute the same score, bit for bit

**Date:** 2026-09-09 · **Commit:** `5c66642` · **Tool:**
`examples/arch_scores.rs` · **Hosts:** Apple M4 Pro (aarch64, `neon` tier,
rustc 1.98.0) and Ryzen 9 7900X (Zen 4, `v3` = AVX2+FMA tier, rustc 1.98.1).

Every parity number in `cpp_parity_2026-08-31.md` and
`version_divergence_2026-09-09.md` was measured on aarch64, and both records
list x86_64 under "not measured". This closes half of that gap: not
fast-ssim2-vs-C++ on x86 (which needs a libjxl build on the box — see below),
but **fast-ssim2-on-x86 vs fast-ssim2-on-aarch64**, which is what decides
whether the aarch64 conclusions transfer at all.

## Result

28 pairs — four sizes (64×64, 177×129, 256×256, 640×480) × seven distortions
(self, ±2 and ±12 noise, box blur r1 and r3, quantise to 8 and 32 levels),
scores printed at full `f64` precision:

```
diff <(grep -E '^[0-9]+x' scores-aarch64.tsv) <(grep -E '^[0-9]+x' scores-x86_64.tsv)
→ no differences
```

**All 28 agree to the last bit.** So the NEON and AVX2 arms of every
`#[magetypes(v3, neon, wasm128, scalar)]` kernel — XYB, blur, SSIM map, edge-diff
map, multiply — produce identical results on identical input, and the
aarch64-measured C++ agreement (mean |Δ| 0.0209, bias +0.0067) is also the x86
agreement with an x86 build of the same C++ code.

That is the expected outcome rather than a lucky one: `f32::mul_add` and the
`magetypes` `mul_add` both lower to a real fused multiply-add on NEON and on
AVX2+FMA, and 0.9.0 deliberately unfused the one expression (the opsin matmul)
whose association differed between the vector arms and the scalar polyfill. It
had simply never been checked across two machines.

## Content, and why it is not flat

The sources are synthesised in-process — a diagonal gradient, concentric rings,
a hard edge and a textured quadrant — so both hosts are guaranteed to score
identical input without a corpus or LFS. They are deliberately *structured*:
SSIMULACRA2 divides a near-zero blur residual by `kC2 = 9e-4`, so flat fields
amplify pure rounding by ~1e6 and would manufacture disagreement that says
nothing about the architecture. That is the same degeneracy that makes the
compiled-in reference table's `uniform_shift` family unusable as a gate
(`cpp_parity_2026-08-31.md` §2).

The distortions avoid JPEG deliberately: a JPEG round-trip would drag in an
encoder whose own output may differ between hosts, and then the test would be
measuring `image`'s encoder rather than this crate.

## Reproduce

```sh
cargo run --release --example arch_scores > scores-$(uname -m).tsv   # on each host
diff scores-aarch64.tsv scores-x86_64.tsv
```

## Still not measured

- **fast-ssim2 vs the C++ binary on x86.** `r7900x` has no `ssimulacra2` binary
  and no image corpus (its `codec-corpus` checkout is 3 MB — LFS content not
  pulled), so this needs a libjxl or jpegli build plus a corpus pull on that
  box. Worth doing, because the C++ reference is *itself* not arch-consistent:
  its horizontal Gaussian differs between `HWY_SCALAR` and its vector targets
  (though not between vector widths — it is capped at four lanes, see
  `cbrt_perf_2026-09-09.md`).
- **i686 and wasm128.** These take the `magetypes` scalar polyfill, whose
  `mul_add` is `a * b + c` — two roundings — so the blur lands in a different
  equivalence class and the scores are expected to differ by up to ~0.5. This
  run says nothing about them, and the 28 scores here should **not** be pinned
  as a cross-platform CI fixture until that has been measured on those targets;
  it would be a fixture that fails by design on two of the CI platforms.
