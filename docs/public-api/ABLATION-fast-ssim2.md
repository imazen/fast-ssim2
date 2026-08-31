# fast-ssim2 Public API Ablation Report

**Date:** 2026-06-11
**Snapshot commit:** 73873c51c5a2 (main) — "fix(package): exclude the api-snapshot test from the published crate"
**Released version:** 0.8.2 (published 2026-06-10)
**Snapshot file:** `docs/public-api/fast-ssim2.txt`
**Default surface:** 210 items · **All-features (excl _*):** 226 items (16 added by `hdr-pu` + `imgref` features)
**Grep template:** `find /home/lilith/work -name "*.rs" -not -path "*/fast-ssim2/*" -not -path "*/target/*" -not -path "*/.jj/*" -exec grep -l "<pattern>" {} \;`

---

## Summary

| Tier | Count | % of default surface |
|------|-------|---------------------|
| Flagged class A (`#[doc(hidden)]`/`#[deprecated]`) | **1** | ~0.5% |
| Flagged class B (join existing 0.9.0 queue) | **1** | ~0.5% |
| Already queued for 0.9.0 (listed per brief) | 2 fns | — |
| API-style note (not a mistake) | 1 | — |
| KEEP (confirmed consumers or deliberate design) | all others | — |

**Total flagged:** 2 items (1 A, 1 B). 1.0% of the default surface.

---

## Already-queued 0.9.0 removals (per CHANGELOG — list as requested, no new proposal)

| Item | Status |
|------|--------|
| `compute_frame_ssimulacra2<T,U>(T, U)` | **SHIPPED 0.9.0** — removed |
| `compute_frame_ssimulacra2_with_config<T,U>(T, U, Ssimulacra2Config)` | **SHIPPED 0.9.0** — removed |

No external org consumers found in the scan. Migration: `compute_ssimulacra2` / `compute_ssimulacra2_with_config`.

---

## Flagged items

### A-1: `Ssimulacra2Error::GaussianBlurError` — class A (`#[doc(hidden)]` or `#[deprecated]`)

**Evidence:**
- Defined at `/home/lilith/work/zen/fast-ssim2/fast-ssim2/src/lib.rs:289`
- **Never constructed anywhere in the codebase.** grep for `GaussianBlurError` across the entire `~/work/` tree (excluding the repo): **0 hits**. grep within the repo itself (excluding docs): **1 hit — the definition only**.
- No code path in the library returns `Err(Ssimulacra2Error::GaussianBlurError)`. The Gaussian blur implementation does not have a fallible return path; the enum variant was likely a placeholder from an earlier design where blur could fail.

**Proposal:** **A** — add `#[doc(hidden)]` now; add to the 0.9.0 removal queue alongside the deprecated fns (the enum is `Copy + Eq` but not `#[non_exhaustive]`, so removal is breaking; batch with the existing 0.9.0 breaking set).

**CHANGELOG queue entry:**
```
- Remove `Ssimulacra2Error::GaussianBlurError` — never constructed; batched with 0.9.0 breaking set
```

---

### B-1: `LinearRgbConversionFailed` becomes unreachable post-0.9.0

**Evidence:**
- Constructed at `src/lib.rs:445,449` — **both sites are inside `compute_frame_ssimulacra2_impl`**, the private helper called only by the two deprecated fns.
- After 0.9.0 removes `compute_frame_ssimulacra2` / `compute_frame_ssimulacra2_with_config`, `LinearRgbConversionFailed` will have **zero construction sites**.
- No external consumer matches this variant in error-handling code (0 org hits).
- The non-deprecated `compute_ssimulacra2_*` / `Ssimulacra2Reference` paths use `TryFrom<T>` directly and propagate errors via `NonMatchingImageDimensions`, `InvalidImageSize`, `ImageTooLarge` — not this variant.

**Proposal:** **B** — add `LinearRgbConversionFailed` to the 0.9.0 queue alongside the deprecated fns. It becomes dead on the same removal boundary.

**CHANGELOG queue entry:**
```
- Remove `Ssimulacra2Error::LinearRgbConversionFailed` — only constructed inside the deprecated `compute_frame_ssimulacra2*` path; unreachable after that removal
```

> **SUPERSEDED 2026-08-31 (0.9.0) — do NOT remove this variant.** The premise
> ("zero construction sites after the removal") no longer holds. 0.9.0 made
> `ToLinearRgb` fallible and added `ToLinearRgb` impls for `yuvxyb::Yuv<T>` /
> `&Yuv<T>` / `Xyb`, which is how YUV input reaches the non-deprecated API at
> all. `LinearRgbConversionFailed` is now constructed in `src/input.rs` by the
> `Yuv` and `Rgb` impls and is covered by tests in `input.rs::yuv_tests`. It is
> load-bearing.

> **Note on A-1 (`GaussianBlurError`):** still never constructed as of 0.9.0,
> and still **not** removed — it was not in the approved 0.9.0 scope. It stays
> queued.

---

## Items audited and confirmed KEEP

| Item | Reason |
|------|--------|
| `fast_ssim2::Blur` struct + all methods | No direct external consumer found (`fast_ssim2::Blur` — 0 org hits), BUT: (a) `Blur` is the internal engine exposed for callers who want to manage blur state; `CompareContext` is the higher-level allocation-reuse surface. `Blur` gives lower-level control to callers doing unusual things (custom pyramids, GPU co-use). (b) The surface cost is small. Conservative ruling: KEEP. |
| `Ssimulacra2Config::impl_type` pub field | No external callers reading the field directly (0 org hits). However the field enables callers to inspect which backend was selected — a legitimate diagnostic use. `Ssimulacra2Config::new(SimdImpl)` / `simd()` / `scalar()` constructors make the struct usable without direct field access, but the field is neither surprising nor harmful. KEEP. |
| `compute_ssimulacra2_strip<S,D>(S,D,u32)` | Active consumer: zenmetrics `cpu_adapter.rs:554` and heaptrack driver binaries. KEEP. |
| `srgb_to_linear(f32) -> f32` | Different signature from `srgb_u8_to_linear(u8)`. The f32→f32 version is a per-sample linear-domain transfer function inverse; `srgb_u8_to_linear` is the LUT-backed integer path. No external consumers found for the f32 variant in this scan, but it is a legitimate complementary function for HDR / non-integer pipelines. KEEP (no consumer today ≠ accidental pub for a utility function this clear). |
| `srgb_u8_to_linear(u8) -> f32` | Active consumers: zenjpeg, zenjxl examples and tests. KEEP. |
| `srgb_u16_to_linear(u16) -> f32` | Companion to u8 version; no org hits but deliberate, useful, and part of a clear family. KEEP. |
| `compute_ssimulacra2_pu_nits` | Fresh deliberate HDR API (0.8.2, hdr-pu feature). KEEP unconditionally. |
| `LinearRgbConversionFailed` (current state) | Still reachable until 0.9.0 removal of deprecated fns. KEEP as-is for 0.8.x; queue for B. |
| `Ssimulacra2Error::InvalidImageSize` | Constructed in non-deprecated path. KEEP. |
| All `CompareContext` / `Ssimulacra2Reference` / `Ssimulacra2StripConfig` surface | Active consumers in zenmetrics CPU path. KEEP. |
| All `ToLinearRgb` impls (`imgref::ImgRef<…>`, `yuvxyb::*`) | Required for the generic API; `imgref` feature gates the ImgRef impls cleanly. KEEP. |

---

## API-style note (not a mistake)

`compute_ssimulacra2_strip<S,D>(S, D, **u32**)` — strip height parameter is `u32`. The matching `compare_strip` on `Ssimulacra2Reference` also uses `u32`. This mirrors the butteraugli strip API (also `u32`) and was likely chosen to discourage accidental usize-max inputs. Not wrong, but a future cleanup could consider `NonZeroU32` or `usize` for ergonomics. **Not flagging** — mixed int-width APIs are common for height/stride parameters and it is consistent across the fast-ssim2 + butteraugli families.

---

## Grep commands and counts

```bash
# GaussianBlurError — construction sites
grep -rn "GaussianBlurError" /home/lilith/work/zen/fast-ssim2/ --include="*.rs" | grep -v "target/" | grep -v "public-api"
# → 1 hit (definition only, src/lib.rs:289)

find /home/lilith/work -name "*.rs" -not -path "*/fast-ssim2/*" -not -path "*/target/*" -not -path "*/.jj/*" \
  -exec grep -l "GaussianBlurError" {} \;
# → 0 hits

# LinearRgbConversionFailed — construction sites  
grep -n "LinearRgbConversionFailed" /home/lilith/work/zen/fast-ssim2/fast-ssim2/src/lib.rs
# → 3 hits: definition (252), Err return (445), Err return (449) — both in compute_frame_ssimulacra2_impl only

# fast_ssim2::Blur external use
find /home/lilith/work -name "*.rs" -not -path "*/fast-ssim2/*" -not -path "*/target/*" -not -path "*/.jj/*" \
  -exec grep -l "fast_ssim2::Blur" {} \;
# → 0 hits (zenmetrics ssim2-gpu uses ssimulacra2::Blur from the old crate v0.5, not fast-ssim2::Blur)

# compute_ssimulacra2_strip external use
find /home/lilith/work -name "*.rs" -not -path "*/fast-ssim2/*" -not -path "*/target/*" -not -path "*/.jj/*" \
  -exec grep -l "compute_ssimulacra2_strip" {} \;
# → 6 hits in zenmetrics (orchestrator + heaptrack drivers, incl. f64-metal-fix worktree)

# srgb_u8_to_linear external use
find /home/lilith/work -name "*.rs" -not -path "*/fast-ssim2/*" -not -path "*/target/*" -not -path "*/.jj/*" \
  -exec grep -l "fast_ssim2::srgb" {} \;
# → 5+ hits in zenjpeg, zenjxl
```

---

## Top 3 observations

1. **`GaussianBlurError` is dead code** — never constructed anywhere in the library or external code. Queue for `#[doc(hidden)]` immediately, removal at 0.9.0.

2. **`LinearRgbConversionFailed` becomes dead at the same boundary as the deprecated fns** — it should enter the 0.9.0 queue alongside them so they are removed together in one clean breaking release.

3. **`Blur` struct has no direct external consumers** but is a legitimately low-level API. The conservative ruling is KEEP — zero-consumer utilities with a clear domain purpose don't qualify as mistakes under the brief's standard.

---

*Report generated by conservative-ablation scan. REPORT ONLY — no source changes.*
