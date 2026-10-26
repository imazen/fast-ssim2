# Native HDR performance and implementation review

The final audit found no SDR regression exceeding zenbench's 1% practical
threshold on the measured cases. Native HLG grayscale is 2.67–5.12% faster
than the initial HDR implementation after removing repeated transfer decoding.
The small PQ case remains 1.27% slower (95% paired CI: +0.42% to +2.07%);
its interval includes effects below the practical threshold. These are
measurements on one Ryzen 5900XT, not a guarantee for other machines or inputs.

## Revisions and method

- Existing SDR baseline: [25ec2fea](https://github.com/imazen/fast-ssim2/commit/25ec2fea8644a2231b64f28c04189ccaf381acd4), immediately before native HDR work.
- Initial native HDR baseline: [d1b74ed7](https://github.com/imazen/fast-ssim2/commit/d1b74ed7d1798ff1cc4d47294d84b6d33837a0a3), before the portable grayscale power correction.
- Audit harness: [22c4aae4](https://github.com/imazen/fast-ssim2/commit/22c4aae4612e06b5f15b2c34c0cfcf1a574671bc), extending `tools/version-audit`.
- Final library: [28b737ce](https://github.com/imazen/fast-ssim2/commit/28b737cedc4cfa8eeecef86a449e38d6c4611818).

All versions are linked into one executable, built with Rust 1.99.1 / LLVM
23.1.1, release mode, thin LTO and default target settings. No
`target-cpu=native`. Each uses its runtime SIMD configuration with rayon off.
Measurements call library functions directly on the same in-memory buffers;
no codec, process startup or file I/O is timed. Input construction is outside
the timed region; borrowed image wrappers are included in both versions.
Inputs contain deterministic noise and abrupt constant patches.
Cached-reference construction is outside the timed comparison; allocations
performed by each measured metric call remain included.

The host was Ryzen 9 5900XT (32 logical CPUs), with no other heavy job or
held heavy-job lock at launch. The audit held the host's heavy-job lock and
ran under `run-heavy --mem 6G --jobs 4`. Both executable hashes were verified
on the build and measurement hosts before execution.

Each group ran 25 interleaved rounds after warmup. Tables use zenbench's
paired analysis after its outlier filtering, rather than mixing all-round
means with filtered confidence intervals. The TSVs retain sample counts and
95% intervals. The final run completed all 20 groups: 486 of 500 rounds were
gate-clean; 14 were marked noisy; the run was not flagged unreliable.
Its inference mode is exploratory. Cross-process replication, Zen 5, ARM,
4K, strided-input timing and rayon scaling were not measured.

## Final measurements

Positive changes mean slower. Values are filtered paired means in milliseconds.

| Case | Baseline ms | Final ms | Change | 95% paired CI |
|---|---:|---:|---:|---:|
| SDR RGB8 32×32 | 0.081306 | 0.081144 | -0.20% | -0.63% to +0.22% |
| SDR RGBA8 32×32 | 0.323350 | 0.322942 | -0.13% | -0.55% to +0.32% |
| SDR RGB8 320×240 | 4.775111 | 4.791809 | +0.35% | -0.00% to +0.72% |
| SDR RGBA8 320×240 | 21.932649 | 21.653808 | -1.27% | -1.75% to -0.79% |
| SDR RGB8 1920×1080 | 242.972720 | 242.427928 | -0.22% | -0.57% to +0.10% |
| SDR RGBA8 1920×1080 | 814.942853 | 814.414077 | -0.06% | -0.31% to +0.18% |
| Linear f32 1920×1080 | 259.503815 | 259.475982 | -0.01% | -0.30% to +0.28% |
| Cached SDR RGB8 1920×1080 | 154.625146 | 154.882486 | +0.17% | -0.37% to +0.67% |
| Native PQ RGB16 32×32 | 0.187959 | 0.190348 | +1.27% | +0.42% to +2.07% |
| Native PQ RGB16 1920×1080 | 469.941434 | 470.270410 | +0.07% | -0.18% to +0.31% |
| Native HLG RGB16 1920×1080 | 378.315185 | 380.088061 | +0.47% | +0.27% to +0.69% |
| Cached native HLG RGB16 1920×1080 | 216.204051 | 216.998348 | +0.37% | +0.18% to +0.56% |
| Native HLG Gray16 32×32 | 0.114130 | 0.108291 | -5.12% | -5.60% to -4.64% |
| Native HLG Gray16 320×240 | 7.252099 | 7.005215 | -3.40% | -3.79% to -3.01% |
| Native HLG Gray16 1920×1080 | 325.734217 | 317.043835 | -2.67% | -3.00% to -2.36% |

The [complete final TSV](../benchmarks/native_hdr_final_2026-10-26.tsv)
also includes the omitted small linear and mid-size PQ/HLG cases.

The first local Ryzen 7950X/WSL2 run had 429 noisy rounds and is exploratory
only. A quieter 5900XT run before cleanup measured +1.40% for 32×32 RGBA8
and +2.55% / +3.74% for 1080p / 320×240 HLG grayscale. The
[pre-cleanup TSV](../benchmarks/native_hdr_before_cleanup_2026-10-26.tsv)
keeps those observations. The tiny RGBA result did not persist in the final
executable; its cause was not isolated. Do not attribute it to the SSIM
rounding correction without a separate controlled experiment.

## Profiling and cleanup

`perf record -F 499 -g` profiled 32 current-kernel HLG grayscale comparisons
at 1920×1080, before and after cleanup, on the same host. The original
frontend decoded three identical grayscale channels before applying the
single-channel OOTF. It now decodes the one stored channel and expands the
result. No arithmetic operation within the transfer function changed.

The exact C++ gate still passes 5,505,024 RGB/grayscale pixel probes, including
all u16 codes in the selected transfer/gamut/target grid. The 648 committed
frontend probes, native score/cache/map audit, Clippy and API inventory also
pass. See the [parity record](native-hdr-parity.md) for coverage and limits.

Disassembly of the release build showed that LLVM already hoisted invariant
HLG gamma computation outside the pixel loop. Adding another prepared-state
abstraction would not remove that work in this build. The frontend remains a
private descriptor/transfer/gamut conversion stage feeding the existing XYB,
multiscale, cache and spatial-map kernels. Supported public signatures are
unchanged. Arithmetic details needed for reference parity stay isolated in
the transfer helpers and the small licensed grayscale power kernel.

Native transfer conversion is scalar; the shared metric uses the selected
scalar/SIMD kernels. Whole-image buffers, allocation during cached comparisons
and rejection of native HDR strips remain explicit limitations. Per-call
allocation/RSS differences were not measured. The audit's process peak is not
a per-image memory bound.

## Reproduce and artifacts

Build and run on a quiet benchmark host, under the resource wrapper:

```sh
just build-version-audit
AUDIT_ROUNDS=25 just native-hdr-performance
just profile-native-hdr HlgGray /path/on/disk/gray.perf
```

The final executable SHA256 was
`03406a45566acdcad0b1470bc279376903855f533bf1becd2bfe1046fcf1dbda`.
The pre-cleanup executable SHA256 was
`aacba7a7ae77b1c6c331cc92e7a0f229b27b9f3df39684ae2b16e991572fc825`.
The final timing command on the measurement host was
`AUDIT_NATIVE_HDR=1 AUDIT_ROUNDS=25 run-heavy --mem 6G --jobs 4 -- ./audit-single-channel`,
inside its held heavy-job lock. The final resource summary was:

```text
run-heavy: done rc=0 188s | peak-RSS 0.82GiB | min-avail 49065MiB | peak-load 1.83
```

The before/after profiler runs each reported a 0.24GiB process peak; these
were separate runs from the timed suite. Raw evidence, full logs and both
reference executables are retained under
`/mnt/v/output/fast-ssim2/native-hdr-performance/2026-10-26/` on the 7950X/WSL2
build host, with copies of measurement evidence on r5900xt. The committed TSVs
are compact summaries; their `zenbench_object_sha256` is the SHA256 of the
decoded JSON bytes used by `tools/native_hdr_performance_report.py`.
Use `just native-hdr-performance-summary` to regenerate a summary from a raw
JSON object with the recorded source, binary and host metadata.
