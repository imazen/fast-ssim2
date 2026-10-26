# Native HDR frontend parity

The test oracle uses [Cloudinary SSIMULACRA2 2.1](https://github.com/cloudinary/ssimulacra2/tree/81feacf6a5b0bbd7ec11b4a881a4615033ab0b10),
including its bundled libjxl CMS. It converts to linear sRGB, then applies the
standard cube-root XYB encoding. PU21 is a separate scoring variant.

`tools/native-hdr-oracle.cc` calls the upstream CMS and XYB implementation
on exact f32 input bits. Its `score` mode calls upstream SSIMULACRA2 directly.
No decoder, file format or 8-bit conversion intervenes. This C++ code is test
instrumentation, outside the library dependency graph.

The frontend fixture in `fast-ssim2/src/source/native_hdr_cpp.tsv` stores
324 pixel probes: linear/PQ/HLG, BT.709/BT.2020/Display P3, default and custom
intensity targets, opaque input and alpha compositing on both backgrounds.
Each pixel line contains ten consecutive eight-digit hex f32 representations:
input RGBA, output linear RGB, output positive XYB. Case headers identify
transfer, upstream primaries number (11 is P3 with D65), intensity target,
background and presence of alpha. The file records the oracle executable hash.

Verified against the oracle on 2026-10-26:

- The reference uses PQ's rational polynomial, including fused Horner steps
  and sign preservation. An exact ST 2084 `pow` implementation is numerically
  different.
- The opsin matrix coefficients are multiplied by `intensity_target / 255`
  before mixing RGB. Scaling pixels first has different rounding.
- Alpha is composited in the input transfer domain before conversion.
- Wide-gamut conversion includes ICC colorant quantization. The captured
  f64 matrices from the generated profiles matched the CMS on 4096 finite,
  extended-range RGB triples per gamut (BT.2020 and Display P3).

## Reproduce

Build upstream's `jxl` static library with lcms2 and Highway available. From
this repository, under the resource wrapper used for other builds:

```sh
just build-native-hdr-oracle /path/to/ssimulacra2/src /path/to/build /path/to/oracle
just capture-native-hdr /path/to/oracle /path/to/new-fixture.tsv
```

The initial capture used lcms2 2.12 and a `libjxl.a` with SHA256
`5bd1b0cc5bf4f443d0b8fd632987e21ebaf124b45fd5c0ee2a2e49f4f5c87a57`.
The test binary uses `-ffp-contract=off`; upstream's explicit Highway FMA
operations remain fused. Generated fixtures are capped at 30 KB by the capture
script. Existing changed captures are preserved as `.bak`; keep comparisons
outside the repository until reviewed.
