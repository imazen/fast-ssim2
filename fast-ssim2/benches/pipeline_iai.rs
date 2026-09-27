//! iai-callgrind bench: instruction counts for the pipeline.
//! Deterministic input → instruction counts are load-independent.
use fast_ssim2::pipeline::{self, EncodedSrgb, Kernel};
use fast_ssim2::{PixelBuffer, Ssimulacra2Config, compute_ssimulacra2_with_config};
use iai_callgrind::{library_benchmark, library_benchmark_group, main as iai_main};
use std::hint::black_box;

fn enc(w: usize, h: usize, seed: u32) -> EncodedSrgb {
    let rgb: Vec<[u8; 3]> = (0..w * h)
        .map(|i| {
            let v = (((i as u32 * 2654435761u32) ^ seed) >> 4) & 0xff;
            [v as u8, ((v >> 3) + 17) as u8, ((v >> 6) + 40) as u8]
        })
        .collect();
    let data: Vec<u8> = rgb.iter().flat_map(|p| p.iter().copied()).collect();
    EncodedSrgb {
        width: w,
        height: h,
        data: pipeline::EncodedData::U8(data),
        alpha: None,
    }
}

fn setup() -> (EncodedSrgb, EncodedSrgb) {
    (enc(512, 512, 0x1234), enc(512, 512, 0x9abc))
}

#[library_benchmark]
#[bench::encoded(setup())]
fn simd_512((a, b): (EncodedSrgb, EncodedSrgb)) -> f64 {
    black_box(pipeline::compute_encoded(&a, &b, Kernel::Simd).unwrap())
}

fn mk_planes() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = 512 * 512;
    let mk = |s: u32| {
        (0..n)
            .map(|i| ((((i as u32 * 2654435761) ^ s) & 0xffff) as f32) / 65536.0)
            .collect::<Vec<f32>>()
    };
    (mk(1), mk(2), mk(3), mk(4))
}

#[library_benchmark]
#[bench::edge(mk_planes())]
fn edge_scalar((a, b, c, d): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)) -> [f64; 12] {
    let (i1, m1, i2, m2) = (
        [a.clone(), b.clone(), c.clone()],
        [a.clone(), b.clone(), c.clone()],
        [a.clone(), b.clone(), c.clone()],
        [a.clone(), b.clone(), c.clone()],
    );
    let _ = d;
    black_box(pipeline::maps::edge_diff_map(&i1, &m1, &i2, &m2, 512, 512))
}

#[library_benchmark]
#[bench::edge(mk_planes())]
fn edge_fast((a, b, c, d): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)) -> [f64; 4] {
    let mut e = [0f64; 4];
    pipeline::simd::edge_sums_fast(0, 512, 512, &a, &b, &c, &d, &mut e);
    black_box(e)
}

#[library_benchmark]
#[bench::edge(mk_planes())]
fn edge_simd((a, b, c, d): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)) -> [f64; 12] {
    let (i1, m1, i2, m2) = (
        [a.clone(), b.clone(), c.clone()],
        [a.clone(), b.clone(), c.clone()],
        [a, b, c],
        [d.clone(), d.clone(), d],
    );
    black_box(pipeline::simd::edge_diff_map_simd(
        &i1, &m1, &i2, &m2, 512, 512,
    ))
}

fn lin512() -> (PixelBuffer, PixelBuffer) {
    let mk = |seed: u32| {
        let px: Vec<[f32; 3]> = (0..512 * 512)
            .map(|i| {
                let v = (((i as u32 * 2654435761u32) ^ seed) >> 4) & 0xffff;
                let f = v as f32 / 65536.0;
                [f, f * 0.9, f * 0.8]
            })
            .collect();
        lin_f32_buf(px, 512, 512)
    };
    (mk(0x1234), mk(0x9abc))
}

#[library_benchmark]
#[bench::lin(lin512())]
fn linear_input_512((a, b): (PixelBuffer, PixelBuffer)) -> f64 {
    black_box(
        compute_ssimulacra2_with_config(&a.as_slice(), &b.as_slice(), &Ssimulacra2Config::simd())
            .unwrap(),
    )
}

#[library_benchmark]
#[bench::encoded(setup())]
fn ref_new((a, _b): (EncodedSrgb, EncodedSrgb)) -> usize {
    black_box(
        pipeline::precompute::ReferenceCache::new(&a)
            .unwrap()
            .width(),
    )
}

#[library_benchmark]
#[bench::encoded(setup())]
fn ref_compare((a, b): (EncodedSrgb, EncodedSrgb)) -> f64 {
    let r = pipeline::precompute::ReferenceCache::new(&a).unwrap();
    black_box(r.compare(&b).unwrap())
}

#[library_benchmark]
#[bench::encoded(setup())]
fn scalar_512((a, b): (EncodedSrgb, EncodedSrgb)) -> f64 {
    black_box(pipeline::compute_encoded(&a, &b, Kernel::Scalar).unwrap())
}

library_benchmark_group!(
    name = grp_pipeline;
    benchmarks = simd_512, scalar_512, ref_new, ref_compare, linear_input_512
);
library_benchmark_group!(
    name = grp_edge;
    benchmarks = edge_scalar, edge_simd, edge_fast
);
library_benchmark_group!(
    name = grp_xyb;
    benchmarks = xyb_ref, xyb_midp, xyb_lowp
);
library_benchmark_group!(
    name = grp_flavor;
    benchmarks = e2e_ref, e2e_midp, e2e_lowp
);
iai_main!(
    library_benchmark_groups = grp_pipeline,
    grp_edge,
    grp_xyb,
    grp_flavor
);

// ── cbrt flavor isolation ──────────────────────────────────────────────
fn xybin() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = 512 * 512;
    let mk = |s: u32| {
        (0..n)
            .map(|i| ((((i as u32 * 2654435761) ^ s) & 0xffff) as f32) / 65536.0)
            .collect::<Vec<f32>>()
    };
    (mk(7), mk(8), mk(9))
}

#[library_benchmark]
#[bench::p(xybin())]
fn xyb_ref((a, b, c): (Vec<f32>, Vec<f32>, Vec<f32>)) -> Vec<f32> {
    let mut planes = [a, b, c];
    pipeline::simd::planes_to_positive_xyb_simd(&mut planes);
    black_box(planes[0].clone())
}

#[library_benchmark]
#[bench::p(xybin())]
fn xyb_midp((a, b, c): (Vec<f32>, Vec<f32>, Vec<f32>)) -> Vec<f32> {
    let mut planes = [a, b, c];
    pipeline::simd::planes_to_positive_xyb_hi_simd(&mut planes);
    black_box(planes[0].clone())
}

#[library_benchmark]
#[bench::p(xybin())]
fn xyb_lowp((a, b, c): (Vec<f32>, Vec<f32>, Vec<f32>)) -> Vec<f32> {
    let mut planes = [a, b, c];
    pipeline::simd::planes_to_positive_xyb_lo_simd(&mut planes);
    black_box(planes[0].clone())
}

// ── cbrt flavor e2e ────────────────────────────────────────────────────
fn planes512() -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let mk = |s: u32| {
        let n = 512 * 512;
        [
            (0..n)
                .map(|i| ((((i as u32 * 2654435761) ^ s) & 0xffff) as f32) / 65536.0)
                .collect::<Vec<f32>>(),
            (0..n)
                .map(|i| ((((i as u32 * 2246822519) ^ s) & 0xffff) as f32) / 65536.0)
                .collect::<Vec<f32>>(),
            (0..n)
                .map(|i| ((((i as u32 * 3266489917) ^ s) & 0xffff) as f32) / 65536.0)
                .collect::<Vec<f32>>(),
        ]
    };
    (mk(0xaaa), mk(0xbbb))
}

use pipeline::{Opts, XybFlavor};

fn lin_f32_buf(data: Vec<[f32; 3]>, w: usize, h: usize) -> PixelBuffer {
    PixelBuffer::from_vec(
        bytemuck::cast_slice::<f32, u8>(data.as_flattened()).to_vec(),
        w as u32,
        h as u32,
        fast_ssim2::PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap()
}

#[library_benchmark]
#[bench::p(planes512())]
fn e2e_ref((a, b): ([Vec<f32>; 3], [Vec<f32>; 3])) -> f64 {
    black_box(
        pipeline::compute_planar_stop(
            a,
            b,
            512,
            512,
            Opts {
                kernel: Kernel::Simd,
                flavor: XybFlavor::CubeRoot,
            },
            &enough::Unstoppable,
        )
        .unwrap(),
    )
}

#[library_benchmark]
#[bench::p(planes512())]
fn e2e_midp((a, b): ([Vec<f32>; 3], [Vec<f32>; 3])) -> f64 {
    black_box(
        pipeline::compute_planar_stop(
            a,
            b,
            512,
            512,
            Opts {
                kernel: Kernel::Simd,
                flavor: XybFlavor::CubeRootHi,
            },
            &enough::Unstoppable,
        )
        .unwrap(),
    )
}

#[library_benchmark]
#[bench::p(planes512())]
fn e2e_lowp((a, b): ([Vec<f32>; 3], [Vec<f32>; 3])) -> f64 {
    black_box(
        pipeline::compute_planar_stop(
            a,
            b,
            512,
            512,
            Opts {
                kernel: Kernel::Simd,
                flavor: XybFlavor::CubeRootLo,
            },
            &enough::Unstoppable,
        )
        .unwrap(),
    )
}
