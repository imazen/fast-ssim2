//! iai-callgrind bench: instruction counts for the official pipeline.
//! Deterministic input → instruction counts are load-independent.
use fast_ssim2::official::{self, EncodedSrgb, PermuteOpts, BlurSel};
use iai_callgrind::{
    library_benchmark, library_benchmark_group, main as iai_main,
};
use std::hint::black_box;

fn enc(w: usize, h: usize, seed: u32) -> EncodedSrgb {
    let rgb: Vec<[u8; 3]> = (0..w * h)
        .map(|i| {
            let v = ((i as u32 * 2654435761u32 ^ seed) >> 4) & 0xff;
            [v as u8, ((v >> 3) + 17) as u8, ((v >> 6) + 40) as u8]
        })
        .collect();
    let data: Vec<u8> = rgb.iter().flat_map(|p| p.iter().copied()).collect();
    EncodedSrgb {
        width: w,
        height: h,
        data: official::EncodedData::U8(data),
        alpha: None,
    }
}

fn setup() -> (EncodedSrgb, EncodedSrgb) {
    (enc(512, 512, 0x1234), enc(512, 512, 0x9abc))
}

#[library_benchmark]
#[bench::encoded(setup())]
fn official_simd_512((a, b): (EncodedSrgb, EncodedSrgb)) -> f64 {
    let opts = PermuteOpts {
        blur: BlurSel::OfficialSimd,
        ..PermuteOpts::OFFICIAL
    };
    black_box(official::compute_encoded_opts(&a, &b, opts).unwrap())
}

fn mk_planes() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = 512 * 512;
    let mk = |s: u32| {
        (0..n)
            .map(|i| (((i as u32 * 2654435761 ^ s) & 0xffff) as f32) / 65536.0)
            .collect::<Vec<f32>>()
    };
    (mk(1), mk(2), mk(3), mk(4))
}

#[library_benchmark]
#[bench::edge(mk_planes())]
fn edge_scalar((a,b,c,d): (Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>)) -> [f64;12] {
    let (i1,m1,i2,m2) = ([a.clone(),b.clone(),c.clone()],[a.clone(),b.clone(),c.clone()],[a.clone(),b.clone(),c.clone()],[a.clone(),b.clone(),c.clone()]);
    let _ = d;
    black_box(official::maps::edge_diff_map(&i1,&m1,&i2,&m2,512,512))
}

#[library_benchmark]
#[bench::edge(mk_planes())]
fn edge_fast((a,b,c,d): (Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>)) -> [f64;4] {
    let mut e = [0f64; 4];
    official::simd::edge_sums_fast(0, 512, 512, &a, &b, &c, &d, &mut e);
    black_box(e)
}

#[library_benchmark]
#[bench::edge(mk_planes())]
fn edge_simd((a,b,c,d): (Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>)) -> [f64;12] {
    let (i1,m1,i2,m2) = ([a.clone(),b.clone(),c.clone()],[a.clone(),b.clone(),c.clone()],[a.clone(),b.clone(),c.clone()],[a,b,c]);
    black_box(official::simd::edge_diff_map_simd(&i1,&m1,&i2,&m2,512,512))
}

#[library_benchmark]
#[bench::encoded(setup())]
fn official_ref_new((a, _b): (EncodedSrgb, EncodedSrgb)) -> usize {
    black_box(official::precompute::OfficialReference::new(&a).unwrap().width())
}

#[library_benchmark]
#[bench::encoded(setup())]
fn official_ref_compare((a, b): (EncodedSrgb, EncodedSrgb)) -> f64 {
    let r = official::precompute::OfficialReference::new(&a).unwrap();
    black_box(r.compare(&b).unwrap())
}

#[library_benchmark]
#[bench::encoded(setup())]
fn official_scalar_512((a, b): (EncodedSrgb, EncodedSrgb)) -> f64 {
    black_box(official::compute_encoded_opts(&a, &b, PermuteOpts::OFFICIAL).unwrap())
}

library_benchmark_group!(
    name = grp_official;
    benchmarks = official_simd_512, official_scalar_512, official_ref_new, official_ref_compare
);
library_benchmark_group!(
    name = grp_edge;
    benchmarks = edge_scalar, edge_simd, edge_fast
);
iai_main!(library_benchmark_groups = grp_official, grp_edge);
