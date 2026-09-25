use fast_ssim2::ToLinearRgb;
use fast_ssim2::official::{self, EncodedSrgb, PermuteOpts, BlurSel};
use imgref::ImgVec;
use std::time::Instant;
fn load(p:&str)->EncodedSrgb{let i=image::open(p).unwrap().into_rgb8();let(w,h)=i.dimensions();let a:ImgVec<[u8;3]>=ImgVec::new(i.pixels().map(|x|[x[0],x[1],x[2]]).collect(),w as _,h as _);a.as_ref().to_encoded_srgb().unwrap()}
fn main(){
    let e1=load(&std::env::args().nth(1).unwrap()); let e2=load(&std::env::args().nth(2).unwrap());
    let opts = PermuteOpts{blur:BlurSel::OfficialSimd,..PermuteOpts::OFFICIAL};
    std::hint::black_box(official::compute_encoded_opts(&e1,&e2,opts).unwrap());
    let t=Instant::now();
    for _ in 0..10 { std::hint::black_box(official::compute_encoded_opts(&e1,&e2,opts).unwrap()); }
    println!("official-simd: {:.2}ms", t.elapsed().as_secs_f64()/10.0*1e3);
}
