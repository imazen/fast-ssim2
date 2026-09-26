// Minimal timing loop on encoded inputs — pipeline only, no decode.
use fast_ssim2::pipeline::{self, EncodedData, EncodedSrgb, Kernel};
use std::time::Instant;
fn load(p:&str)->EncodedSrgb{let i=image::open(p).unwrap().into_rgb8();let(w,h)=i.dimensions();EncodedSrgb{width:w as usize,height:h as usize,data:EncodedData::U8(i.into_raw()),alpha:None}}
fn main(){
    let e1=load(&std::env::args().nth(1).unwrap()); let e2=load(&std::env::args().nth(2).unwrap());
    std::hint::black_box(pipeline::compute_encoded(&e1,&e2,Kernel::Simd).unwrap());
    let t=Instant::now();
    for _ in 0..10 { std::hint::black_box(pipeline::compute_encoded(&e1,&e2,Kernel::Simd).unwrap()); }
    println!("simd: {:.2}ms", t.elapsed().as_secs_f64()/10.0*1e3);
}
