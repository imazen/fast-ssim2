use imgref::ImgVec;
fn main(){
    let t0=std::time::Instant::now();
    let d1=image::open(std::env::args().nth(1).unwrap()).unwrap();
    let d2=image::open(std::env::args().nth(2).unwrap()).unwrap();
    let alpha = d1.color().has_alpha();
    let (w,h);
    eprintln!("decode: {:.3}s", t0.elapsed().as_secs_f64());
    let s;
    let t1=std::time::Instant::now();
    if alpha {
        let i1=d1.into_rgba8(); let i2=d2.into_rgba8();
        (w,h)=i1.dimensions();
        let a1:ImgVec<[u8;4]>=ImgVec::new(i1.pixels().map(|x|[x[0],x[1],x[2],x[3]]).collect(),w as _,h as _);
        let a2:ImgVec<[u8;4]>=ImgVec::new(i2.pixels().map(|x|[x[0],x[1],x[2],x[3]]).collect(),w as _,h as _);
        let sh: u32 = std::env::args().nth(3).unwrap_or("64".into()).parse().unwrap();
        let par = std::env::var("PAR").is_ok();
        let cfg = fast_ssim2::Ssimulacra2StripConfig::default().with_parallel_strips(par);
        s = fast_ssim2::compute_ssimulacra2_strip_with_config(a1.as_ref(),a2.as_ref(),sh,cfg).unwrap();
        println!("strip{sh}(par={par}): {s}");
    eprintln!("compute: {:.3}s", t1.elapsed().as_secs_f64());
        eprintln!("compute: {:.3}s", t1.elapsed().as_secs_f64());
        return;
    }
    let i1=d1.into_rgb8(); let i2=d2.into_rgb8();
    (w,h)=i1.dimensions();
    let a1:ImgVec<[u8;3]>=ImgVec::new(i1.pixels().map(|x|[x[0],x[1],x[2]]).collect(),w as _,h as _);
    let a2:ImgVec<[u8;3]>=ImgVec::new(i2.pixels().map(|x|[x[0],x[1],x[2]]).collect(),w as _,h as _);
    let sh: u32 = std::env::args().nth(3).unwrap_or("64".into()).parse().unwrap();
    let par = std::env::var("PAR").is_ok();
    let cfg = fast_ssim2::Ssimulacra2StripConfig::default().with_parallel_strips(par);
    let s = fast_ssim2::compute_ssimulacra2_strip_with_config(a1.as_ref(),a2.as_ref(),sh,cfg).unwrap();
    println!("strip{sh}(par={par}): {s}");
    eprintln!("compute: {:.3}s", t1.elapsed().as_secs_f64());
}
