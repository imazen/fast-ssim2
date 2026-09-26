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
        let (p1,p2)=(i1.into_raw(),i2.into_raw());
        let s1=zenpixels::PixelSlice::new(&p1,w,h,(w*4) as usize,zenpixels::PixelDescriptor::RGBA8_SRGB).unwrap();
        let s2=zenpixels::PixelSlice::new(&p2,w,h,(w*4) as usize,zenpixels::PixelDescriptor::RGBA8_SRGB).unwrap();
        let sh: u32 = std::env::args().nth(3).unwrap_or("64".into()).parse().unwrap();
        let par = std::env::var("PAR").is_ok();
        let cfg = fast_ssim2::StripConfig::default().with_parallel_strips(par);
        s = fast_ssim2::compute_ssimulacra2_with_config(&s1, &s2, &{ let mut c = fast_ssim2::Ssimulacra2Config::strips(sh as usize); c.strip.as_mut().unwrap().parallel_strips = cfg.parallel_strips; c }).unwrap();
        println!("strip{sh}(par={par}): {s}");
    eprintln!("compute: {:.3}s", t1.elapsed().as_secs_f64());
        eprintln!("compute: {:.3}s", t1.elapsed().as_secs_f64());
        return;
    }
    let i1=d1.into_rgb8(); let i2=d2.into_rgb8();
    (w,h)=i1.dimensions();
    let (p1,p2)=(i1.into_raw(),i2.into_raw());
    let s1=zenpixels::PixelSlice::new(&p1,w,h,(w*3) as usize,zenpixels::PixelDescriptor::RGB8_SRGB).unwrap();
    let s2=zenpixels::PixelSlice::new(&p2,w,h,(w*3) as usize,zenpixels::PixelDescriptor::RGB8_SRGB).unwrap();
    let sh: u32 = std::env::args().nth(3).unwrap_or("64".into()).parse().unwrap();
    let par = std::env::var("PAR").is_ok();
    let cfg = fast_ssim2::StripConfig::default().with_parallel_strips(par);
    let s = fast_ssim2::compute_ssimulacra2_with_config(&s1, &s2, &{ let mut c = fast_ssim2::Ssimulacra2Config::strips(sh as usize); c.strip.as_mut().unwrap().parallel_strips = cfg.parallel_strips; c }).unwrap();
    println!("strip{sh}(par={par}): {s}");
    eprintln!("compute: {:.3}s", t1.elapsed().as_secs_f64());
}
