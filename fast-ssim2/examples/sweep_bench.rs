// Timing+RSS sweep: sizes × modes. Prints CSV lines.
use fast_ssim2::ToLinearRgb;
use fast_ssim2::official::{self, EncodedSrgb, PermuteOpts, BlurSel};
use imgref::ImgVec;
use std::time::Instant;
fn load(p:&str)->(EncodedSrgb,ImgVec<[u8;3]>){let i=image::open(p).unwrap().into_rgb8();let(w,h)=i.dimensions();let a:ImgVec<[u8;3]>=ImgVec::new(i.pixels().map(|x|[x[0],x[1],x[2]]).collect(),w as _,h as _);(a.as_ref().to_encoded_srgb().unwrap(),a)}
fn rss_kb()->u64{std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l|l.starts_with("VmHWM")).map(|l|l.split_whitespace().nth(1).unwrap().parse().unwrap()).unwrap_or(0)}
fn main(){
    let a=std::env::args().nth(1).unwrap(); let b=std::env::args().nth(2).unwrap();
    let mode=std::env::args().nth(3).unwrap_or_else(||"all".into());
    let reps: u32 = std::env::args().nth(4).map(|s|s.parse().unwrap()).unwrap_or(5);
    let (e1,img1)=load(&a); let (e2,img2)=load(&b);
    let opts = PermuteOpts{blur:BlurSel::OfficialSimd,..PermuteOpts::OFFICIAL};
    let modes: [(&str, Box<dyn Fn()->f64>); 4] = [
        ("precise", Box::new(|| fast_ssim2::compute_ssimulacra2(img1.as_ref(),img2.as_ref()).unwrap()) as _),
        ("official", Box::new(|| official::compute_encoded_opts(&e1,&e2,opts).unwrap()) as _),
        ("official-strip64", Box::new(|| {
            let cfg = fast_ssim2::Ssimulacra2StripConfig::default().with_inner(fast_ssim2::Ssimulacra2Config::official());
            fast_ssim2::compute_ssimulacra2_strip_with_config(img1.as_ref(),img2.as_ref(),64,cfg).unwrap()
        }) as _),
        ("precise-strip64", Box::new(|| fast_ssim2::compute_ssimulacra2_strip(img1.as_ref(),img2.as_ref(),64).unwrap()) as _),
    ];
    for (name, run) in modes {
        if mode!="all" && mode!=name { continue; }
        let r0=rss_kb();
        std::hint::black_box(run()); // warm
        let warm=rss_kb();
        let t=Instant::now();
        for _ in 0..reps { std::hint::black_box(run()); }
        let dt=t.elapsed().as_secs_f64()/reps as f64*1e3;
        println!("{name}: {dt:.1}ms peak~{}MB", warm/1024);
        let _=r0;
    }
}
