// Timing+RSS sweep: sizes × modes. Prints CSV lines.
use fast_ssim2::pipeline::{self, EncodedData, EncodedSrgb, Kernel};
use std::time::Instant;
use zenpixels::{PixelDescriptor, PixelSlice};
fn load(p: &str) -> (EncodedSrgb, Vec<[u8; 3]>, u32, u32) {
    let i = image::open(p).unwrap().into_rgb8();
    let (w, h) = i.dimensions();
    let px: Vec<[u8; 3]> = i.pixels().map(|x| [x[0], x[1], x[2]]).collect();
    (
        EncodedSrgb {
            width: w as usize,
            height: h as usize,
            data: EncodedData::U8(px.iter().flatten().copied().collect()),
            alpha: None,
        },
        px,
        w,
        h,
    )
}
fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|l| l.starts_with("VmHWM"))
        .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
        .unwrap_or(0)
}
fn main() {
    let a = std::env::args().nth(1).unwrap();
    let b = std::env::args().nth(2).unwrap();
    let mode = std::env::args().nth(3).unwrap_or_else(|| "all".into());
    let reps: u32 = std::env::args()
        .nth(4)
        .map(|s| s.parse().unwrap())
        .unwrap_or(5);
    let (e1, img1, w, h) = load(&a);
    let (e2, img2, _, _) = load(&b);

    type Run<'a> = Box<dyn Fn() -> f64 + 'a>;
    let modes: [(&str, Run); 4] = [
        (
            "full",
            Box::new(|| {
                let a = PixelSlice::new(
                    img1.as_flattened(),
                    w,
                    h,
                    (w * 3) as usize,
                    PixelDescriptor::RGB8_SRGB,
                )
                .unwrap();
                let b = PixelSlice::new(
                    img2.as_flattened(),
                    w,
                    h,
                    (w * 3) as usize,
                    PixelDescriptor::RGB8_SRGB,
                )
                .unwrap();
                fast_ssim2::compute_ssimulacra2(&a, &b).unwrap()
            }) as _,
        ),
        (
            "encoded",
            Box::new(|| pipeline::compute_encoded(&e1, &e2, Kernel::Simd).unwrap()) as _,
        ),
        (
            "strip64",
            Box::new(|| {
                let a = PixelSlice::new(
                    img1.as_flattened(),
                    w,
                    h,
                    (w * 3) as usize,
                    PixelDescriptor::RGB8_SRGB,
                )
                .unwrap();
                let b = PixelSlice::new(
                    img2.as_flattened(),
                    w,
                    h,
                    (w * 3) as usize,
                    PixelDescriptor::RGB8_SRGB,
                )
                .unwrap();
                fast_ssim2::compute_ssimulacra2_with_config(
                    &a,
                    &b,
                    &fast_ssim2::Ssimulacra2Config::strips(64),
                )
                .unwrap()
            }) as _,
        ),
        (
            "scalar",
            Box::new(|| pipeline::compute_encoded(&e1, &e2, Kernel::Scalar).unwrap()) as _,
        ),
    ];
    for (name, run) in modes {
        if mode != "all" && mode != name {
            continue;
        }
        let r0 = rss_kb();
        std::hint::black_box(run()); // warm
        let warm = rss_kb();
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(run());
        }
        let dt = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        println!("{name}: {dt:.1}ms peak~{}MB", warm / 1024);
        let _ = r0;
    }
}
