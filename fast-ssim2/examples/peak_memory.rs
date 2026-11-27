//! Run under /usr/bin/time -v or heaptrack to measure full/reference peak memory.
//! Usage: cargo run --release --example peak_memory -- [full|ref|cached] [W] [H]
#![forbid(unsafe_code)]
use fast_ssim2::{PixelBuf, PixelFormat, Ssimulacra2Reference, compute_ssimulacra2};
fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "full".into());
    let w: u32 = args.next().map(|v| v.parse().unwrap()).unwrap_or(3840);
    let h: u32 = args.next().map(|v| v.parse().unwrap()).unwrap_or(2160);
    let a = PixelBuf::from_bytes(
        vec![128; w as usize * h as usize * 3],
        w,
        h,
        PixelFormat::RGB8_SRGB,
    )
    .unwrap();
    let b = PixelBuf::from_bytes(
        vec![120; w as usize * h as usize * 3],
        w,
        h,
        PixelFormat::RGB8_SRGB,
    )
    .unwrap();
    match mode.as_str() {
        "full" => println!(
            "score {}",
            compute_ssimulacra2(&a.as_view(), &b.as_view()).unwrap()
        ),
        "ref" => println!(
            "scales {}",
            Ssimulacra2Reference::new(&a.as_view())
                .unwrap()
                .num_scales()
        ),
        "cached" => {
            let reference = Ssimulacra2Reference::new(&a.as_view()).unwrap();
            println!("score {}", reference.compare(&b.as_view()).unwrap());
        }
        _ => panic!("expected full, ref, or cached"),
    }
}
