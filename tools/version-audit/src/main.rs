//! Interleaved single-thread comparisons on identical RGB8 input buffers.
//! Input construction and decoding are outside the timed region.
use std::{sync::Arc, time::Duration};

fn main() {
    let rounds: usize = std::env::var("AUDIT_ROUNDS")
        .unwrap_or_else(|_| "7".into())
        .parse()
        .unwrap();
    assert!(rounds > 0);
    let result = zenbench::run_gated(zenbench::GateConfig::disabled(), |suite| {
        for (w, h) in [(320, 240), (1920, 1080), (3840, 2160)] {
            let mut state = 0x0bad_c0de_dead_beefu64;
            let src: Vec<[u8; 3]> = (0..w * h)
                .map(|i| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let noise = ((state >> 33) & 255) as u8;
                    [
                        (i % w * 255 / (w - 1)) as u8,
                        (i / w * 255 / (h - 1)) as u8,
                        noise,
                    ]
                })
                .collect();
            let dst: Vec<[u8; 3]> = src
                .iter()
                .map(|p| [p[0].saturating_add(3), p[1], p[2].saturating_sub(2)])
                .collect();
            let src = Arc::new(src);
            let dst = Arc::new(dst);
            for (version, label) in ["0.7.1", "0.8.2", "current"].into_iter().enumerate() {
                println!(
                    "score\t{w}\t{h}\t{label}\t{:.12}",
                    score(&src, &dst, w, h, version)
                );
            }
            suite.compare(format!("rgb8_{w}x{h}"), move |g| {
                g.baseline("0.8.2");
                g.config()
                    .max_rounds(rounds)
                    .min_rounds(rounds)
                    .warmup_time(Duration::from_millis(200));
                g.config().auto_rounds = false;
                g.config().max_time = Duration::from_secs(90);
                g.config().max_wall_time = Duration::from_secs(120);
                for (version, label) in ["0.7.1", "0.8.2", "current"].into_iter().enumerate() {
                    let (src, dst) = (Arc::clone(&src), Arc::clone(&dst));
                    g.bench(label, move |b| b.iter(|| score(&src, &dst, w, h, version)));
                }
            });
        }
    });
    zenbench::postprocess_result(&result);
}

fn score(src: &[[u8; 3]], dst: &[[u8; 3]], w: usize, h: usize, version: usize) -> f64 {
    match version {
        0 => v071::compute_ssimulacra2(imgref::Img::new(src, w, h), imgref::Img::new(dst, w, h))
            .unwrap(),
        1 => v082::compute_ssimulacra2(imgref::Img::new(src, w, h), imgref::Img::new(dst, w, h))
            .unwrap(),
        2 => {
            let a = current::PixelSlice::new(
                bytemuck::cast_slice(src),
                w as u32,
                h as u32,
                w * 3,
                current::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            let b = current::PixelSlice::new(
                bytemuck::cast_slice(dst),
                w as u32,
                h as u32,
                w * 3,
                current::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            current::compute_ssimulacra2(&a, &b).unwrap()
        }
        _ => unreachable!(),
    }
}
