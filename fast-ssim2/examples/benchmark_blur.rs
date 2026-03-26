/// Quick benchmark to compare blur backend performance
use fast_ssim2::Blur;
use std::time::Instant;

fn main() {
    let sizes = [(512, 512), (1024, 1024), (2048, 2048)];

    for (width, height) in sizes {
        println!("\n=== {}x{} image ===", width, height);

        let test_data = vec![0.5f32; width * height];
        let img = [test_data.clone(), test_data.clone(), test_data];

        let mut blur = Blur::new(width, height);

        // Warmup
        blur.blur(&img);

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            blur.blur(&img);
        }
        let elapsed = start.elapsed();

        let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let mpix_per_sec = (width * height) as f64 / 1_000_000.0 / (ms_per_iter / 1000.0);

        println!("Time per blur: {:.3} ms", ms_per_iter);
        println!("Throughput: {:.1} Mpix/sec", mpix_per_sec);
    }
}
