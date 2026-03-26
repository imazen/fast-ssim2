//! Benchmark that tracks memory allocations during SSIMULACRA2 computation
//!
//! Run with:
//!   cargo run --release --example benchmark_allocations

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use fast_ssim2::{Ssimulacra2Config, compute_frame_ssimulacra2_with_config};
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

// Custom allocator that tracks allocations
struct TrackingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        DEALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn reset_counters() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn get_stats() -> (usize, usize, usize, usize) {
    (
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
        DEALLOC_COUNT.load(Ordering::Relaxed),
        DEALLOC_BYTES.load(Ordering::Relaxed),
    )
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn create_test_image(width: usize, height: usize, seed: u64) -> Rgb {
    let mut state = seed;
    let data: Vec<[f32; 3]> = (0..width * height)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 33) & 0xFF) as f32 / 255.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let g = ((state >> 33) & 0xFF) as f32 / 255.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b = ((state >> 33) & 0xFF) as f32 / 255.0;
            [r, g, b]
        })
        .collect();

    Rgb::new(
        data,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap()
}

fn benchmark_with_tracking(
    width: usize,
    height: usize,
    config: Ssimulacra2Config,
) -> (f64, usize, usize) {
    let source = create_test_image(width, height, 12345);
    let distorted = create_test_image(width, height, 67890);

    // Warmup (don't count)
    let _ = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config);

    // Reset and measure
    reset_counters();
    let start = Instant::now();
    let _ = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let (alloc_count, alloc_bytes, _, _) = get_stats();

    (elapsed_ms, alloc_count, alloc_bytes)
}

fn main() {
    println!("SSIMULACRA2 Allocation Benchmark");
    println!("=================================\n");

    let sizes = [
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
        (1920, 1080, "FHD"),
        (2560, 1440, "QHD"),
        (3840, 2160, "4K"),
    ];

    // Theoretical minimum: 2 images * width * height * 3 channels * 4 bytes
    println!("Image data sizes (2 images, RGB f32):");
    for (w, h, name) in &sizes {
        let image_bytes = w * h * 3 * 4 * 2;
        println!("  {:12}: {}", name, format_bytes(image_bytes));
    }
    println!();

    println!("Scalar configuration:");
    println!(
        "{:12} {:>10} {:>12} {:>14} {:>10}",
        "Size", "Time (ms)", "Allocs", "Bytes", "Bytes/px"
    );
    println!("{:-<65}", "");

    for (w, h, name) in &sizes {
        let (ms, allocs, bytes) = benchmark_with_tracking(*w, *h, Ssimulacra2Config::scalar());
        let bytes_per_pixel = bytes as f64 / (*w * *h) as f64;
        println!(
            "{:12} {:>10.1} {:>12} {:>14} {:>10.1}",
            name,
            ms,
            allocs,
            format_bytes(bytes),
            bytes_per_pixel
        );
    }

    println!("\nSIMD configuration:");
    println!(
        "{:12} {:>10} {:>12} {:>14} {:>10}",
        "Size", "Time (ms)", "Allocs", "Bytes", "Bytes/px"
    );
    println!("{:-<65}", "");

    for (w, h, name) in &sizes {
        let (ms, allocs, bytes) = benchmark_with_tracking(*w, *h, Ssimulacra2Config::simd());
        let bytes_per_pixel = bytes as f64 / (*w * *h) as f64;
        println!(
            "{:12} {:>10.1} {:>12} {:>14} {:>10.1}",
            name,
            ms,
            allocs,
            format_bytes(bytes),
            bytes_per_pixel
        );
    }

    println!("\nDone.");
}
