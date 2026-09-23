//! Peak live-heap measurement for the metric's working buffers.
//!
//! Wraps the global allocator and reports the high-water mark of
//! `alloc'd - dealloc'd` bytes across one `compute_ssimulacra2` call.
//!
//! Run with:
//!   cargo run --release --example peak_memory -- [full|ref|ctx] [W] [H]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Instant;

use fast_ssim2::{LinearRgbImage, Ssimulacra2Reference, compute_ssimulacra2};

struct Peak;

static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Peak {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed)
                + layout.size() as isize;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            let delta = new_size as isize - layout.size() as isize;
            let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Peak = Peak;

fn make_pair(width: usize, height: usize) -> (LinearRgbImage, LinearRgbImage) {
    let mut state = 0x12345u64;
    let mut mk = |off: f32| {
        let mut v = Vec::with_capacity(width * height);
        for _ in 0..width * height {
            let mut px = [0f32; 3];
            for c in &mut px {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *c = (((state >> 33) & 0xFF) as f32 / 255.0) + off;
            }
            v.push(px);
        }
        v
    };
    (
        LinearRgbImage::new(mk(0.0), width, height),
        LinearRgbImage::new(mk(0.01), width, height),
    )
}

fn mib(b: isize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn peak() -> isize {
    PEAK.load(Ordering::Relaxed)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "full".to_string());
    let width: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3840);
    let height: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2160);

    let (src, dst) = make_pair(width, height);

    // Warm up so lazy tables/dispatch caches don't pollute the peak. The
    // fixture allocations stay live, so only the high-water mark resets.
    let _ = compute_ssimulacra2(src.clone(), dst.clone());
    PEAK.store(0, Ordering::Relaxed);

    let t = Instant::now();
    match mode.as_str() {
        "full" => {
            let s = compute_ssimulacra2(src, dst).unwrap();
            eprintln!("score {s:.4}");
        }
        // Reference precompute alone (the persistent-footprint case).
        "ref" => {
            let r = Ssimulacra2Reference::new(src).unwrap();
            eprintln!(
                "ref built, {} scales, peak {:.1} MiB",
                r.num_scales(),
                mib(peak())
            );
            let mut ctx = r.compare_context();
            PEAK.store(0, Ordering::Relaxed);
            let t2 = Instant::now();
            let s = r.compare_with(&mut ctx, dst).unwrap();
            eprintln!("compare_with score {s:.4}");
            println!(
                "compare_with {width}x{height}: peak live = {:.1} MiB in {:.1} ms",
                mib(peak()),
                t2.elapsed().as_secs_f64() * 1000.0
            );
            return;
        }
        other => {
            eprintln!("unknown mode {other} (full|ref)");
            std::process::exit(2);
        }
    }
    println!(
        "{mode} {width}x{height}: peak live = {:.1} MiB in {:.1} ms",
        mib(peak()),
        t.elapsed().as_secs_f64() * 1000.0
    );
}
