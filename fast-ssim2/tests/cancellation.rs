//! Cooperative-cancellation tests for the `*_with_stop` entry points.
//!
//! Verifies that passing an already-cancelled [`enough::Stop`] token to
//! [`compute_ssimulacra2_with_stop`] / [`compute_ssimulacra2_strip_with_stop`]
//! short-circuits with [`Ssimulacra2Error::Cancelled`], and that passing
//! [`enough::Unstoppable`] computes a score normally (identical inputs
//! land near the 100 ceiling).
//!
//! The cancellation check lives at the per-scale (one-shot) and
//! per-strip (strip) OUTER-loop boundary, so an already-cancelled token
//! fires on the very first iteration.

#![forbid(unsafe_code)]

use almost_enough::Stopper;
use enough::{StopReason, Unstoppable};
use fast_ssim2::{
    LinearRgbImage, Ssimulacra2Error, Ssimulacra2Reference, compute_ssimulacra2_strip_with_stop,
    compute_ssimulacra2_with_stop,
};

/// Build a deterministic non-trivial `width`x`height` linear-RGB image.
fn generate_image(width: usize, height: usize, seed: u32) -> LinearRgbImage {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = (((x as u32)
                .wrapping_mul(7)
                .wrapping_add((y as u32).wrapping_mul(13))
                .wrapping_add(seed))
                & 0xff) as f32
                / 255.0;
            let g = (((x as u32)
                .wrapping_mul(11)
                .wrapping_add((y as u32).wrapping_mul(3))
                .wrapping_add(seed.wrapping_add(50)))
                & 0xff) as f32
                / 255.0;
            let b = (((x as u32)
                .wrapping_mul(5)
                .wrapping_add((y as u32).wrapping_mul(17))
                .wrapping_add(seed.wrapping_add(100)))
                & 0xff) as f32
                / 255.0;
            data.push([r, g, b]);
        }
    }
    LinearRgbImage::new(data, width, height)
}

#[test]
fn one_shot_cancelled_token_returns_cancelled() {
    let source = generate_image(64, 64, 0);
    let distorted = generate_image(64, 64, 1);
    let result = compute_ssimulacra2_with_stop(source, distorted, &Stopper::cancelled());
    match result {
        Err(Ssimulacra2Error::Cancelled(reason)) => {
            assert_eq!(reason, StopReason::Cancelled);
        }
        other => panic!("expected Err(Cancelled(_)), got {other:?}"),
    }
}

#[test]
fn one_shot_unstoppable_token_computes_score() {
    let img = generate_image(64, 64, 42);
    let score =
        compute_ssimulacra2_with_stop(img.clone(), img, &Unstoppable).expect("Unstoppable must Ok");
    // Identical inputs score near the 100 ceiling.
    assert!(
        score > 99.0,
        "identical-image score {score:.4} should be near 100",
    );
}

#[test]
fn strip_cancelled_token_returns_cancelled() {
    let source = generate_image(64, 64, 7);
    let distorted = generate_image(64, 64, 8);
    let result = compute_ssimulacra2_strip_with_stop(source, distorted, 32, &Stopper::cancelled());
    match result {
        Err(Ssimulacra2Error::Cancelled(reason)) => {
            assert_eq!(reason, StopReason::Cancelled);
        }
        other => panic!("expected Err(Cancelled(_)), got {other:?}"),
    }
}

#[test]
fn strip_unstoppable_token_computes_score() {
    let img = generate_image(64, 64, 99);
    let score = compute_ssimulacra2_strip_with_stop(img.clone(), img, 32, &Unstoppable)
        .expect("Unstoppable must Ok");
    assert!(
        score > 99.0,
        "identical-image strip score {score:.4} should be near 100",
    );
}

// --- Cached-reference (Ssimulacra2Reference) batch paths ---

#[test]
fn cached_ref_compare_cancelled_returns_cancelled() {
    let reference = Ssimulacra2Reference::new(generate_image(64, 64, 3)).expect("reference build");
    let distorted = generate_image(64, 64, 4);
    match reference.compare_with_stop(distorted, &Stopper::cancelled()) {
        Err(Ssimulacra2Error::Cancelled(reason)) => assert_eq!(reason, StopReason::Cancelled),
        other => panic!("expected Err(Cancelled(_)), got {other:?}"),
    }
}

#[test]
fn cached_ref_compare_unstoppable_computes_score() {
    let img = generate_image(64, 64, 5);
    let reference = Ssimulacra2Reference::new(img.clone()).expect("reference build");
    let score = reference
        .compare_with_stop(img, &Unstoppable)
        .expect("Unstoppable must Ok");
    assert!(
        score > 99.0,
        "identical cached-ref score {score:.4} near 100"
    );
}

#[test]
fn cached_ref_strip_cancelled_returns_cancelled() {
    let reference = Ssimulacra2Reference::new(generate_image(64, 64, 6)).expect("reference build");
    let distorted = generate_image(64, 64, 7);
    match reference.compare_strip_with_stop(distorted, 32, &Stopper::cancelled()) {
        Err(Ssimulacra2Error::Cancelled(reason)) => assert_eq!(reason, StopReason::Cancelled),
        other => panic!("expected Err(Cancelled(_)), got {other:?}"),
    }
}

#[test]
fn cached_ref_strip_unstoppable_computes_score() {
    let img = generate_image(64, 64, 8);
    let reference = Ssimulacra2Reference::new(img.clone()).expect("reference build");
    let score = reference
        .compare_strip_with_stop(img, 32, &Unstoppable)
        .expect("Unstoppable must Ok");
    assert!(
        score > 99.0,
        "identical cached-ref strip score {score:.4} near 100"
    );
}
