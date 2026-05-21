//! Precomputed reference data for fast repeated SSIMULACRA2 comparisons.
//!
//! When comparing multiple distorted images against the same reference image,
//! you can precompute the reference data once and reuse it for ~2x speedup.
//!
//! # Example
//!
//! ```
//! use fast_ssim2::Ssimulacra2Reference;
//! use yuvxyb::{Rgb, TransferCharacteristic, ColorPrimaries};
//!
//! // Load reference image
//! use std::num::NonZeroUsize;
//! let reference_rgb = vec![[1.0f32, 1.0, 1.0]; 512 * 512];
//! let reference = Rgb::new(
//!     reference_rgb,
//!     NonZeroUsize::new(512).unwrap(),
//!     NonZeroUsize::new(512).unwrap(),
//!     TransferCharacteristic::SRGB,
//!     ColorPrimaries::BT709,
//! ).unwrap();
//!
//! // Precompute reference data once
//! let precomputed = Ssimulacra2Reference::new(reference).unwrap();
//!
//! // Compare against a distorted image
//! let distorted_rgb = vec![[0.9f32, 0.95, 1.05]; 512 * 512];
//! let distorted = Rgb::new(
//!     distorted_rgb,
//!     NonZeroUsize::new(512).unwrap(),
//!     NonZeroUsize::new(512).unwrap(),
//!     TransferCharacteristic::SRGB,
//!     ColorPrimaries::BT709,
//! ).unwrap();
//! let score = precomputed.compare(distorted).unwrap();
//! println!("SSIMULACRA2 score: {}", score);
//! ```

use crate::blur::Blur;
use crate::input::ToLinearRgb;
use crate::{
    LinearRgb, Msssim, MsssimScale, NUM_SCALES, SimdImpl, Ssimulacra2Error, downscale_by_2,
    edge_diff_map, image_multiply, linear_rgb_to_xyb_simd, make_positive_xyb, ssim_map,
    xyb_to_planar, xyb_to_planar_into,
};

/// Reusable scratch buffers for [`Ssimulacra2Reference::compare_with`].
///
/// `Ssimulacra2Reference::compare` allocates roughly 13 image-sized
/// `Vec<f32>` planes (`mul`, `mu2`, `sigma2_sq`, `sigma12`, `img2_planar`)
/// plus the [`Blur`] working memory on every call. When you compare many
/// distorted images against the same reference (encoder rate-distortion
/// search, simulated annealing, picker training), reuse a `CompareContext`
/// to amortise those allocations across all calls. Buffers grow only on
/// the first call and are reused thereafter; later calls do no `Vec` heap
/// allocation.
///
/// Allocated for a specific reference dimension via
/// [`Ssimulacra2Reference::compare_context`]; passed to
/// [`Ssimulacra2Reference::compare_with`].
///
/// `Send` but not `Sync` — give each worker thread its own context.
pub struct CompareContext {
    width: usize,
    height: usize,
    blur: Blur,
    mul: [Vec<f32>; 3],
    mu2: [Vec<f32>; 3],
    sigma2_sq: [Vec<f32>; 3],
    sigma12: [Vec<f32>; 3],
    img2_planar: [Vec<f32>; 3],
}

impl CompareContext {
    fn new(width: usize, height: usize) -> Self {
        let alloc_plane = || vec![0.0f32; width * height];
        let alloc_3planes = || [alloc_plane(), alloc_plane(), alloc_plane()];
        Self {
            width,
            height,
            blur: Blur::new(width, height),
            mul: alloc_3planes(),
            mu2: alloc_3planes(),
            sigma2_sq: alloc_3planes(),
            sigma12: alloc_3planes(),
            img2_planar: alloc_3planes(),
        }
    }

    /// Restore the working buffers to the original reference dimensions.
    /// Called at the start of each comparison so previous calls' truncations
    /// don't leave the buffers under-sized for the next call's scale 0.
    /// Cheap: the underlying `Vec` capacity is retained from construction,
    /// so this only updates length (no allocation) plus fills the regrown
    /// portion with zero.
    fn reset_to_full(&mut self) {
        let size = self.width * self.height;
        for buf in [
            &mut self.mul,
            &mut self.mu2,
            &mut self.sigma2_sq,
            &mut self.sigma12,
            &mut self.img2_planar,
        ] {
            for c in buf.iter_mut() {
                c.resize(size, 0.0);
            }
        }
        self.blur.shrink_to(self.width, self.height);
    }

    /// Truncate the working buffers to fit `width * height` of the current scale.
    /// `Vec::truncate` does not free memory, so subsequent scales just shrink
    /// and we never reallocate while iterating the pyramid.
    fn shrink_to(&mut self, width: usize, height: usize) {
        let size = width * height;
        for buf in [
            &mut self.mul,
            &mut self.mu2,
            &mut self.sigma2_sq,
            &mut self.sigma12,
            &mut self.img2_planar,
        ] {
            for c in buf.iter_mut() {
                c.truncate(size);
            }
        }
        self.blur.shrink_to(width, height);
    }
}

/// Precomputed reference data for a single scale.
#[derive(Clone, Debug)]
struct ScaleData {
    /// Planar XYB representation of reference image
    img1_planar: [Vec<f32>; 3],
    /// blur(img1) - mean of reference
    mu1: [Vec<f32>; 3],
    /// blur(img1 * img1) - variance component of reference
    sigma1_sq: [Vec<f32>; 3],
}

/// Precomputed SSIMULACRA2 reference data for fast repeated comparisons.
///
/// This struct stores precomputed data for the reference image at all scales,
/// allowing you to quickly compare multiple distorted images against the same
/// reference without recomputing the reference-side data each time.
///
/// For simulated annealing or other optimization where you compare many variations
/// against the same source, this provides approximately 2x speedup.
#[derive(Clone, Debug)]
pub struct Ssimulacra2Reference {
    scales: Vec<ScaleData>,
    original_width: usize,
    original_height: usize,
}

impl Ssimulacra2Reference {
    /// Precompute reference data for the given source image.
    ///
    /// Supports:
    /// - `imgref` types (with the `imgref` feature): `ImgRef<[u8; 3]>`, `ImgRef<[f32; 3]>`, etc.
    /// - `yuvxyb` types: `Rgb`, `LinearRgb`
    /// - Custom types implementing [`ToLinearRgb`]
    ///
    /// # Errors
    /// - If the image is smaller than 8x8 pixels
    /// - If the image exceeds [`crate::MAX_IMAGE_PIXELS`] pixels
    pub fn new<T: ToLinearRgb>(source: T) -> Result<Self, Ssimulacra2Error> {
        let mut img1: LinearRgb = source.into_linear_rgb().into();
        if img1.width().get() < 8 || img1.height().get() < 8 {
            return Err(Ssimulacra2Error::InvalidImageSize);
        }

        // Cap pixel count to prevent unbounded working-buffer allocation.
        let pixels = img1
            .width()
            .get()
            .checked_mul(img1.height().get())
            .ok_or(Ssimulacra2Error::ImageTooLarge { actual: usize::MAX })?;
        if pixels > crate::MAX_IMAGE_PIXELS {
            return Err(Ssimulacra2Error::ImageTooLarge { actual: pixels });
        }

        let original_width = img1.width().get();
        let original_height = img1.height().get();
        let mut width = original_width;
        let mut height = original_height;

        let mut mul = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];
        let mut blur = Blur::new(width, height);
        let mut scales = Vec::with_capacity(NUM_SCALES);

        for scale in 0..NUM_SCALES {
            if width < 8 || height < 8 {
                break;
            }

            if scale > 0 {
                img1 = downscale_by_2(&img1);
                width = img1.width().get();
                height = img1.height().get();
            }

            for c in &mut mul {
                c.truncate(width * height);
            }
            blur.shrink_to(width, height);

            let mut img1_xyb = linear_rgb_to_xyb_simd(img1.clone());
            make_positive_xyb(&mut img1_xyb);

            let img1_planar = xyb_to_planar(&img1_xyb);

            // Precompute mu1 = blur(img1)
            let mu1 = blur.blur(&img1_planar);

            // Precompute sigma1_sq = blur(img1 * img1)
            image_multiply(&img1_planar, &img1_planar, &mut mul, SimdImpl::default());
            let sigma1_sq = blur.blur(&mul);

            scales.push(ScaleData {
                img1_planar,
                mu1,
                sigma1_sq,
            });
        }

        Ok(Self {
            scales,
            original_width,
            original_height,
        })
    }

    /// Allocate a [`CompareContext`] sized for this reference's dimensions.
    ///
    /// Pair this with [`Self::compare_with`] to do repeated comparisons
    /// without allocating fresh working buffers on each call.
    #[must_use]
    pub fn compare_context(&self) -> CompareContext {
        CompareContext::new(self.original_width, self.original_height)
    }

    /// Compare a distorted image against the precomputed reference.
    ///
    /// This is approximately 2x faster than calling `compute_ssimulacra2`
    /// because it only needs to process the distorted image and compute cross-terms.
    ///
    /// For batch comparisons (many distorted images vs the same reference),
    /// prefer [`Self::compare_with`] together with a reusable
    /// [`CompareContext`] — that path performs zero `Vec` allocations after
    /// the first call.
    ///
    /// # Errors
    /// - If the distorted image dimensions don't match the reference
    pub fn compare<T: ToLinearRgb>(&self, distorted: T) -> Result<f64, Ssimulacra2Error> {
        let mut ctx = self.compare_context();
        self.compare_with(&mut ctx, distorted)
    }

    /// Compare a distorted image against the precomputed reference, reusing
    /// the scratch buffers in `ctx`. Zero `Vec` allocations after the first
    /// call (`ctx` retains its buffers between invocations).
    ///
    /// `ctx` must have been produced by [`Self::compare_context`] on this
    /// reference. Using a context sized for different dimensions returns
    /// [`Ssimulacra2Error::NonMatchingImageDimensions`].
    ///
    /// # Errors
    /// - If the distorted image dimensions don't match the reference
    /// - If `ctx` was sized for a different reference
    pub fn compare_with<T: ToLinearRgb>(
        &self,
        ctx: &mut CompareContext,
        distorted: T,
    ) -> Result<f64, Ssimulacra2Error> {
        let mut img2: LinearRgb = distorted.into_linear_rgb().into();
        if img2.width().get() != self.original_width || img2.height().get() != self.original_height
        {
            return Err(Ssimulacra2Error::NonMatchingImageDimensions);
        }
        if ctx.width != self.original_width || ctx.height != self.original_height {
            return Err(Ssimulacra2Error::NonMatchingImageDimensions);
        }

        let mut width = img2.width().get();
        let mut height = img2.height().get();

        // Re-expand buffers to full reference size in case a previous call
        // left them truncated to a small scale. `Vec::resize` reuses
        // existing capacity, so no heap allocation happens after the first
        // `compare_context()` call.
        ctx.reset_to_full();

        // Use the actual number of cached reference scales — the skip-map
        // must agree with what `score()`'s linear WEIGHT walk will index.
        let scales_n = self.scales.len();
        let mut msssim = Msssim::default();

        for (scale_idx, scale_data) in self.scales.iter().enumerate() {
            if width < 8 || height < 8 {
                break;
            }

            if scale_idx > 0 {
                img2 = downscale_by_2(&img2);
                width = img2.width().get();
                height = img2.height().get();
            }

            ctx.shrink_to(width, height);

            let mut img2_xyb = linear_rgb_to_xyb_simd(img2.clone());
            make_positive_xyb(&mut img2_xyb);

            // Reuse ctx.img2_planar instead of allocating a fresh [Vec; 3].
            xyb_to_planar_into(&img2_xyb, &mut ctx.img2_planar);

            // mu2 = blur(img2)
            ctx.blur.blur_into(&ctx.img2_planar, &mut ctx.mu2);

            // sigma2_sq = blur(img2 * img2)
            image_multiply(
                &ctx.img2_planar,
                &ctx.img2_planar,
                &mut ctx.mul,
                SimdImpl::default(),
            );
            ctx.blur.blur_into(&ctx.mul, &mut ctx.sigma2_sq);

            // sigma12 = blur(img1 * img2) — cross-term
            image_multiply(
                &scale_data.img1_planar,
                &ctx.img2_planar,
                &mut ctx.mul,
                SimdImpl::default(),
            );
            ctx.blur.blur_into(&ctx.mul, &mut ctx.sigma12);

            // Use precomputed mu1 and sigma1_sq from reference
            let avg_ssim = ssim_map(
                scales_n,
                scale_idx,
                width,
                height,
                &scale_data.mu1,
                &ctx.mu2,
                &scale_data.sigma1_sq,
                &ctx.sigma2_sq,
                &ctx.sigma12,
                SimdImpl::default(),
            );

            let avg_edgediff = edge_diff_map(
                scales_n,
                scale_idx,
                width,
                height,
                &scale_data.img1_planar,
                &scale_data.mu1,
                &ctx.img2_planar,
                &ctx.mu2,
                SimdImpl::default(),
            );

            msssim.scales.push(MsssimScale {
                avg_ssim,
                avg_edgediff,
            });
        }

        Ok(msssim.score())
    }

    /// Get the width of the original reference image.
    #[must_use]
    pub fn width(&self) -> usize {
        self.original_width
    }

    /// Get the height of the original reference image.
    #[must_use]
    pub fn height(&self) -> usize {
        self.original_height
    }

    /// Get the number of scales that were precomputed.
    #[must_use]
    pub fn num_scales(&self) -> usize {
        self.scales.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_ssimulacra2;
    use std::num::NonZeroUsize;
    use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

    #[test]
    fn test_precompute_matches_full_compute() {
        // Create a simple test image
        let width = 64usize;
        let height = 64usize;
        let nz_width = NonZeroUsize::new(width).unwrap();
        let nz_height = NonZeroUsize::new(height).unwrap();
        let source_data: Vec<[f32; 3]> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f32 / width as f32;
                let y = (i / width) as f32 / height as f32;
                [x, y, 0.5]
            })
            .collect();

        let distorted_data: Vec<[f32; 3]> = source_data
            .iter()
            .map(|&[r, g, b]| [r * 0.9, g * 0.95, b * 1.05])
            .collect();

        let source = Rgb::new(
            source_data.clone(),
            nz_width,
            nz_height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let distorted = Rgb::new(
            distorted_data,
            nz_width,
            nz_height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        // Compute using full method
        let source_clone = Rgb::new(
            source_data,
            nz_width,
            nz_height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let full_score = compute_ssimulacra2(source_clone, distorted.clone()).unwrap();

        // Compute using precomputed reference
        let precomputed = Ssimulacra2Reference::new(source).unwrap();
        let precomputed_score = precomputed.compare(distorted).unwrap();

        // Scores should match exactly (both use same SIMD XYB path)
        assert!(
            (full_score - precomputed_score).abs() < 1e-6,
            "Scores don't match: full={}, precomputed={}",
            full_score,
            precomputed_score
        );
    }

    #[test]
    fn test_precompute_dimension_mismatch() {
        let source_data: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 64 * 64];
        let distorted_data: Vec<[f32; 3]> = vec![[0.4, 0.4, 0.4]; 32 * 32]; // Wrong size

        let source = Rgb::new(
            source_data,
            NonZeroUsize::new(64).unwrap(),
            NonZeroUsize::new(64).unwrap(),
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let distorted = Rgb::new(
            distorted_data,
            NonZeroUsize::new(32).unwrap(),
            NonZeroUsize::new(32).unwrap(),
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let precomputed = Ssimulacra2Reference::new(source).unwrap();
        let result = precomputed.compare(distorted);

        assert!(matches!(
            result,
            Err(Ssimulacra2Error::NonMatchingImageDimensions)
        ));
    }

    #[test]
    fn test_compare_with_matches_compare() {
        // `compare_with(ctx, ..)` must produce the same score as `compare(..)`
        // — it's just the zero-alloc form of the same computation. We compare
        // the two paths on a small JPEG-like RGB pair.
        let width = 64usize;
        let height = 64usize;
        let nz_width = NonZeroUsize::new(width).unwrap();
        let nz_height = NonZeroUsize::new(height).unwrap();
        let source_data: Vec<[f32; 3]> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f32 / width as f32;
                let y = (i / width) as f32 / height as f32;
                [x, y, 0.5]
            })
            .collect();
        let distorted_data: Vec<[f32; 3]> = source_data
            .iter()
            .map(|&[r, g, b]| [r * 0.92, g * 0.97, b * 1.03])
            .collect();
        let source = Rgb::new(
            source_data,
            nz_width,
            nz_height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let distorted = Rgb::new(
            distorted_data,
            nz_width,
            nz_height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let precomputed = Ssimulacra2Reference::new(source).unwrap();
        let score_compare = precomputed.compare(distorted.clone()).unwrap();
        let mut ctx = precomputed.compare_context();
        let score_compare_with = precomputed
            .compare_with(&mut ctx, distorted.clone())
            .unwrap();
        // Calling compare_with a second time exercises buffer reuse — the
        // result must still match exactly.
        let score_compare_with_repeat = precomputed.compare_with(&mut ctx, distorted).unwrap();

        // Both paths share the SIMD ops, so the scores should be exactly
        // equal modulo reduce-order. 1e-9 leaves room for the f64
        // accumulator order to differ if rustc reorders the loops.
        assert!(
            (score_compare - score_compare_with).abs() < 1e-9,
            "compare={} vs compare_with={}",
            score_compare,
            score_compare_with
        );
        assert!(
            (score_compare_with - score_compare_with_repeat).abs() < 1e-12,
            "compare_with should be deterministic across reuse"
        );
    }

    #[test]
    fn test_compare_context_dimension_mismatch() {
        // A context allocated for one reference is rejected if used with a
        // different-dimension distorted image (would be impossible to size
        // the scratch buffers correctly).
        let source_a: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 64 * 64];
        let source_b: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 32 * 32];
        let ref_a = Ssimulacra2Reference::new(
            Rgb::new(
                source_a,
                NonZeroUsize::new(64).unwrap(),
                NonZeroUsize::new(64).unwrap(),
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap(),
        )
        .unwrap();
        let distorted_b = Rgb::new(
            source_b,
            NonZeroUsize::new(32).unwrap(),
            NonZeroUsize::new(32).unwrap(),
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let mut ctx = ref_a.compare_context();
        assert!(matches!(
            ref_a.compare_with(&mut ctx, distorted_b),
            Err(Ssimulacra2Error::NonMatchingImageDimensions)
        ));
    }

    #[test]
    fn test_precompute_metadata() {
        let data: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 128 * 96];
        let source = Rgb::new(
            data,
            NonZeroUsize::new(128).unwrap(),
            NonZeroUsize::new(96).unwrap(),
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let precomputed = Ssimulacra2Reference::new(source).unwrap();

        assert_eq!(precomputed.width(), 128);
        assert_eq!(precomputed.height(), 96);
        assert!(precomputed.num_scales() > 0);
        assert!(precomputed.num_scales() <= NUM_SCALES);
    }
}
