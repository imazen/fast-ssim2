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
//! let reference_rgb = vec![[1.0f32, 1.0, 1.0]; 512 * 512];
//! let reference = Rgb::new(
//!     reference_rgb,
//!     512,
//!     512,
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
//!     512,
//!     512,
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
    xyb_to_planar,
};

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
    pub fn new<T: ToLinearRgb>(source: T) -> Result<Self, Ssimulacra2Error> {
        let mut img1: LinearRgb = source.into_linear_rgb().into();
        if img1.width() < 8 || img1.height() < 8 {
            return Err(Ssimulacra2Error::InvalidImageSize);
        }

        let original_width = img1.width();
        let original_height = img1.height();
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
                width = img1.width();
                height = img1.height();
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

    /// Compare a distorted image against the precomputed reference.
    ///
    /// This is approximately 2x faster than calling `compute_ssimulacra2`
    /// because it only needs to process the distorted image and compute cross-terms.
    ///
    /// # Errors
    /// - If the distorted image dimensions don't match the reference
    pub fn compare<T: ToLinearRgb>(&self, distorted: T) -> Result<f64, Ssimulacra2Error> {
        let mut img2: LinearRgb = distorted.into_linear_rgb().into();
        if img2.width() != self.original_width || img2.height() != self.original_height {
            return Err(Ssimulacra2Error::NonMatchingImageDimensions);
        }

        let mut width = img2.width();
        let mut height = img2.height();

        let mut mul = [
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
            vec![0.0f32; width * height],
        ];
        let mut blur = Blur::new(width, height);
        let mut msssim = Msssim::default();

        for (scale_idx, scale_data) in self.scales.iter().enumerate() {
            if width < 8 || height < 8 {
                break;
            }

            if scale_idx > 0 {
                img2 = downscale_by_2(&img2);
                width = img2.width();
                height = img2.height();
            }

            for c in &mut mul {
                c.truncate(width * height);
            }
            blur.shrink_to(width, height);

            let mut img2_xyb = linear_rgb_to_xyb_simd(img2.clone());
            make_positive_xyb(&mut img2_xyb);

            let img2_planar = xyb_to_planar(&img2_xyb);

            // Compute mu2 = blur(img2)
            let mu2 = blur.blur(&img2_planar);

            // Compute sigma2_sq = blur(img2 * img2)
            image_multiply(&img2_planar, &img2_planar, &mut mul, SimdImpl::default());
            let sigma2_sq = blur.blur(&mul);

            // Compute sigma12 = blur(img1 * img2) - cross-term
            image_multiply(
                &scale_data.img1_planar,
                &img2_planar,
                &mut mul,
                SimdImpl::default(),
            );
            let sigma12 = blur.blur(&mul);

            // Use precomputed mu1 and sigma1_sq from reference
            let avg_ssim = ssim_map(
                width,
                height,
                &scale_data.mu1,
                &mu2,
                &scale_data.sigma1_sq,
                &sigma2_sq,
                &sigma12,
                SimdImpl::default(),
            );

            let avg_edgediff = edge_diff_map(
                width,
                height,
                &scale_data.img1_planar,
                &scale_data.mu1,
                &img2_planar,
                &mu2,
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
    use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

    #[test]
    fn test_precompute_matches_full_compute() {
        // Create a simple test image
        let width = 64;
        let height = 64;
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
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let distorted = Rgb::new(
            distorted_data,
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        // Compute using full method
        let source_clone = Rgb::new(
            source_data,
            width,
            height,
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
            64,
            64,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let distorted = Rgb::new(
            distorted_data,
            32,
            32,
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
    fn test_precompute_metadata() {
        let data: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 128 * 96];
        let source = Rgb::new(
            data,
            128,
            96,
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
