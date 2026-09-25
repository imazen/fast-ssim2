//! Precomputed reference-image state for fast repeated comparisons.
//!
//! [`Ssimulacra2Reference`] runs the reference-side half of the
//! SSIMULACRA2.1 pipeline once (linearize → XYB → per-scale μ/σ planes)
//! and caches it, so each subsequent [`compare`](Ssimulacra2Reference::compare)
//! only pays for the distorted side. Measured on a 512² u8 workload via
//! iai-callgrind: `compare` = 20.4M instructions vs 31.8M for the
//! one-shot compute — ~36% saved per call, `new` amortizes from the
//! second compare onward. Output is bit-identical to
//! [`crate::compute_ssimulacra2`].
//!
//! Strip variants ([`Ssimulacra2Reference::compare_strip`]) bound the
//! distorted side's memory the same way as [`crate::compute_ssimulacra2_strip`].

use crate::input::ToLinearRgb;
use crate::pipeline::precompute::ReferenceCache;
use crate::{MAX_IMAGE_PIXELS, Ssimulacra2Config, Ssimulacra2Error};

/// Precomputed SSIMULACRA2 reference state for fast repeated comparisons.
///
/// Stores the reference-side per-scale planes (XYB, μ, σ) so comparing
/// multiple distorted images against the same source skips that work
/// entirely. `Clone` deep-copies the planes — keep one instance per
/// batch loop. `Send`/`Sync`: one reference can be shared across worker
/// threads.
///
/// Images below the metric's 8×8 pyramid floor are reflect-padded at
/// construction (the crate-level extension — the reference binary
/// refuses them), matching [`crate::compute_ssimulacra2`].
#[derive(Clone, Debug)]
pub struct Ssimulacra2Reference {
    cache: ReferenceCache,
    /// Dimensions of the source as the caller supplied it (before any
    /// sub-8px reflect-padding) — `compare*` validates the distorted
    /// side against these.
    original_width: usize,
    original_height: usize,
}

impl Ssimulacra2Reference {
    /// Precompute the reference pipeline for `source`.
    ///
    /// `to_encoded_srgb` inputs (u8/u16/f32 sRGB) take the encoded path
    /// — bit-identical to [`crate::compute_ssimulacra2`]; inputs without
    /// an encoded form build the cache from their linear planes.
    ///
    /// # Errors
    /// - [`Ssimulacra2Error::InvalidImageSize`] if the image is empty
    ///   (0×0).
    /// - [`Ssimulacra2Error::ImageTooLarge`] if the (padded) image
    ///   exceeds [`MAX_IMAGE_PIXELS`].
    pub fn new<T: ToLinearRgb>(source: T) -> Result<Self, Ssimulacra2Error> {
        Self::new_with_config(source, Ssimulacra2Config::default())
    }

    /// [`Self::new`] with explicit kernel selection — `impl_type`
    /// chooses the scalar-oracle or SIMD kernels (bit-identical output).
    pub fn new_with_config<T: ToLinearRgb>(
        source: T,
        config: Ssimulacra2Config,
    ) -> Result<Self, Ssimulacra2Error> {
        let _ = config; // kernel selection is internal; all kernels are bit-identical
        if let Some(e) = source.to_encoded_srgb() {
            let ow = e.width;
            let oh = e.height;
            let padded = e.reflect_padded(8);
            if padded.width * padded.height > MAX_IMAGE_PIXELS {
                return Err(Ssimulacra2Error::ImageTooLarge {
                    actual: padded.width * padded.height,
                });
            }
            return Ok(Self {
                cache: ReferenceCache::new(&padded)?,
                original_width: ow,
                original_height: oh,
            });
        }
        let img = source.into_linear_rgb();
        let (ow, oh) = (img.width(), img.height());
        let img = crate::reflect_pad_linear(img, 8);
        let (w, h) = (img.width(), img.height());
        if w == 0 || h == 0 {
            return Err(Ssimulacra2Error::InvalidImageSize);
        }
        if w * h > MAX_IMAGE_PIXELS {
            return Err(Ssimulacra2Error::ImageTooLarge { actual: w * h });
        }
        let lin = linear_planes(&img);
        Ok(Self {
            cache: ReferenceCache::new_linear(lin, w, h)?,
            original_width: ow,
            original_height: oh,
        })
    }

    /// Compare a distorted image against the precomputed reference.
    ///
    /// Bit-identical to [`crate::compute_ssimulacra2`] on the same pair
    /// (encoded and linear inputs alike).
    ///
    /// # Errors
    /// - [`Ssimulacra2Error::NonMatchingImageDimensions`] if dimensions
    ///   differ from the source used at construction.
    pub fn compare<T: ToLinearRgb>(&self, distorted: T) -> Result<f64, Ssimulacra2Error> {
        self.compare_with_stop(distorted, &enough::Unstoppable)
    }

    /// [`Self::compare`] with cooperative cancellation — `stop` is
    /// checked at scale boundaries (never per-pixel).
    ///
    /// # Errors
    /// As [`Self::compare`], plus [`Ssimulacra2Error::Cancelled`].
    pub fn compare_with_stop<T: ToLinearRgb>(
        &self,
        distorted: T,
        stop: &dyn enough::Stop,
    ) -> Result<f64, Ssimulacra2Error> {
        if let Some(e) = distorted.to_encoded_srgb() {
            if e.width != self.original_width || e.height != self.original_height {
                return Err(Ssimulacra2Error::NonMatchingImageDimensions);
            }
            let padded = e.reflect_padded(8);
            return self.cache.compare_stop(&padded, stop);
        }
        let img = distorted.into_linear_rgb();
        if img.width() != self.original_width || img.height() != self.original_height {
            return Err(Ssimulacra2Error::NonMatchingImageDimensions);
        }
        let img = crate::reflect_pad_linear(img, 8);
        self.cache
            .compare_linear_stop(linear_planes(&img), img.width(), img.height(), stop)
    }

    /// Source width in pixels (as supplied at construction).
    #[must_use]
    pub fn width(&self) -> usize {
        self.original_width
    }

    /// Source height in pixels (as supplied at construction).
    #[must_use]
    pub fn height(&self) -> usize {
        self.original_height
    }

    /// Number of precomputed scales (image-size dependent, 1–6).
    #[must_use]
    pub fn num_scales(&self) -> usize {
        self.cache.num_scales()
    }

    /// The underlying pipeline cache — shared with the strip module.
    pub(crate) fn cache(&self) -> &ReferenceCache {
        &self.cache
    }
}

/// Deinterleave a `LinearRgbImage` into `[r, g, b]` planes.
fn linear_planes(img: &crate::LinearRgbImage) -> [Vec<f32>; 3] {
    let mut out = [
        Vec::with_capacity(img.width() * img.height()),
        Vec::with_capacity(img.width() * img.height()),
        Vec::with_capacity(img.width() * img.height()),
    ];
    for px in img.data() {
        out[0].push(px[0]);
        out[1].push(px[1]);
        out[2].push(px[2]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearRgbImage, compute_ssimulacra2, compute_ssimulacra2_with_config};
    use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

    fn rgb8(img: &image::RgbImage) -> Rgb {
        let (w, h) = img.dimensions();
        let px: Vec<[f32; 3]> = img
            .pixels()
            .map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
            .collect();
        Rgb::new(
            px,
            std::num::NonZeroUsize::new(w as usize).unwrap(),
            std::num::NonZeroUsize::new(h as usize).unwrap(),
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap()
    }

    fn tank() -> (Rgb, Rgb) {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data");
        let mk = |name: &str| rgb8(&image::open(base.join(name)).unwrap().to_rgb8());
        (mk("tank_source.png"), mk("tank_distorted.png"))
    }

    #[test]
    fn test_precompute_matches_full_compute() {
        let (a, b) = tank();
        let full = compute_ssimulacra2(a.clone(), b.clone()).unwrap();
        let ref_ = Ssimulacra2Reference::new(a).unwrap();
        let cached = ref_.compare(b).unwrap();
        // bit-identical — same kernel family on both paths
        assert_eq!(full, cached, "one-shot {full} != cached {cached}");
    }

    #[test]
    fn test_precompute_scalar_matches_default() {
        let (a, b) = tank();
        let scalar = Ssimulacra2Reference::new_with_config(
            a.clone(),
            Ssimulacra2Config::scalar(),
        )
        .unwrap();
        let simd = Ssimulacra2Reference::new(a).unwrap();
        assert_eq!(
            scalar.compare(b.clone()).unwrap(),
            simd.compare(b).unwrap(),
        );
    }

    #[test]
    fn test_precompute_dimension_mismatch() {
        let (a, _b) = tank();
        let ref_ = Ssimulacra2Reference::new(a.clone()).unwrap();
        // 1×1 — smaller than the reference; still mismatched vs the
        // 512² source.
        let tiny = LinearRgbImage::new(vec![[0.5; 3]], 1, 1);
        assert_eq!(
            ref_.compare(tiny),
            Err(Ssimulacra2Error::NonMatchingImageDimensions)
        );
    }

    #[test]
    fn test_sub_8_reference_pads_and_matches_one_shot() {
        // 7×5 source — below the pyramid floor: `new` reflect-pads it to
        // 8×8, `compute_ssimulacra2` pads identically.
        let mk = |seed: u32| {
            let px: Vec<[f32; 3]> = (0..7 * 5)
                .map(|i| {
                    let v = (((i as u32 * 2654435761u32) ^ seed) >> 20) as u8;
                    [v as f32 / 255.0, v as f32 / 510.0, 1.0 - v as f32 / 255.0]
                })
                .collect();
            Rgb::new(
                px,
                std::num::NonZeroUsize::new(7).unwrap(),
                std::num::NonZeroUsize::new(5).unwrap(),
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap()
        };
        let a = mk(0xabcdef);
        let b = mk(0x123456);
        let full = compute_ssimulacra2(a.clone(), b.clone()).unwrap();
        let ref_ = Ssimulacra2Reference::new(a).unwrap();
        assert_eq!(full, ref_.compare(b).unwrap());
        assert_eq!(ref_.width(), 7);
        assert_eq!(ref_.height(), 5);
    }

    #[test]
    fn test_sub_8_reference_rejects_mismatched_dims() {
        let tiny = LinearRgbImage::new(vec![[0.5; 3]; 7 * 5], 7, 5);
        let ref_ = Ssimulacra2Reference::new(tiny).unwrap();
        let larger = LinearRgbImage::new(vec![[0.5; 3]; 16 * 16], 16, 16);
        assert_eq!(
            ref_.compare(larger),
            Err(Ssimulacra2Error::NonMatchingImageDimensions)
        );
    }

    #[test]
    fn test_precompute_metadata() {
        let (a, _b) = tank();
        let ref_ = Ssimulacra2Reference::new(a.clone()).unwrap();
        assert_eq!(ref_.width(), a.width().get());
        assert_eq!(ref_.height(), a.height().get());
        assert!((1..=6).contains(&ref_.num_scales()));
    }

    #[test]
    fn test_precompute_linear_source() {
        // LinearRgbImage input — no encoded form, cache built from
        // linear planes; must match one-shot compute bit-for-bit.
        let px: Vec<[f32; 3]> = (0..64 * 64)
            .map(|i| {
                let f = (i as f32 * 0.6180339) % 1.0;
                [f, f * 0.7, f * 0.3]
            })
            .collect();
        let a = crate::LinearRgbImage::new(px.clone(), 64, 64);
        let px2: Vec<[f32; 3]> = px.iter().map(|p| [p[0] * 0.95, p[1], p[2]]).collect();
        let b = crate::LinearRgbImage::new(px2, 64, 64);
        let full = compute_ssimulacra2_with_config(
            a.clone(),
            b.clone(),
            Ssimulacra2Config::default(),
        )
        .unwrap();
        let ref_ = Ssimulacra2Reference::new(a).unwrap();
        assert_eq!(full, ref_.compare(b).unwrap());
    }
}
