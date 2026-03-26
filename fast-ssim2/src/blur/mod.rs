mod gaussian;
mod simd_gaussian;

use crate::SimdImpl;
use gaussian::RecursiveGaussian;
use simd_gaussian::SimdGaussian;

/// Structure handling image blur with selectable implementation.
///
/// Supports runtime switching between:
/// - Scalar: f64 IIR baseline (most accurate)
/// - SIMD: archmage cross-platform SIMD (AVX2, AVX-512, NEON, WASM128)
pub struct Blur {
    width: usize,
    height: usize,
    impl_type: SimdImpl,
    // Scalar backend
    scalar_kernel: RecursiveGaussian,
    scalar_temp: Vec<f32>,
    // SIMD backend (archmage)
    simd: SimdGaussian,
}

impl Blur {
    /// Create a new [Blur] with the default implementation (SIMD).
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_simd_impl(width, height, SimdImpl::default())
    }

    /// Create a new [Blur] with a specific implementation.
    #[must_use]
    pub fn with_simd_impl(width: usize, height: usize, impl_type: SimdImpl) -> Self {
        Blur {
            width,
            height,
            impl_type,
            scalar_kernel: RecursiveGaussian,
            scalar_temp: vec![0.0f32; width * height],
            simd: SimdGaussian::new(width),
        }
    }

    /// Get the current implementation type.
    pub fn impl_type(&self) -> SimdImpl {
        self.impl_type
    }

    /// Set the implementation type.
    pub fn set_impl(&mut self, impl_type: SimdImpl) {
        self.impl_type = impl_type;
    }

    /// Truncates the internal buffers to fit images of the given width and height.
    pub fn shrink_to(&mut self, width: usize, height: usize) {
        self.scalar_temp.truncate(width * height);
        self.simd.shrink_to(width, height);
        self.width = width;
        self.height = height;
    }

    /// Blur the given image using the selected implementation.
    pub fn blur(&mut self, img: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
        [
            self.blur_plane(&img[0]),
            self.blur_plane(&img[1]),
            self.blur_plane(&img[2]),
        ]
    }

    /// Blur the given image into pre-allocated output buffers (zero-allocation).
    pub fn blur_into(&mut self, img: &[Vec<f32>; 3], out: &mut [Vec<f32>; 3]) {
        self.blur_plane_into(&img[0], &mut out[0]);
        self.blur_plane_into(&img[1], &mut out[1]);
        self.blur_plane_into(&img[2], &mut out[2]);
    }

    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.width * self.height];
        self.blur_plane_into(plane, &mut out);
        out
    }

    fn blur_plane_into(&mut self, plane: &[f32], out: &mut [f32]) {
        match self.impl_type {
            SimdImpl::Scalar => self.blur_plane_scalar_into(plane, out),
            SimdImpl::Simd => self.blur_plane_simd_into(plane, out),
        }
    }

    fn blur_plane_scalar_into(&mut self, plane: &[f32], out: &mut [f32]) {
        self.scalar_kernel
            .horizontal_pass(plane, &mut self.scalar_temp, self.width);
        self.scalar_kernel.vertical_pass_chunked::<128, 32>(
            &self.scalar_temp,
            out,
            self.width,
            self.height,
        );
    }

    fn blur_plane_simd_into(&mut self, plane: &[f32], out: &mut [f32]) {
        self.simd
            .blur_single_plane_into(plane, out, self.width, self.height);
    }
}
