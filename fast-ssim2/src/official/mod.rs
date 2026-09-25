//! `Fidelity::MatchOfficial` — bit-exact reimplementation of the reference
//! SSIMULACRA2 pipeline (`ssimulacra2.cc` + vendored libjxl primitives).
//!
//! Cloudinary's standalone `ssimulacra2` and the `ssimulacra2` tool shipped
//! in libjxl v0.12.0 produce identical scores (verified on 400+ image pairs
//! and by source diff); this module reproduces that single behavior,
//! including its floating-point quirks:
//!
//! - sRGB → linear via captured LUTs (lcms2's sampled-curve evaluation,
//!   which differs from both the exact EOTF and `TF_SRGB`'s rational
//!   polynomial by up to ~2.6e-7)
//! - `CubeRootAndAdd` (not `cbrt`)
//! - `FastGaussian` with the reference's exact 4-lane-unrolled FMA ordering
//! - `SSIMMap`/`EdgeDiffMap` with the f32-quotient/f64-aggregate split
//! - `Score` with the reference's exact f64 op ordering
//!
//! Bit-exactness is guaranteed for quantized inputs (8-bit/16-bit images,
//! JPEG decoded to u8). Arbitrary f32-encoded inputs are linearized by
//! interpolation into the captured LUT — close to, but not guaranteed
//! bit-identical with, the reference's lcms evaluation.

pub mod gauss;
pub mod simd;
pub mod strip;
mod lut8;
pub mod precompute;
pub mod maps;
#[cfg(feature = "rayon")]
use archmage::incant;
pub mod score;
pub mod xyb;

pub(crate) use score::score as final_score;
use gauss::{create_recursive_gaussian, multiply_planes, RecursiveGaussian};
use score::ScaleAggregates;

/// Encoded sRGB pixel data for the match-official input path.
///
/// Returned by [`crate::input::ToLinearRgb::to_encoded_srgb`] for inputs
/// that carry quantized/encoded sRGB values rather than linear data.
pub enum EncodedData {
    /// Interleaved RGB u8 triples (e.g. PNG-8 decode).
    U8(Vec<u8>),
    /// Interleaved RGB u16 triples (e.g. PNG-16 decode).
    U16(Vec<u16>),
    /// Interleaved encoded-sRGB f32 triples in [0, 1] (e.g. `yuvxyb::Rgb`).
    F32(Vec<f32>),
}

/// Encoded sRGB image handed to the match-official pipeline.
pub struct EncodedSrgb {
    pub width: usize,
    pub height: usize,
    pub data: EncodedData,
    /// Normalized alpha plane (`v * (1/max)` like the reference decode), if
    /// the input has an alpha channel. When present the reference binary
    /// alpha-blends onto a background and takes the worst of two scores.
    pub alpha: Option<Vec<f32>>,
}

impl EncodedSrgb {
    /// Extract the row range `[y0, y1)` as a new `EncodedSrgb` (strip
    /// slicing — keeps encoded data encoded so linearization still uses
    /// the LUT path per-strip).
    pub fn strip_rows(&self, y0: usize, y1: usize) -> Self {
        let w = self.width;
        let h = y1 - y0;
        let (a0, a1) = (y0 * w * 3, y1 * w * 3);
        let data = match &self.data {
            EncodedData::U8(d) => EncodedData::U8(d[a0..a1].to_vec()),
            EncodedData::U16(d) => EncodedData::U16(d[a0..a1].to_vec()),
            EncodedData::F32(d) => EncodedData::F32(d[a0..a1].to_vec()),
        };
        let alpha = self
            .alpha
            .as_ref()
            .map(|a| a[y0 * w..y1 * w].to_vec());
        EncodedSrgb { width: w, height: h, data, alpha }
    }

    /// Reflect(mirror)-pad encoded planes up to `min` px on each axis —
    /// the crate-level sub-8px contract (`compute_ssimulacra2` scores
    /// down to 1×1 where the reference binary refuses). Per-pixel LUT
    /// linearization makes padding encoded planes equivalent to padding
    /// post-linearization, so this is faithful to [`reflect_pad_linear`]
    /// while preserving `U8` exactness. NO-OP at ≥ `min` (or empty).
    pub fn reflect_padded(&self, min: usize) -> Self {
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 || (w >= min && h >= min) {
            return self.clone_shallow();
        }
        let (pw, ph) = (w.max(min), h.max(min));
        let data = match &self.data {
            EncodedData::U8(d) => EncodedData::U8(pad_px(w, h, pw, ph, d)),
            EncodedData::U16(d) => EncodedData::U16(pad_px(w, h, pw, ph, d)),
            EncodedData::F32(d) => EncodedData::F32(pad_px(w, h, pw, ph, d)),
        };
        let alpha = self
            .alpha
            .as_ref()
            .map(|a| pad_scalars(w, h, pw, ph, a));
        EncodedSrgb {
            width: pw,
            height: ph,
            data,
            alpha,
        }
    }

    /// Cheap clone for the no-pad case (data Vec is shared via clone —
    /// encode data only cloned when padding actually fires).
    fn clone_shallow(&self) -> Self {
        EncodedSrgb {
            width: self.width,
            height: self.height,
            data: match &self.data {
                EncodedData::U8(d) => EncodedData::U8(d.clone()),
                EncodedData::U16(d) => EncodedData::U16(d.clone()),
                EncodedData::F32(d) => EncodedData::F32(d.clone()),
            },
            alpha: self.alpha.clone(),
        }
    }
}

/// Periodic edge-mirror, same convention as `reflect_pad_linear` in the
/// crate root: `i` beyond `n` mirrors without repeating the edge sample;
/// `n <= 1` collapses to 0.
fn reflect_index(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut k = i % period;
    if k >= n {
        k = period - k;
    }
    k
}

fn pad_px<T: Copy>(w: usize, h: usize, pw: usize, ph: usize, data: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(pw * ph * 3);
    for y in 0..ph {
        let row = reflect_index(y, h) * w;
        for x in 0..pw {
            let i = (row + reflect_index(x, w)) * 3;
            out.extend_from_slice(&data[i..i + 3]);
        }
    }
    out
}

fn pad_scalars(w: usize, h: usize, pw: usize, ph: usize, data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pw * ph);
    for y in 0..ph {
        let row = reflect_index(y, h) * w;
        for x in 0..pw {
            out.push(data[row + reflect_index(x, w)]);
        }
    }
    out
}

/// Reference linearization: the sRGB curve as evaluated by the official
/// binary's cms (lcms2 in the solo repo), captured as a 256-entry LUT for
/// u8 inputs. u16 and arbitrary f32 inputs use the crate's `srgb_to_linear`
/// rational polynomial (spec algorithm; not bit-exact vs lcms).
///
/// `bg` is the alpha-blend background used by the reference `AlphaBlend`
/// (`a * v + (1 - a) * bg` in encoded space). The standalone binary calls
/// the metric twice — `bg = 0.1` and `bg = 0.9` — and keeps the worse
/// score. `bg` is ignored when `enc.alpha` is `None`.
pub fn official_linearize(enc: &EncodedSrgb, bg: f32) -> [Vec<f32>; 3] {
    official_linearize_opts(enc, bg, false)
}

/// `lin_poly`: use the polynomial `srgb_to_linear` for u8 inputs instead of
/// the captured reference LUT (ablation axis — breaks bit-exactness).
pub fn official_linearize_opts(enc: &EncodedSrgb, bg: f32, lin_poly: bool) -> [Vec<f32>; 3] {
    let n = enc.width * enc.height;
    let mut out = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
    if let Some(alpha) = &enc.alpha {
        let encoded = |i: usize, v: f32| {
            let af = alpha[i];
            encoded_f32_to_linear(af * v + (1.0 - af) * bg)
        };
        match &enc.data {
            EncodedData::U8(data) => {
                for (i, px) in data.chunks_exact(3).enumerate() {
                    out[0].push(encoded(i, px[0] as f32 * (1.0 / 255.0)));
                    out[1].push(encoded(i, px[1] as f32 * (1.0 / 255.0)));
                    out[2].push(encoded(i, px[2] as f32 * (1.0 / 255.0)));
                }
            }
            EncodedData::U16(data) => {
                for (i, px) in data.chunks_exact(3).enumerate() {
                    out[0].push(encoded(i, px[0] as f32 * (1.0 / 65535.0)));
                    out[1].push(encoded(i, px[1] as f32 * (1.0 / 65535.0)));
                    out[2].push(encoded(i, px[2] as f32 * (1.0 / 65535.0)));
                }
            }
            EncodedData::F32(data) => {
                for (i, px) in data.chunks_exact(3).enumerate() {
                    out[0].push(encoded(i, px[0]));
                    out[1].push(encoded(i, px[1]));
                    out[2].push(encoded(i, px[2]));
                }
            }
        }
        return out;
    }
    match &enc.data {
        EncodedData::U8(data) => {
            if lin_poly {
                for px in data.chunks_exact(3) {
                    out[0].push(encoded_f32_to_linear(px[0] as f32 * (1.0 / 255.0)));
                    out[1].push(encoded_f32_to_linear(px[1] as f32 * (1.0 / 255.0)));
                    out[2].push(encoded_f32_to_linear(px[2] as f32 * (1.0 / 255.0)));
                }
            } else {
                for px in data.chunks_exact(3) {
                    out[0].push(lut8::LINEAR_LUT_U8[px[0] as usize]);
                    out[1].push(lut8::LINEAR_LUT_U8[px[1] as usize]);
                    out[2].push(lut8::LINEAR_LUT_U8[px[2] as usize]);
                }
            }
        }
        EncodedData::U16(data) => {
            for px in data.chunks_exact(3) {
                out[0].push(crate::input::srgb_to_linear(px[0] as f32 * (1.0 / 65535.0)));
                out[1].push(crate::input::srgb_to_linear(px[1] as f32 * (1.0 / 65535.0)));
                out[2].push(crate::input::srgb_to_linear(px[2] as f32 * (1.0 / 65535.0)));
            }
        }
        EncodedData::F32(data) => {
            for px in data.chunks_exact(3) {
                out[0].push(encoded_f32_to_linear_grid(px[0], !lin_poly));
                out[1].push(encoded_f32_to_linear_grid(px[1], !lin_poly));
                out[2].push(encoded_f32_to_linear_grid(px[2], !lin_poly));
            }
        }
    }
    out
}

/// Reference-linearization approximation for arbitrary encoded f32 in
/// [0,1] (alpha-blended or unquantized inputs): the crate's `srgb_to_linear`
/// rational polynomial (libjxl `TF_SRGB`). NOT bit-exact vs the reference's
/// lcms evaluation on off-grid values — deviation ~1e-7 max.
#[inline]
fn encoded_f32_to_linear(x: f32) -> f32 {
    crate::input::srgb_to_linear(x.clamp(0.0, 1.0))
}

/// `lin_poly = false` variant for [`EncodedData::F32`]: values sitting on
/// the u8 grid (`x ≈ k/255`, the overwhelmingly common caller — u8 data
/// widened to f32) snap to the captured reference LUT, making them
/// bit-exact with the reference's u8 decode; off-grid values (alpha
/// blends, true arbitrary f32) use the spec polynomial — the closest
/// available approximation of lcms's actual float transform.
#[inline]
fn encoded_f32_to_linear_grid(x: f32, use_lut: bool) -> f32 {
    let v = x.clamp(0.0, 1.0) * 255.0;
    if use_lut {
        let r = v.round();
        if (v - r).abs() < 1e-5 {
            return lut8::LINEAR_LUT_U8[r as usize];
        }
    }
    crate::input::srgb_to_linear(x.clamp(0.0, 1.0))
}

/// Reference `Downsample` — linear RGB, box 2×2, ceil output size,
/// clamped edge taps, `sum += ` in iy-outer/ix-inner order, `* 0.25`.
pub fn downsample_planes(p: &[Vec<f32>; 3], width: usize, height: usize) -> ([Vec<f32>; 3], usize, usize) {
    let nw = width.div_ceil(2);
    let nh = height.div_ceil(2);
    let mut out = [
        vec![0f32; nw * nh],
        vec![0f32; nw * nh],
        vec![0f32; nw * nh],
    ];
    // Per-(channel,row-group) writes are disjoint — parallel when rayon
    // is on; the per-pixel math is identical either way.
    let fill = |c: usize, oy: usize, row: &mut [f32]| {
        for (ox, o) in row.iter_mut().enumerate() {
            let x0 = (2 * ox).min(width - 1);
            let x1 = (2 * ox + 1).min(width - 1);
            let y0 = (2 * oy).min(height - 1);
            let y1 = (2 * oy + 1).min(height - 1);
            *o = (p[c][y0 * width + x0]
                + p[c][y0 * width + x1]
                + p[c][y1 * width + x0]
                + p[c][y1 * width + x1])
                * 0.25;
        }
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(c, plane)| {
            plane
                .par_chunks_mut(nw)
                .enumerate()
                .for_each(|(oy, row)| fill(c, oy, row));
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for c in 0..3 {
            for (oy, row) in out[c].chunks_mut(nw).enumerate() {
                fill(c, oy, row);
            }
        }
    }
    (out, nw, nh)
}

/// Gaussian-blur a 3-plane image with the reference `FastGaussian`
/// (horizontal into temp, then vertical).
pub fn blur_planes(rg: &RecursiveGaussian, p: &[Vec<f32>; 3], width: usize, height: usize) -> [Vec<f32>; 3] {
    let mut out = [vec![0f32; width * height], vec![0f32; width * height], vec![0f32; width * height]];
    let mut tmp = vec![0f32; width * height];
    for c in 0..3 {
        // Horizontal pass: each row independently.
        for y in 0..height {
            rg.fast_gaussian_1d(
                &p[c][y * width..(y + 1) * width],
                &mut tmp[y * width..(y + 1) * width],
            );
        }
        // Vertical pass: each column independently.
        let tmp_ref = &tmp;
        rg.fast_gaussian_vertical_1d(
            width,
            height,
            |row, x| tmp_ref[row * width + x],
            &mut |row, x, v| out[c][row * width + x] = v,
        );
    }
    out
}

/// `blur_planes` writing into caller-provided scratch (strip-mode reuse).
/// `out`/`tmp` planes sized ≥ width*height; `tmp` shared across channels
/// (scalar path is serial anyway).
pub fn blur_planes_into(
    rg: &RecursiveGaussian,
    p: &[Vec<f32>; 3],
    width: usize,
    height: usize,
    out: &mut [Vec<f32>; 3],
    tmp: &mut [Vec<f32>; 3],
) {
    for c in 0..3 {
        for y in 0..height {
            rg.fast_gaussian_1d(
                &p[c][y * width..(y + 1) * width],
                &mut tmp[c][y * width..(y + 1) * width],
            );
        }
        let tmp_ref = &tmp[c];
        rg.fast_gaussian_vertical_1d(
            width,
            height,
            |row, x| tmp_ref[row * width + x],
            &mut |row, x, v| out[c][row * width + x] = v,
        );
    }
}

/// Convert linear-RGB planes to positive-XYB planes in place
/// (`LinearRGBToXYB` + `MakePositiveXYB` per pixel).
pub fn planes_to_positive_xyb(p: &mut [Vec<f32>; 3], npix: usize) {
    planes_to_positive_xyb_opts(p, npix, CbrtMode::Official)
}

fn planes_to_positive_xyb_opts(p: &mut [Vec<f32>; 3], npix: usize, cbrt: CbrtMode) {
    for i in 0..npix {
        let px = xyb::linear_rgb_to_xyb_pixel_opts([p[0][i], p[1][i], p[2][i]], cbrt);
        p[0][i] = px[0];
        p[1][i] = px[1];
        p[2][i] = px[2];
    }
}

/// Run the metric on encoded inputs, including the reference binary's
/// alpha handling: when the source has an alpha channel it is evaluated
/// twice — blended against `bg = 0.1` and `bg = 0.9` in encoded space —
/// and the worse (lower) score is returned, matching `ssimulacra2_main.cc`.
/// A distorted-side alpha without a source-side alpha is blended at the
/// single default `bg = 0.5`, as in `ComputeSSIMULACRA2`'s default arg.
#[allow(dead_code)]
pub(crate) fn compute_encoded(
    enc1: &EncodedSrgb,
    enc2: &EncodedSrgb,
) -> Result<f64, crate::Ssimulacra2Error> {
    compute_encoded_opts(enc1, enc2, PermuteOpts::OFFICIAL)
}

/// Variant-selectable [`compute_encoded`] — the permutation-study entry.
pub fn compute_encoded_opts(
    enc1: &EncodedSrgb,
    enc2: &EncodedSrgb,
    opts: PermuteOpts,
) -> Result<f64, crate::Ssimulacra2Error> {
    compute_encoded_opts_stop(enc1, enc2, opts, &enough::Unstoppable)
}

/// [`compute_encoded_opts`] with cooperative cancellation — `stop` is
/// checked once per scale (never per-pixel), same semantics as the
/// precise path.
pub fn compute_encoded_opts_stop(
    enc1: &EncodedSrgb,
    enc2: &EncodedSrgb,
    opts: PermuteOpts,
    stop: &dyn enough::Stop,
) -> Result<f64, crate::Ssimulacra2Error> {
    let (w, h) = (enc1.width, enc1.height);
    if w != enc2.width || h != enc2.height {
        return Err(crate::Ssimulacra2Error::NonMatchingImageDimensions);
    }
    let lin = |e: &EncodedSrgb, bg| official_linearize_opts(e, bg, opts.lin_poly);
    if enc1.alpha.is_some() {
        let lo = compute_opts_stop(lin(enc1, 0.1), lin(enc2, 0.1), w, h, opts, stop)?;
        let hi = compute_opts_stop(lin(enc1, 0.9), lin(enc2, 0.9), w, h, opts, stop)?;
        return Ok(lo.min(hi));
    }
    compute_opts_stop(lin(enc1, 0.5), lin(enc2, 0.5), w, h, opts, stop)
}

/// Cube-root implementation used inside `linear_rgb_to_xyb` +
/// `MakePositiveXYB`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CbrtMode {
    /// Reference `CubeRootAndAdd` bit-hack (bit-exact match to official).
    Official,
    /// Standard `f32::cbrt` (correctly rounded).
    Std,
    /// `magetypes::cbrt_midp_f32` (~3 ulp, Halley).
    MagetypesMidp,
}

/// Gaussian blur implementation variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlurSel {
    /// Scalar port of the reference `FastGaussian` (bit-exact ordering).
    Official,
    /// Lane-wise SIMD of the same sequence — bit-identical to `Official`.
    OfficialSimd,
    /// The crate's fast path, scalar tier.
    PreciseScalar,
    /// The crate's fast path, SIMD tier.
    PreciseSimd,
}

/// Per-axis toggles for ablation experiments. `PermuteOpts::OFFICIAL`
/// reproduces the reference binary bit-for-bit (for u8-encoded inputs).
#[derive(Clone, Copy, Debug)]
pub struct PermuteOpts {
    pub cbrt: CbrtMode,
    /// σ cross-terms in f64 (removes the reference's f32 cancellation
    /// noise) instead of the official f32 computation.
    pub sigma_f64: bool,
    pub blur: BlurSel,
    /// u8 linearization: polynomial `srgb_to_linear` instead of the
    /// captured lcms LUT.
    pub lin_poly: bool,
}

impl PermuteOpts {
    /// All-official configuration — bit-exact reproduction.
    pub const OFFICIAL: Self = Self {
        cbrt: CbrtMode::Official,
        sigma_f64: false,
        blur: BlurSel::Official,
        lin_poly: false,
    };
    /// "More precise" configuration — correctly-rounded cbrt, f64 σ
    /// cancellation-free math, fast SIMD blur, polynomial linearization.
    pub const PRECISE_LEAN: Self = Self {
        cbrt: CbrtMode::Std,
        sigma_f64: true,
        blur: BlurSel::PreciseSimd,
        lin_poly: true,
    };
}

/// The complete match-official metric. Takes already-linear planes
/// (from [`official_linearize`] or equivalent).
///
/// Uses the lane-wise SIMD kernels (bit-identical to the scalar port);
/// `PermuteOpts::OFFICIAL` (scalar) remains the reference oracle.
///
/// Returns `Err(Ssimulacra2Error::InvalidImageSize)` below 8×8, matching
/// the reference binary's minimum-size behavior.
pub fn compute(
    lin1: [Vec<f32>; 3],
    lin2: [Vec<f32>; 3],
    width: usize,
    height: usize,
) -> Result<f64, crate::Ssimulacra2Error> {
    compute_opts_stop(
        lin1, lin2, width, height,
        PermuteOpts { blur: BlurSel::OfficialSimd, ..PermuteOpts::OFFICIAL },
        &enough::Unstoppable,
    )
}

/// The complete pipeline with per-axis implementation selection.
pub fn compute_opts(
    lin1: [Vec<f32>; 3],
    lin2: [Vec<f32>; 3],
    width: usize,
    height: usize,
    opts: PermuteOpts,
) -> Result<f64, crate::Ssimulacra2Error> {
    compute_opts_stop(lin1, lin2, width, height, opts, &enough::Unstoppable)
}

/// [`compute_opts`] with cooperative cancellation — `stop` is checked
/// once per scale (never per-pixel).
pub fn compute_opts_stop(
    lin1: [Vec<f32>; 3],
    lin2: [Vec<f32>; 3],
    width: usize,
    height: usize,
    opts: PermuteOpts,
    stop: &dyn enough::Stop,
) -> Result<f64, crate::Ssimulacra2Error> {
    if width < 8 || height < 8 {
        return Err(crate::Ssimulacra2Error::InvalidImageSize);
    }

    let rg = create_recursive_gaussian(1.5);
    const NUM_SCALES: usize = 6;

    let mut lin1 = lin1;
    let mut lin2 = lin2;
    let mut w = width;
    let mut h = height;
    let mut scales = Vec::with_capacity(NUM_SCALES);

    // NOTE: the reference gates `size < 8` on the *pre-downsample* dims —
    // a scale produced by downsampling into <8 territory still runs its
    // maps (e.g. 10x8 -> 5x4 runs scale maps at 5x4). `gw`,`gh` track the
    // gate dims (previous iteration's); `w`,`h` are the current scale's.
    let mut gw = width;
    let mut gh = height;
    for _scale in 0..NUM_SCALES {
        stop.check().map_err(crate::Ssimulacra2Error::Cancelled)?;
        if gw < 8 || gh < 8 {
            break;
        }
        // Produce next scale's linear planes first, then convert the
        // current linear planes to XYB *in place* — saves 6 plane
        // allocations (lin1/lin2 clones) per scale.
        let (lin1_next, nw, nh) = downsample_planes(&lin1, w, h);
        let (lin2_next, _, _) = downsample_planes(&lin2, w, h);

        let npix = w * h;
        let mut xyb1 = lin1;
        let mut xyb2 = lin2;
        if opts.blur == BlurSel::OfficialSimd && opts.cbrt == CbrtMode::Official {
            simd::planes_to_positive_xyb_simd(&mut xyb1);
            simd::planes_to_positive_xyb_simd(&mut xyb2);
        } else {
            planes_to_positive_xyb_opts(&mut xyb1, npix, opts.cbrt);
            planes_to_positive_xyb_opts(&mut xyb2, npix, opts.cbrt);
        }

        let blur_sel = |p: &[Vec<f32>; 3]| -> [Vec<f32>; 3] {
            match opts.blur {
                BlurSel::Official => blur_planes(&rg, p, w, h),
                BlurSel::OfficialSimd => simd::blur_planes_simd(&rg, p, w, h),
                sel => {
                    let impl_type = match sel {
                        BlurSel::PreciseScalar => crate::SimdImpl::Scalar,
                        _ => crate::SimdImpl::Simd,
                    };
                    crate::blur::Blur::with_simd_impl(w, h, impl_type).blur(p)
                }
            }
        };

        // mul = xyb_i * xyb_i → blurred squares; mul = xyb1 * xyb2 → cross.
        // The five blurs (σ1,σ2,σ12,μ1,μ2) are mutually independent.
        // SIMD path: products fuse into the blur's input read (the f32
        // elementwise product is identical → bit-exact), so no `mul`
        // planes are materialized at all. Scalar path keeps the
        // reference's literal mul-then-blur sequence.
        let (sigma1_sq, sigma2_sq, sigma12, mu1, mu2);
        if opts.blur == BlurSel::OfficialSimd {
            // jobs: (a, b) — b present → blur of the product a·b.
            let jobs: [(&[Vec<f32>; 3], Option<&[Vec<f32>; 3]>); 5] = [
                (&xyb1, Some(&xyb1)),
                (&xyb2, Some(&xyb2)),
                (&xyb1, Some(&xyb2)),
                (&xyb1, None),
                (&xyb2, None),
            ];
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                let chan: Vec<(usize, usize)> =
                    (0..5).flat_map(|j| (0..3).map(move |c| (j, c))).collect();
                let flat: Vec<Vec<f32>> = chan
                    .into_par_iter()
                    .map(|(j, c)| {
                        let mut out = vec![0f32; npix];
                        let mut tmp = vec![0f32; npix];
                        let (a, b) = jobs[j];
                        simd::fast_gaussian_simd(
                            &rg, &a[c], b.map(|bb| bb[c].as_slice()),
                            w, h, &mut out, &mut tmp,
                        );
                        out
                    })
                    .collect();
                let take3 = |it: &mut std::vec::IntoIter<Vec<f32>>| {
                    [it.next().unwrap(), it.next().unwrap(), it.next().unwrap()]
                };
                let mut it = flat.into_iter();
                sigma1_sq = take3(&mut it);
                sigma2_sq = take3(&mut it);
                sigma12 = take3(&mut it);
                mu1 = take3(&mut it);
                mu2 = take3(&mut it);
            }
            #[cfg(not(feature = "rayon"))]
            {
                let run = |(a, b): (&[Vec<f32>; 3], Option<&[Vec<f32>; 3]>)| -> [Vec<f32>; 3] {
                    let mut out = [
                        vec![0f32; npix], vec![0f32; npix], vec![0f32; npix],
                    ];
                    let mut tmp = vec![0f32; npix];
                    for c in 0..3 {
                        simd::fast_gaussian_simd(
                            &rg, &a[c], b.map(|bb| bb[c].as_slice()),
                            w, h, &mut out[c], &mut tmp,
                        );
                    }
                    out
                };
                sigma1_sq = run(jobs[0]);
                sigma2_sq = run(jobs[1]);
                sigma12 = run(jobs[2]);
                mu1 = run(jobs[3]);
                mu2 = run(jobs[4]);
            }
        } else {
            let mut mul = [vec![0f32; npix], vec![0f32; npix], vec![0f32; npix]];
            multiply_planes(&xyb1, &xyb1, &mut mul);
            sigma1_sq = blur_sel(&mul);
            multiply_planes(&xyb2, &xyb2, &mut mul);
            sigma2_sq = blur_sel(&mul);
            multiply_planes(&xyb1, &xyb2, &mut mul);
            sigma12 = blur_sel(&mul);
            mu1 = blur_sel(&xyb1);
            mu2 = blur_sel(&xyb2);
        }

        // ssim (lanes) + edge_diff (scalar f64) run per channel — the
        // per-channel accumulation order is unchanged → bit-exact. The
        // six channel-tasks are independent → parallel under rayon.
        let (avg_ssim, avg_edgediff) = if opts.blur == BlurSel::OfficialSimd && !opts.sigma_f64 {
            #[cfg(feature = "rayon")]
            {
                let (mut so, mut eo) = ([0f64; 6], [0f64; 12]);
                use rayon::prelude::*;
                let n = w * h;
                let opp = 1.0 / n as f64;
                let parts: Vec<(f64, f64, [f64; 4])> = (0..3).into_par_iter()
                    .map(|c| {
                        let (mut s0, mut s1) = (0f64, 0f64);
                        incant!(simd::ssim_map_inner(
                            &mu1[c], &mu2[c], &sigma1_sq[c], &sigma2_sq[c], &sigma12[c],
                            &mut s0, &mut s1), [v3, neon, wasm128, scalar]);
                        let mut e = [0f64; 4];
                        simd::edge_sums_fast(0, h, w, &xyb1[c], &mu1[c], &xyb2[c], &mu2[c], &mut e);
                        (opp * s0, (opp * s1).sqrt().sqrt(), [
                            opp * e[0], (opp * e[1]).sqrt().sqrt(),
                            opp * e[2], (opp * e[3]).sqrt().sqrt()])
                    })
                    .collect();
                for c in 0..3 {
                    so[c * 2] = parts[c].0;
                    so[c * 2 + 1] = parts[c].1;
                    eo[c * 4..c * 4 + 4].copy_from_slice(&parts[c].2);
                }
                (so, eo)
            }
            #[cfg(not(feature = "rayon"))]
            {
                simd::maps_fused_simd(
                    &mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12, &xyb1, &xyb2, w, h,
                )
            }
        } else {
            maps::maps_fused(&mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12, &xyb1, &xyb2, w, h)
        };
        scales.push(ScaleAggregates {
            avg_ssim,
            avg_edgediff,
        });

        lin1 = lin1_next;
        lin2 = lin2_next;
        gw = w;
        gh = h;
        w = nw;
        h = nh;
    }

    Ok(final_score(&scales))
}
