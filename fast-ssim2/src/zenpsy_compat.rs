//! `zenpsy-api` trait implementations for `fast-ssim2`.
//!
//! Provides [`Ssimulacra2Metric`] implementing [`zenpsy_api::Metric`] and
//! [`zenpsy_api::PrepareReference`] (returning owned [`PreparedSsimulacra2`]):
//! - **Score-only by default:** [`EvalOpts::new()`] computes only the scalar
//!   SSIMULACRA2 score and allocates no spatial maps.
//! - **Opt-in maps:**
//!   - [`MapRequest::Native`] exposes the native per-scale XYB `ssim_error`,
//!     `artifact`, and `detail_lost` planes (`s{scale}_{x|y|b}_{term}`) as
//!     auxiliary [`PlaneRole::Extra`] planes.
//!   - [`MapRequest::Named("diffmap")`] exposes the full-resolution fused
//!     error map ([`Ssimulacra2Maps::diffmap`]) as [`PlaneRole::Extra(0)`].
//!   - Unknown named map requests fail immediately with [`ErrorKind::NoMap`]
//!     before any computation or allocation.
//! - **Strict ingress & domain binding:** Rejects pending non-identity
//!   orientation, premultiplied alpha, non-finite `f32` samples, unsupported
//!   explicit HLG display rendering overrides (non-zero black, custom gamma,
//!   or `DiffuseWhite` overriding `HlgMode::Display`), mixed SDR vs native-HDR
//!   pairs, and strip-mode `PrepareReference` (whose full-image reference
//!   pyramid cannot preserve strip-local downsampling grid parity).

use std::sync::OnceLock;

use enough::{Stop, StopReason};
use zenpixels::{
    AlphaMode, ChannelOrder, ColorSpace, DiffuseWhite, HlgMode, Orientation, PixelView,
    PlaneFormat, PlaneRole, PlaneView, Primaries, SampleStride, SampleType, Transfer,
};
use zenpsy_api::{
    Direction, ErrorKind, EvalOpts, Evaluation, MapGrid, MapRequest, MapView, Metric, MetricError,
    MetricId, MetricResult, PrepareReference, Score, ScoreSpec, WarmReference, fail, whereat,
};

use crate::{
    SimdImpl, Ssimulacra2Config, Ssimulacra2Error, Ssimulacra2Maps, Ssimulacra2Reference,
    StripConfig, compute_ssimulacra2_maps_with_config, compute_ssimulacra2_with_config,
};

/// Latches the first observed [`StopReason`] so a one-shot stop token (e.g. a
/// transient timeout) is preserved across phases and parallel work.
struct LatchingStop<'a> {
    inner: &'a dyn Stop,
    latched: OnceLock<StopReason>,
}

impl<'a> LatchingStop<'a> {
    fn new(inner: &'a dyn Stop) -> Self {
        Self {
            inner,
            latched: OnceLock::new(),
        }
    }

    fn reason(&self) -> Option<StopReason> {
        self.latched.get().copied()
    }
}

impl Stop for LatchingStop<'_> {
    fn check(&self) -> Result<(), StopReason> {
        if let Some(&reason) = self.latched.get() {
            return Err(reason);
        }
        if let Err(reason) = self.inner.check() {
            let _ = self.latched.set(reason);
            return Err(self.latched.get().copied().unwrap_or(reason));
        }
        Ok(())
    }

    fn may_stop(&self) -> bool {
        self.inner.may_stop() || self.latched.get().is_some()
    }
}

static SSIM2_SCORE_SPEC: ScoreSpec =
    ScoreSpec::new("ssimulacra2", Direction::Higher).with_perfect(100.0);
static SSIM2_DIFFMAP_SPEC: ScoreSpec =
    ScoreSpec::new("diffmap", Direction::Lower).with_perfect(0.0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestedMaps {
    None,
    Native,
    Diffmap,
}

/// Configured SSIMULACRA2 scorer implementing [`zenpsy_api::Metric`] and
/// [`zenpsy_api::PrepareReference`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ssimulacra2Metric {
    impl_type: SimdImpl,
    strip: Option<StripConfig>,
}

impl Ssimulacra2Metric {
    /// Default SIMD SSIMULACRA2 scorer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            impl_type: SimdImpl::Simd,
            strip: None,
        }
    }

    /// Reference-order scalar scorer.
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            impl_type: SimdImpl::Scalar,
            strip: None,
        }
    }

    /// Select SIMD or scalar kernel implementation.
    #[must_use]
    pub const fn with_impl_type(mut self, impl_type: SimdImpl) -> Self {
        self.impl_type = impl_type;
        self
    }

    /// Enable bounded-memory strip evaluation for cold score-only comparisons.
    ///
    /// Note: strip mode uses strip-local downsampling and is refused by
    /// [`PrepareReference`] and map requests to preserve exact grid parity.
    #[must_use]
    pub const fn with_strip(mut self, strip: StripConfig) -> Self {
        self.strip = Some(strip);
        self
    }

    /// Configured kernel implementation.
    #[must_use]
    pub const fn impl_type(&self) -> SimdImpl {
        self.impl_type
    }

    /// Configured strip settings, if any.
    #[must_use]
    pub const fn strip(&self) -> Option<StripConfig> {
        self.strip
    }

    fn config<'a>(&self, stop: &'a dyn enough::Stop, for_maps: bool) -> Ssimulacra2Config<'a> {
        let mut cfg = Ssimulacra2Config::new(self.impl_type).with_stop(stop);
        if !for_maps {
            cfg.strip = self.strip;
        }
        cfg
    }
}

#[derive(Clone, Debug)]
struct NativeMapPlane {
    spec: ScoreSpec,
    extra_id: u16,
    scale_idx: usize,
    channel: usize,
    term: usize,
}

/// Native `zenpsy-api` assessment returned by [`Ssimulacra2Metric`].
#[derive(Clone, Debug)]
pub struct Ssimulacra2Evaluation {
    score: f64,
    maps: Option<Ssimulacra2Maps>,
    fused_diffmap: Option<(Vec<f32>, u32, u32)>,
    native_entries: Vec<NativeMapPlane>,
}

impl Ssimulacra2Evaluation {
    /// Headline SSIMULACRA2 score (`100.0` = identical, higher = better).
    #[must_use]
    pub const fn score(&self) -> f64 {
        self.score
    }

    /// Native per-scale spatial terms, present only when maps were requested.
    #[must_use]
    pub fn spatial_maps(&self) -> Option<&Ssimulacra2Maps> {
        self.maps.as_ref()
    }

    /// Fused full-resolution error map, present only when `"diffmap"` was requested.
    #[must_use]
    pub fn diffmap(&self) -> Option<&[f32]> {
        self.fused_diffmap.as_ref().map(|(v, _, _)| v.as_slice())
    }

    fn score_only(score: f64) -> Self {
        Self {
            score,
            maps: None,
            fused_diffmap: None,
            native_entries: Vec::new(),
        }
    }

    fn from_maps(
        maps: Ssimulacra2Maps,
        req: RequestedMaps,
        latch: &LatchingStop<'_>,
    ) -> MetricResult<Self> {
        let score = maps.score;
        match req {
            RequestedMaps::None => Ok(Self::score_only(score)),
            RequestedMaps::Diffmap => {
                let w = u32::try_from(maps.width).map_err(|_| {
                    whereat::at(
                        MetricError::new(ErrorKind::Resource)
                            .with_metric("ssimulacra2")
                            .with_message("image width exceeds u32"),
                    )
                })?;
                let h = u32::try_from(maps.height).map_err(|_| {
                    whereat::at(
                        MetricError::new(ErrorKind::Resource)
                            .with_metric("ssimulacra2")
                            .with_message("image height exceeds u32"),
                    )
                })?;
                let fused = maps
                    .diffmap_stoppable(latch)
                    .map_err(|r| whereat::at(MetricError::stopped(r).with_metric("ssimulacra2")))?;
                Ok(Self {
                    score,
                    maps: Some(maps),
                    fused_diffmap: Some((fused, w, h)),
                    native_entries: Vec::new(),
                })
            }
            RequestedMaps::Native => {
                let ch_names = ["x", "y", "b"];
                let term_names = ["ssim_error", "artifact", "detail_lost"];
                let mut entries = Vec::with_capacity(maps.scales.len() * 9);
                let mut extra_id = 0u16;
                for (s, _) in maps.scales.iter().enumerate() {
                    for (c, ch) in ch_names.iter().enumerate() {
                        for (t, term) in term_names.iter().enumerate() {
                            let key = format!("s{s}_{ch}_{term}");
                            entries.push(NativeMapPlane {
                                spec: ScoreSpec::new("", Direction::Lower)
                                    .with_perfect(0.0)
                                    .with_key(key),
                                extra_id,
                                scale_idx: s,
                                channel: c,
                                term: t,
                            });
                            extra_id = extra_id.saturating_add(1);
                        }
                    }
                }
                Ok(Self {
                    score,
                    maps: Some(maps),
                    fused_diffmap: None,
                    native_entries: entries,
                })
            }
        }
    }
}

impl Evaluation for Ssimulacra2Evaluation {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        core::iter::once(Score::new(&SSIM2_SCORE_SPEC, self.score))
    }

    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        let fused = self.fused_diffmap.iter().filter_map(|(data, w, h)| {
            let plane = PlaneView::from_samples_strided(
                data,
                *w,
                *h,
                SampleStride(*w as usize),
                PlaneFormat::new(PlaneRole::Extra(0), SampleType::F32),
            )
            .ok()?;
            let grid = MapGrid::new(*w, *h).ok()?;
            Some(MapView::new(&SSIM2_DIFFMAP_SPEC, plane, grid))
        });

        let native = self.native_entries.iter().filter_map(|entry| {
            let maps = self.maps.as_ref()?;
            let scale = maps.scales.get(entry.scale_idx)?;
            let slice = match entry.term {
                0 => &scale.ssim[entry.channel],
                1 => &scale.artifact[entry.channel],
                _ => &scale.detail_lost[entry.channel],
            };
            let sw = u32::try_from(scale.width).ok()?;
            let sh = u32::try_from(scale.height).ok()?;
            let iw = u32::try_from(maps.width).ok()?;
            let ih = u32::try_from(maps.height).ok()?;
            let step = 1u32.checked_shl(entry.scale_idx as u32)?;
            let plane = PlaneView::from_samples_strided(
                slice,
                sw,
                sh,
                SampleStride(scale.width),
                PlaneFormat::new(PlaneRole::Extra(entry.extra_id), SampleType::F32),
            )
            .ok()?;
            let grid = MapGrid::new(iw, ih)
                .ok()?
                .with_cells(0, 0, step, step)
                .ok()?;
            Some(MapView::new(&entry.spec, plane, grid))
        });

        fused.chain(native)
    }
}

/// Owned precomputed SSIMULACRA2 reference for repeated `zenpsy-api` scoring.
#[derive(Clone, Debug)]
pub struct PreparedSsimulacra2 {
    reference: Ssimulacra2Reference,
    impl_type: SimdImpl,
}

impl WarmReference for PreparedSsimulacra2 {
    type Output = Ssimulacra2Evaluation;

    fn evaluate(
        &mut self,
        candidate: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let req = validate_map_request(opts.map_request(), false)?;
        let latch = LatchingStop::new(opts.stop());
        check_stop(&latch)?;
        validate_view_ingress(candidate, opts.max_pixels(), &latch)?;
        let cand_hdr = is_native_hdr(candidate);
        if cand_hdr != self.reference.native_hdr {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("ssimulacra2")
                    .with_message("candidate SDR/HDR domain does not match prepared reference"),
            );
        }

        let metric = Ssimulacra2Metric {
            impl_type: self.impl_type,
            strip: None,
        };
        let out = match req {
            RequestedMaps::None => {
                let cfg = metric.config(&latch, false);
                let score = self
                    .reference
                    .compare_with_config(candidate, &cfg)
                    .map_err(|e| map_ssim2_error(e, &latch))?;
                Ssimulacra2Evaluation::score_only(score)
            }
            RequestedMaps::Native | RequestedMaps::Diffmap => {
                let cfg = metric.config(&latch, true);
                let maps = self
                    .reference
                    .compare_with_maps_with_config(candidate, &cfg)
                    .map_err(|e| map_ssim2_error(e, &latch))?;
                Ssimulacra2Evaluation::from_maps(maps, req, &latch)?
            }
        };
        check_stop(&latch)?;
        Ok(out)
    }
}

impl Metric for Ssimulacra2Metric {
    type Output = Ssimulacra2Evaluation;

    fn id(&self) -> MetricId {
        let mut id = MetricId::new("ssimulacra2", zenpsy_api::implementation!(1))
            .with_reference("libjxl ssimulacra2")
            .with_config("impl", self.impl_type.name());
        if let Some(strip) = self.strip {
            id = id
                .with_config("strip_height", strip.strip_height)
                .with_config("strip_halo", strip.halo_rows);
        }
        id
    }

    fn evaluate(
        &mut self,
        reference: &PixelView<'_>,
        candidate: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let req = validate_map_request(opts.map_request(), self.strip.is_some())?;
        let latch = LatchingStop::new(opts.stop());
        check_stop(&latch)?;
        validate_view_ingress(reference, opts.max_pixels(), &latch)?;
        validate_view_ingress(candidate, opts.max_pixels(), &latch)?;
        if is_native_hdr(reference) != is_native_hdr(candidate) {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("ssimulacra2")
                    .with_message("reference and candidate disagree on SDR vs native HDR domain"),
            );
        }

        let out = match req {
            RequestedMaps::None => {
                let cfg = self.config(&latch, false);
                let score = compute_ssimulacra2_with_config(reference, candidate, &cfg)
                    .map_err(|e| map_ssim2_error(e, &latch))?;
                Ssimulacra2Evaluation::score_only(score)
            }
            RequestedMaps::Native | RequestedMaps::Diffmap => {
                let cfg = self.config(&latch, true);
                let maps = compute_ssimulacra2_maps_with_config(reference, candidate, &cfg)
                    .map_err(|e| map_ssim2_error(e, &latch))?;
                Ssimulacra2Evaluation::from_maps(maps, req, &latch)?
            }
        };
        check_stop(&latch)?;
        Ok(out)
    }
}

impl PrepareReference for Ssimulacra2Metric {
    type Prepared<'a> = PreparedSsimulacra2;

    fn warm_reference(
        &mut self,
        reference: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        if self.strip.is_some() {
            return fail(
                MetricError::new(ErrorKind::InvalidConfig)
                    .with_metric("ssimulacra2")
                    .with_message(
                        "strip mode uses strip-local downsampling and cannot prepare a full-image warm reference with bit-exact parity",
                    ),
            );
        }
        let _ = validate_map_request(opts.map_request(), false)?;
        let latch = LatchingStop::new(opts.stop());
        check_stop(&latch)?;
        validate_view_ingress(reference, opts.max_pixels(), &latch)?;

        let cfg = Ssimulacra2Config::new(self.impl_type).with_stop(&latch);
        let prepared = Ssimulacra2Reference::new_with_config(reference, &cfg)
            .map_err(|e| map_ssim2_error(e, &latch))?;
        check_stop(&latch)?;
        Ok(PreparedSsimulacra2 {
            reference: prepared,
            impl_type: self.impl_type,
        })
    }
}

#[track_caller]
fn validate_map_request(req: MapRequest<'_>, strip_enabled: bool) -> MetricResult<RequestedMaps> {
    match req {
        MapRequest::None => Ok(RequestedMaps::None),
        MapRequest::Native => {
            if strip_enabled {
                return fail(
                    MetricError::new(ErrorKind::InvalidConfig)
                        .with_metric("ssimulacra2")
                        .with_message("spatial maps require whole-image evaluation (strip mode is unsupported)"),
                );
            }
            Ok(RequestedMaps::Native)
        }
        MapRequest::Named("diffmap") => {
            if strip_enabled {
                return fail(
                    MetricError::new(ErrorKind::InvalidConfig)
                        .with_metric("ssimulacra2")
                        .with_message(
                            "diffmap requires whole-image evaluation (strip mode is unsupported)",
                        ),
                );
            }
            Ok(RequestedMaps::Diffmap)
        }
        MapRequest::Named(_) => fail(
            MetricError::new(ErrorKind::NoMap)
                .with_metric("ssimulacra2")
                .with_message(
                    "fast-ssim2 only publishes named 'diffmap' or native per-scale planes",
                ),
        ),
        _ => fail(
            MetricError::new(ErrorKind::UnsupportedRequest)
                .with_metric("ssimulacra2")
                .with_message("unsupported map request"),
        ),
    }
}

#[track_caller]
fn check_stop(latch: &LatchingStop<'_>) -> MetricResult<()> {
    latch
        .check()
        .map_err(|r| whereat::at(MetricError::stopped(r).with_metric("ssimulacra2")))
}

#[track_caller]
fn validate_hlg_rendering(transfer: Transfer, white: Option<DiffuseWhite>) -> MetricResult<()> {
    let Transfer::Hlg(mode) = transfer else {
        return Ok(());
    };
    match mode {
        HlgMode::Scene => Ok(()),
        HlgMode::Display(rendering) => {
            if white.is_some() {
                return fail(
                    MetricError::new(ErrorKind::UnsupportedInput)
                        .with_metric("ssimulacra2")
                        .with_message(
                            "DiffuseWhite cannot override explicit HlgMode::Display peak luminance",
                        ),
                );
            }
            if rendering.black_nits() != 0.0 {
                return fail(
                    MetricError::new(ErrorKind::UnsupportedInput)
                        .with_metric("ssimulacra2")
                        .with_message(
                            "non-zero HLG display black_nits is not supported by fast-ssim2 native HDR",
                        ),
                );
            }
            let expected_gamma =
                crate::source::native_hdr::hlg_reference_gamma(rendering.peak().nits());
            if rendering.system_gamma() != expected_gamma {
                return fail(
                    MetricError::new(ErrorKind::UnsupportedInput)
                        .with_metric("ssimulacra2")
                        .with_message(
                            "explicit HLG system_gamma does not match fast-ssim2's BT.2100 reference OOTF",
                        ),
                );
            }
            Ok(())
        }
        _ => fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("ssimulacra2")
                .with_message("unsupported HLG display mode"),
        ),
    }
}

#[track_caller]
fn validate_f32_samples_stoppable(
    view: &PixelView<'_>,
    latch: &LatchingStop<'_>,
) -> MetricResult<()> {
    let fmt = view.format();
    if fmt.layout.sample != SampleType::F32 {
        return Ok(());
    }
    let alpha_active = matches!(fmt.color.alpha(), AlphaMode::Straight);
    let (channels, bpp) = match fmt.layout.order {
        ChannelOrder::Rgb => (3usize, 12usize),
        ChannelOrder::Rgba => (if alpha_active { 4 } else { 3 }, 16),
        ChannelOrder::Gray => (1, 4),
        ChannelOrder::GrayAlpha => (if alpha_active { 2 } else { 1 }, 8),
        _ => return Ok(()),
    };
    let w = view.width() as usize;
    let h = view.height() as usize;
    for y in 0..h {
        if y & 63 == 0 {
            check_stop(latch)?;
        }
        let row = view.row(y as u32);
        for x in 0..w {
            let off = x * bpp;
            for c in 0..channels {
                let o = off + c * 4;
                let v = f32::from_ne_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]]);
                if !v.is_finite() {
                    return fail(
                        MetricError::new(ErrorKind::InvalidInput)
                            .with_metric("ssimulacra2")
                            .with_message("non-finite f32 sample in active pixel data"),
                    );
                }
            }
        }
    }
    check_stop(latch)
}

#[track_caller]
fn validate_view_ingress(
    view: &PixelView<'_>,
    max_pixels: Option<u64>,
    latch: &LatchingStop<'_>,
) -> MetricResult<()> {
    check_stop(latch)?;
    if view.orientation() != Orientation::Identity {
        return fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("ssimulacra2")
                .with_message(
                    "pending EXIF orientation is not supported — apply orientation before scoring",
                ),
        );
    }
    if view.format().color.alpha() == AlphaMode::Premultiplied {
        return fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("ssimulacra2")
                .with_message(
                    "premultiplied alpha is not supported on zenpsy-api ingress; unpremultiply explicitly before scoring",
                ),
        );
    }
    match view.format().color {
        ColorSpace::Rgb {
            transfer, white, ..
        }
        | ColorSpace::Gray {
            transfer, white, ..
        } => validate_hlg_rendering(transfer, white)?,
        _ => {}
    }
    if let Some(limit) = max_pixels {
        let w = u64::from(view.width());
        let h = u64::from(view.height());
        let orig_px = w.checked_mul(h).ok_or_else(|| {
            whereat::at(
                MetricError::new(ErrorKind::Resource)
                    .with_metric("ssimulacra2")
                    .with_message("pixel count overflow"),
            )
        })?;
        if orig_px > limit {
            return fail(
                MetricError::new(ErrorKind::Resource)
                    .with_metric("ssimulacra2")
                    .with_message(format!(
                        "image has {orig_px} active pixels ({w}x{h}), exceeding max_pixels={limit}"
                    )),
            );
        }
    }
    validate_f32_samples_stoppable(view, latch)?;
    Ok(())
}

fn is_native_hdr(view: &PixelView<'_>) -> bool {
    match view.format().color {
        ColorSpace::Rgb {
            primaries,
            transfer,
            white,
            ..
        } => {
            matches!(transfer, Transfer::Pq | Transfer::Hlg(_))
                || (transfer == Transfer::Linear
                    && (primaries != Primaries::Bt709 || white.is_some()))
        }
        ColorSpace::Gray {
            transfer, white, ..
        } => {
            matches!(transfer, Transfer::Pq | Transfer::Hlg(_))
                || (transfer == Transfer::Linear && white.is_some())
        }
        _ => false,
    }
}

#[track_caller]
fn map_ssim2_error(err: Ssimulacra2Error, latch: &LatchingStop<'_>) -> whereat::At<MetricError> {
    if let Some(reason) = latch.reason() {
        return whereat::at(MetricError::stopped(reason).with_metric("ssimulacra2"));
    }
    let kind = match err {
        Ssimulacra2Error::InvalidConfiguration(_) => ErrorKind::InvalidConfig,
        Ssimulacra2Error::UnsupportedInput(_) => ErrorKind::UnsupportedInput,
        Ssimulacra2Error::NonMatchingImageDimensions => ErrorKind::DimensionMismatch,
        Ssimulacra2Error::InvalidImageSize => ErrorKind::TooSmall,
        Ssimulacra2Error::ImageTooLarge { .. } => ErrorKind::Resource,
        Ssimulacra2Error::GaussianBlurError => ErrorKind::Internal,
        Ssimulacra2Error::Cancelled(reason) => ErrorKind::Stopped(reason),
    };
    whereat::at(MetricError::from_detail(kind, err).with_metric("ssimulacra2"))
}
