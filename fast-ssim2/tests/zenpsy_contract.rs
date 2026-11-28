#![cfg(feature = "zenpsy-api")]
//! End-to-end contract tests for `fast-ssim2`'s `zenpsy-api` integration.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fast_ssim2::{Ssimulacra2Metric, StripConfig, compute_ssimulacra2};
use zenpixels::{
    AlphaMode, ColorSpace, DiffuseWhite, HlgMode, HlgRendering, HlgRenderingParams, Nits,
    Orientation, PixelFormat, PixelView, PlaneRole, Primaries, SampleLayout, SignalRange, Transfer,
    formats,
};
use zenpsy_api::how_far::{Stop, StopOnly, StopReason};
use zenpsy_api::{Direction, ErrorKind, EvalOpts, Evaluation, Metric, prepare_reference, score};

fn synth_rgb(w: u32, h: u32, distort: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 23 + y * 17 + 11) % 256) as u8;
            let g = ((x * 11 + y * 31 + 41) % 256) as u8;
            let b = ((x ^ y).wrapping_mul(13) % 256) as u8;
            if distort {
                out.push(r.saturating_add(5));
                out.push(g.saturating_sub(3));
                out.push(b.saturating_add(4));
            } else {
                out.push(r);
                out.push(g);
                out.push(b);
            }
        }
    }
    out
}

#[derive(Clone)]
struct CountingStop {
    checks: Arc<AtomicUsize>,
    trip_after: usize,
}

impl CountingStop {
    fn new(trip_after: usize) -> Self {
        Self {
            checks: Arc::new(AtomicUsize::new(0)),
            trip_after,
        }
    }

    fn checks(&self) -> usize {
        self.checks.load(Ordering::SeqCst)
    }
}

impl Stop for CountingStop {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.checks.fetch_add(1, Ordering::SeqCst);
        if n >= self.trip_after {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone)]
struct OneShotTimeoutStop {
    checks: Arc<AtomicUsize>,
    trip_at: usize,
}

impl OneShotTimeoutStop {
    fn new(trip_at: usize) -> Self {
        Self {
            checks: Arc::new(AtomicUsize::new(0)),
            trip_at,
        }
    }
}

impl Stop for OneShotTimeoutStop {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.checks.fetch_add(1, Ordering::SeqCst);
        if n == self.trip_at {
            Err(StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}

#[test]
fn score_only_warm_and_cold_match_native_bit_exact() {
    let mut metric = Ssimulacra2Metric::new();
    let id = metric.id();
    assert_eq!(id.family(), "ssimulacra2");
    assert_eq!(id.implementation().crate_name(), "fast-ssim2");

    let (w, h) = (128u32, 128u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);
    let src = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    let native = compute_ssimulacra2(&src, &dst).unwrap();
    let cold = score(&mut metric, &src, &dst, EvalOpts::new()).unwrap();
    assert!(cold.spatial_maps().is_none());
    assert!(cold.diffmap().is_none());
    assert_eq!(cold.maps().count(), 0);
    assert_eq!(cold.score().to_bits(), native.to_bits());

    let scores: Vec<_> = cold.scores().collect();
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[0].spec.key(), "ssimulacra2");
    assert_eq!(scores[0].spec.direction(), Direction::Higher);
    assert_eq!(scores[0].spec.perfect(), Some(100.0));

    // Owned prepared reference is `'static` and bit-exact with cold score on row-varying data.
    fn assert_static<T: 'static>(_: &T) {}
    let mut prepared = prepare_reference(&mut metric, &src, EvalOpts::new()).unwrap();
    assert_static(&prepared);
    let warm = prepared.score(&dst, EvalOpts::new()).unwrap();
    assert_eq!(warm.score().to_bits(), cold.score().to_bits());
    assert_eq!(warm.score().to_bits(), native.to_bits());
    assert!(warm.spatial_maps().is_none());

    // Opt-in named "diffmap" and native per-scale planes use auxiliary PlaneRole::Extra(_).
    let warm_dm = prepared
        .score(&dst, EvalOpts::new().with_map("diffmap"))
        .unwrap();
    assert_eq!(warm_dm.score().to_bits(), native.to_bits());
    let dm_maps: Vec<_> = warm_dm.maps().collect();
    assert_eq!(dm_maps.len(), 1);
    assert_eq!(dm_maps[0].spec.key(), "diffmap");
    assert_eq!(dm_maps[0].values.format().role, PlaneRole::Extra(0));
    assert_eq!(dm_maps[0].values.width(), w);
    assert_eq!(dm_maps[0].values.height(), h);

    let warm_native = prepared
        .score(&dst, EvalOpts::new().with_native_map())
        .unwrap();
    assert_eq!(warm_native.score().to_bits(), native.to_bits());
    let native_maps: Vec<_> = warm_native.maps().collect();
    assert!(native_maps.len() >= 9);
    for (idx, m) in native_maps.iter().enumerate() {
        assert_eq!(m.values.format().role, PlaneRole::Extra(idx as u16));
    }

    // Unknown map name fails with NoMap before computation.
    let err = prepared
        .score(&dst, EvalOpts::new().with_map("ssim"))
        .unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::NoMap);
}

#[test]
fn strip_preparation_is_refused_before_cache_construction() {
    let mut strip_metric =
        Ssimulacra2Metric::new().with_strip(StripConfig::new(32).with_halo_rows(1));
    let (w, h) = (64u32, 96u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);
    let src = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    // Cold strip scoring works cleanly.
    let cold_strip = score(&mut strip_metric, &src, &dst, EvalOpts::new()).unwrap();
    assert!(cold_strip.score().is_finite());

    // Warm prepare_reference refuses strip mode explicitly with InvalidConfig.
    let err = prepare_reference(&mut strip_metric, &src, EvalOpts::new())
        .err()
        .expect("strip mode prepare_reference must be refused");
    assert_eq!(err.error().kind(), ErrorKind::InvalidConfig);
}

#[test]
fn rejects_invalid_floats_premultiplied_alpha_and_unsupported_hlg_rendering() {
    let mut metric = Ssimulacra2Metric::new();
    let (w, h) = (16u32, 16u32);
    let rgb = synth_rgb(w, h, false);
    let sdr = PixelView::new(&rgb, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    // Pending orientation rejected.
    let rotated = sdr.clone().with_orientation(Orientation::Rotate90);
    let err = score(&mut metric, &rotated, &rotated, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // Mixed SDR vs native HDR rejected on cold & warm.
    let hdr_bytes = vec![0u8; (w * h * 6) as usize];
    let hdr_fmt = PixelFormat::new(
        SampleLayout::RGB16,
        ColorSpace::Rgb {
            primaries: Primaries::Bt2020,
            transfer: Transfer::Pq,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let hdr = PixelView::new(&hdr_bytes, w, h, w as usize * 6, &hdr_fmt).unwrap();
    let err = score(&mut metric, &sdr, &hdr, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);
    let mut prep = prepare_reference(&mut metric, &sdr, EvalOpts::new()).unwrap();
    let err = prep.score(&hdr, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // NaN and Inf f32 samples rejected on BOTH reference and candidate.
    let valid_f32 = vec![[0.5f32, 0.5, 0.5]; (w * h) as usize];
    let mut nan_f32 = valid_f32.clone();
    nan_f32[5][1] = f32::NAN;
    let valid_v = PixelView::from_typed(&valid_f32, w, h, &formats::RGBF32_LINEAR).unwrap();
    let nan_v = PixelView::from_typed(&nan_f32, w, h, &formats::RGBF32_LINEAR).unwrap();
    assert_eq!(
        score(&mut metric, &nan_v, &valid_v, EvalOpts::new())
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        score(&mut metric, &valid_v, &nan_v, EvalOpts::new())
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::InvalidInput
    );
    let mut prep_f32 = prepare_reference(&mut metric, &valid_v, EvalOpts::new()).unwrap();
    assert_eq!(
        prep_f32
            .score(&nan_v, EvalOpts::new())
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::InvalidInput
    );

    // Premultiplied f32 (including subnormal alpha [0, 0, 0, MIN_SUBNORMAL]) refused.
    let premul_px = vec![[0.0f32, 0.0, 0.0, f32::from_bits(1)]; (w * h) as usize];
    let premul_fmt = PixelFormat::new(
        SampleLayout::RGBAF32,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::Premultiplied,
            white: None,
        },
    );
    let premul_v = PixelView::from_typed(&premul_px, w, h, &premul_fmt).unwrap();
    assert_eq!(
        score(&mut metric, &premul_v, &premul_v, EvalOpts::new())
            .unwrap_err()
            .error()
            .kind(),
        ErrorKind::UnsupportedInput
    );

    // Explicit HLG display rendering with non-default gamma (including near-reference 1.20005),
    // non-zero black, or DiffuseWhite conflict refused.
    let custom_gamma = HlgRendering::new(HlgRenderingParams {
        peak: Nits::new_positive(1000.0).unwrap(),
        black_nits: 0.0,
        system_gamma: 1.5,
    })
    .unwrap();
    let near_ref_gamma = HlgRendering::new(HlgRenderingParams {
        peak: Nits::new_positive(1000.0).unwrap(),
        black_nits: 0.0,
        system_gamma: 1.20005,
    })
    .unwrap();
    let custom_black = HlgRendering::new(HlgRenderingParams {
        peak: Nits::new_positive(1000.0).unwrap(),
        black_nits: 0.05,
        system_gamma: 1.2,
    })
    .unwrap();
    for (transfer, white) in [
        (Transfer::Hlg(HlgMode::Display(custom_gamma)), None),
        (Transfer::Hlg(HlgMode::Display(near_ref_gamma)), None),
        (Transfer::Hlg(HlgMode::Display(custom_black)), None),
        (
            Transfer::Hlg(HlgMode::Display(HlgRendering::REFERENCE_1000_NIT)),
            Some(DiffuseWhite::BT2408),
        ),
    ] {
        let fmt = PixelFormat::new(
            SampleLayout::RGB16,
            ColorSpace::Rgb {
                primaries: Primaries::Bt2020,
                transfer,
                range: SignalRange::Full,
                alpha: AlphaMode::None,
                white,
            },
        );
        let hlg_v = PixelView::new(&hdr_bytes, w, h, w as usize * 6, &fmt).unwrap();
        assert_eq!(
            score(&mut metric, &hlg_v, &hlg_v, EvalOpts::new())
                .unwrap_err()
                .error()
                .kind(),
            ErrorKind::UnsupportedInput
        );
        assert_eq!(
            prepare_reference(&mut metric, &hlg_v, EvalOpts::new())
                .err()
                .expect("unsupported HLG rendering must fail prepare_reference")
                .error()
                .kind(),
            ErrorKind::UnsupportedInput
        );
    }
}

#[test]
fn cancellation_across_unpack_and_diffmap_fusion_and_active_max_pixels() {
    let mut metric = Ssimulacra2Metric::new();
    let (w, h) = (128u32, 128u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);
    let src = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    // Count checks in prepare_reference (must include pixel unpack rows).
    let prep_count = CountingStop::new(usize::MAX);
    let prep_pulse = StopOnly::new(prep_count.clone());
    let mut prep =
        prepare_reference(&mut metric, &src, EvalOpts::new().with_pulse(&prep_pulse)).unwrap();
    let total_prep = prep_count.checks();
    assert!(
        total_prep > 4,
        "expected unpack + pyramid checks, saw {total_prep}"
    );

    // Sweep one-shot TimedOut during unpack phase of prepare_reference.
    for trip in [0, 1, 2, total_prep / 2] {
        let pulse = StopOnly::new(OneShotTimeoutStop::new(trip));
        let err = prepare_reference(&mut metric, &src, EvalOpts::new().with_pulse(&pulse))
            .err()
            .expect("one-shot TimedOut during prepare must fail");
        assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    }

    // Count checks for warm score-only vs warm diffmap fusion.
    let score_count = CountingStop::new(usize::MAX);
    let score_pulse = StopOnly::new(score_count.clone());
    let baseline = prep
        .score(&dst, EvalOpts::new().with_pulse(&score_pulse))
        .unwrap()
        .score();
    let total_score = score_count.checks();

    let map_count = CountingStop::new(usize::MAX);
    let map_pulse = StopOnly::new(map_count.clone());
    let map_eval = prep
        .score(
            &dst,
            EvalOpts::new().with_map("diffmap").with_pulse(&map_pulse),
        )
        .unwrap();
    let total_map = map_count.checks();
    assert_eq!(map_eval.score().to_bits(), baseline.to_bits());
    assert!(
        total_map > total_score + 4,
        "diffmap fusion must poll stop ({total_map} vs {total_score})"
    );

    // One-shot TimedOut mid-unpack (trip=2) and mid-diffmap-fusion must both return TimedOut!
    for trip in [2, (total_score + total_map) / 2, total_map - 2] {
        let pulse = StopOnly::new(OneShotTimeoutStop::new(trip));
        let err = prep
            .score(&dst, EvalOpts::new().with_map("diffmap").with_pulse(&pulse))
            .unwrap_err();
        assert_eq!(
            err.error().kind(),
            ErrorKind::Stopped(StopReason::TimedOut),
            "trip={trip}/{total_map}"
        );
    }

    // Clean reuse after cancellation remains bit-exact.
    assert_eq!(
        prep.score(&dst, EvalOpts::new()).unwrap().score().to_bits(),
        baseline.to_bits()
    );

    // EvalOpts::max_pixels counts active input pixels (excluding internal 8x8 reflect scratch).
    let one_px = synth_rgb(1, 1, false);
    let one_view = PixelView::new(&one_px, 1, 1, 3, &formats::RGB8_SRGB).unwrap();
    assert_eq!(
        score(
            &mut metric,
            &one_view,
            &one_view,
            EvalOpts::new().with_max_pixels(1)
        )
        .unwrap()
        .score(),
        100.0
    );
    let mut prep_one =
        prepare_reference(&mut metric, &one_view, EvalOpts::new().with_max_pixels(1)).unwrap();
    assert_eq!(
        prep_one
            .score(&one_view, EvalOpts::new().with_max_pixels(1))
            .unwrap()
            .score(),
        100.0
    );

    let two_px = synth_rgb(2, 1, false);
    let two_view = PixelView::new(&two_px, 2, 1, 6, &formats::RGB8_SRGB).unwrap();
    assert_eq!(
        score(
            &mut metric,
            &two_view,
            &two_view,
            EvalOpts::new().with_max_pixels(1)
        )
        .unwrap_err()
        .error()
        .kind(),
        ErrorKind::Resource
    );
}
