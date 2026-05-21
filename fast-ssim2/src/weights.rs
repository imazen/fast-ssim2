//! SSIMULACRA2 final-score weight table and compile-time skip tables.
//!
//! The 108-element `WEIGHT` vector is the one shipped by the published
//! `ssimulacra2` Rust crate and the libjxl reference implementation.
//! [`SSIM_HAS_WEIGHT`] and [`EDGE_HAS_WEIGHT`] are derived at compile time
//! from `WEIGHT` and let `simd_ops` skip per-channel SSIM and edge-difference
//! work whose final contribution would be multiplied by zero. The result is
//! bit-identical to the unfiltered path — the skipped slots stay at zero,
//! which is exactly what `mul_add(0.0, ssim)` would have produced.
//!
//! This is the lossless variant of Technique 2 in
//! Kanetaka et al., "Fast Implementation of SSIMULACRA2 for Image Quality
//! Assessment", IWAIT 2026 (DOI 10.1117/12.3100969). Their Fast/Faster modes
//! use looser thresholds (`|w| < 1e-3` / `|w| < 1e-2`) which produce different
//! scores; we keep the `|w| == 0` cells only so users see no score change.

/// Final-score weight table.
///
/// Indexed as `((c * NUM_SCALES + scale) * 2 + n) * 3 + m` where
/// `c ∈ {0,1,2}` = `{X, Y, B}` (XYB channel), `scale ∈ 0..6` (octave),
/// `n ∈ {0,1}` = `{L1, L4}` (norm), `m ∈ {0,1,2}` = `{DSSIM, artifact, detailloss}`.
///
/// 56 of the 108 entries are exactly zero. The compile-time skip tables in
/// this module record which `(channel, scale)` pairs need no work at all.
pub(crate) const WEIGHT: [f64; 108] = [
    0.0,
    0.000_737_660_670_740_658_6,
    0.0,
    0.0,
    0.000_779_348_168_286_730_9,
    0.0,
    0.0,
    0.000_437_115_573_010_737_9,
    0.0,
    1.104_172_642_665_734_6,
    0.000_662_848_341_292_71,
    0.000_152_316_327_837_187_52,
    0.0,
    0.001_640_643_745_659_975_4,
    0.0,
    1.842_245_552_053_929_8,
    11.441_172_603_757_666,
    0.0,
    0.000_798_910_943_601_516_3,
    0.000_176_816_438_078_653,
    0.0,
    1.878_759_497_954_638_7,
    10.949_069_906_051_42,
    0.0,
    0.000_728_934_699_150_807_2,
    0.967_793_708_062_683_3,
    0.0,
    0.000_140_034_242_854_358_84,
    0.998_176_697_785_496_7,
    0.000_319_497_559_344_350_53,
    0.000_455_099_211_379_206_3,
    0.0,
    0.0,
    0.001_364_876_616_324_339_8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466_890_328_078_848,
    0.0,
    17.445_833_984_131_262,
    0.000_623_560_163_404_146_6,
    0.0,
    0.0,
    6.683_678_146_179_332,
    0.000_377_244_079_796_112_96,
    1.027_889_937_768_264,
    225.205_153_008_492_74,
    0.0,
    0.0,
    19.213_238_186_143_016,
    0.001_140_152_458_661_836_1,
    0.001_237_755_635_509_985,
    176.393_175_984_506_94,
    0.0,
    0.0,
    24.433_009_998_704_76,
    0.285_208_026_121_177_57,
    0.000_448_543_692_383_340_8,
    0.0,
    0.0,
    0.0,
    34.779_063_444_837_72,
    44.835_625_328_877_896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_868_055_657_329_169_8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_531_319_187_435_874_7,
    0.0,
    0.000_165_338_141_613_791_12,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_417_917_180_325_133_6,
    0.001_729_082_823_472_283_3,
    0.0,
    0.002_082_700_584_663_643_7,
    0.0,
    0.0,
    8.826_982_764_996_862,
    23.192_433_439_989_26,
    0.0,
    95.108_049_881_108_6,
    0.986_397_803_440_068_2,
    0.983_438_279_246_535_3,
    0.001_228_640_504_827_849_3,
    171.266_725_589_730_7,
    0.980_785_887_243_537_9,
    0.0,
    0.0,
    0.0,
    0.000_513_006_458_899_067_9,
    0.0,
    0.000_108_540_578_584_115_37,
];

/// Number of pyramid scales SSIMULACRA2 evaluates.
pub(crate) const NUM_SCALES: usize = 6;

/// Count of pyramid scales that `compute_frame_ssimulacra2_impl` will produce
/// for an input of the given dimensions. Mirrors the break logic of the main
/// loop (downscale by 2 each octave; stop when either side drops below 8).
pub(crate) const fn count_scales(width: usize, height: usize) -> usize {
    let mut w = width;
    let mut h = height;
    let mut n = 0;
    let mut scale = 0;
    while scale < NUM_SCALES {
        if w < 8 || h < 8 {
            break;
        }
        if scale > 0 {
            // `usize::div_ceil` is const-stable in 1.85+.
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        n += 1;
        scale += 1;
    }
    n
}

/// Skip tables indexed by `[scales_n][channel][scale_iter]`.
///
/// `score()` consumes `WEIGHT[i++]` *linearly* in iteration order
/// `(channel, scale_iter, n, map)` — but when the image is small enough
/// that only `scales_n < NUM_SCALES` scales are produced, that linear walk
/// reads WEIGHT *cells* that don't match their layout meaning. For example
/// at `scales_n = 5` the score's `(c=2, scale_iter=0)` slot is multiplied
/// by `WEIGHT[60]` and `WEIGHT[63]`, which by the layout encode
/// `Y-s4-L*-SSIM` (one of them is `34.78`), not `B-s0`'s zeros. Skipping
/// blindly on the layout would silently drop the 34.78 contribution.
///
/// We therefore tabulate the actual WEIGHT cells score will read for every
/// `(scales_n, channel, scale_iter)` triple and return `true` only when
/// every relevant cell is exactly zero. Bit-identical to the unfiltered
/// path for all input sizes.
///
/// First dimension is `NUM_SCALES + 1` so a runtime `scales_n` of 0 (an
/// invalid input that never reaches these tables) still indexes safely.
pub(crate) const SSIM_HAS_WEIGHT: [[[bool; NUM_SCALES]; 3]; NUM_SCALES + 1] =
    compute_ssim_has_weight();

pub(crate) const EDGE_HAS_WEIGHT: [[[bool; NUM_SCALES]; 3]; NUM_SCALES + 1] =
    compute_edge_has_weight();

const fn compute_ssim_has_weight() -> [[[bool; NUM_SCALES]; 3]; NUM_SCALES + 1] {
    let mut out = [[[true; NUM_SCALES]; 3]; NUM_SCALES + 1];
    let mut sn = 1;
    while sn <= NUM_SCALES {
        let mut c = 0;
        while c < 3 {
            let mut s = 0;
            while s < sn {
                // `n in {0,1}` is the L1/L4 norm; `m = 0` is the SSIM slot.
                let i_l1 = ((c * sn + s) * 2) * 3;
                let i_l4 = ((c * sn + s) * 2 + 1) * 3;
                out[sn][c][s] = WEIGHT[i_l1] != 0.0 || WEIGHT[i_l4] != 0.0;
                s += 1;
            }
            // For `s >= sn` the cell is never read; leave `true` so a stray
            // lookup doesn't trigger an incorrect skip.
            c += 1;
        }
        sn += 1;
    }
    out
}

const fn compute_edge_has_weight() -> [[[bool; NUM_SCALES]; 3]; NUM_SCALES + 1] {
    let mut out = [[[true; NUM_SCALES]; 3]; NUM_SCALES + 1];
    let mut sn = 1;
    while sn <= NUM_SCALES {
        let mut c = 0;
        while c < 3 {
            let mut s = 0;
            while s < sn {
                // `m = 1` is artifact, `m = 2` is detail-loss; both L1 and L4.
                let i_a_l1 = ((c * sn + s) * 2) * 3 + 1;
                let i_a_l4 = ((c * sn + s) * 2 + 1) * 3 + 1;
                let i_d_l1 = ((c * sn + s) * 2) * 3 + 2;
                let i_d_l4 = ((c * sn + s) * 2 + 1) * 3 + 2;
                out[sn][c][s] = WEIGHT[i_a_l1] != 0.0
                    || WEIGHT[i_a_l4] != 0.0
                    || WEIGHT[i_d_l1] != 0.0
                    || WEIGHT[i_d_l4] != 0.0;
                s += 1;
            }
            c += 1;
        }
        sn += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_table_has_108_entries() {
        assert_eq!(WEIGHT.len(), 108);
    }

    #[test]
    fn ssim_skip_table_full_scales_matches_paper_audit() {
        // At scales_n = NUM_SCALES = 6 the linear WEIGHT walk lines up with
        // the layout. Per the audit: X-s0, Y-s5, B-s0 should skip.
        let sn = NUM_SCALES;
        assert!(!SSIM_HAS_WEIGHT[sn][0][0], "X-s0 SSIM should skip at full scales");
        assert!(!SSIM_HAS_WEIGHT[sn][1][5], "Y-s5 SSIM should skip at full scales");
        assert!(!SSIM_HAS_WEIGHT[sn][2][0], "B-s0 SSIM should skip at full scales");
        for c in 0..3 {
            for s in 0..NUM_SCALES {
                let expected_zero = matches!((c, s), (0, 0) | (1, 5) | (2, 0));
                assert_eq!(
                    SSIM_HAS_WEIGHT[sn][c][s],
                    !expected_zero,
                    "SSIM_HAS_WEIGHT[6][{c}][{s}] mismatch"
                );
            }
        }
    }

    #[test]
    fn edge_skip_table_full_scales_matches_paper_audit() {
        let sn = NUM_SCALES;
        assert!(!EDGE_HAS_WEIGHT[sn][0][5], "X-s5 edge should skip at full scales");
        assert!(!EDGE_HAS_WEIGHT[sn][1][5], "Y-s5 edge should skip at full scales");
        for c in 0..3 {
            for s in 0..NUM_SCALES {
                let expected_zero = matches!((c, s), (0, 5) | (1, 5));
                assert_eq!(
                    EDGE_HAS_WEIGHT[sn][c][s],
                    !expected_zero,
                    "EDGE_HAS_WEIGHT[6][{c}][{s}] mismatch"
                );
            }
        }
    }

    #[test]
    fn ssim_skip_at_scales_n_5_respects_score_misindex() {
        // At scales_n = 5 (64x64), `score()`'s i=60 reads WEIGHT[60] for
        // (c=2, scale_iter=0). WEIGHT[60] is layout-position
        // Y-s4-L1-SSIM = 0, but WEIGHT[63] (= L4 for the same cell) is
        // 34.78. The skip table must therefore NOT skip (c=2, s=0) at
        // scales_n=5 — that would silently drop the 34.78 contribution.
        assert!(
            SSIM_HAS_WEIGHT[5][2][0],
            "must NOT skip (c=2, s=0) at scales_n=5: WEIGHT[63] = 34.78"
        );
    }

    #[test]
    fn count_scales_matches_main_loop() {
        // Sanity-check the helper against the dimensions used in the test
        // suite. 64x64 truncates to 5 scales; 1920x1080 fits all 6.
        assert_eq!(count_scales(64, 64), 5);
        assert_eq!(count_scales(8, 8), 2);
        assert_eq!(count_scales(1920, 1080), 6);
        assert_eq!(count_scales(320, 240), 6);
    }
}
