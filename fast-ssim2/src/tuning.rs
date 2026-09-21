//! Machine-adaptive scheduling constants.
//!
//! These knobs decide *when* parallel execution is worth its overhead and how
//! work is divided — never *what* is computed. Every parallelisation they
//! govern is bit-identical to its serial form: the vertical-blur band split
//! happens on whole `LANES`-wide column groups whose IIR state was always
//! independent, and `par_min_samples` only chooses between calling a kernel
//! once and splitting it across workers whose summation order is unchanged.
//! Scores cannot depend on them — only speed can.

/// Plane samples below which `rayon` is not engaged for the per-plane
/// kernels — the canonical default for [`Tuning::par_min_samples`].
///
/// `rayon`'s join/steal overhead is a fixed few microseconds; at 320x240 the
/// whole metric takes ~3 ms single-threaded, and splitting every stage of
/// every pyramid level across workers measured **2x slower** than not
/// bothering (2.99 ms -> 6.24 ms). The pyramid also shrinks by 4x per scale,
/// so even a large image reaches sizes where this matters after a few levels.
pub(crate) const PAR_MIN_SAMPLES: usize = 1 << 18;

/// Default `min_groups_per_band` for the current ISA.
///
/// aarch64 is the only target where banding was measured a win: −21% at 4K
/// on an Apple M4 Pro, flat across band counts 2..32. x86_64 regressed on
/// four of five fleet machines (+11–25%), and the sole win (WSL2, −13%) sits
/// on the same µarch as a +25% loss, so no per-µarch default is defensible —
/// it stays off until a machine proves otherwise via config or env. Other
/// targets are unmeasured. See `benchmarks/vertical_band_parallel_2026-09-10.md`
/// and `benchmarks/fleet_4k_2026-09-10.md`.
#[cfg(target_arch = "aarch64")]
const ISA_MIN_GROUPS_PER_BAND: usize = 64;
#[cfg(not(target_arch = "aarch64"))]
const ISA_MIN_GROUPS_PER_BAND: usize = 0;

/// Scheduling tuning: how work is divided across threads.
///
/// `Default` and every constructor that does not take an explicit `Tuning`
/// resolve to [`Tuning::detect`]: a fixed per-ISA table, then `FAST_SSIM2_*`
/// environment overrides. To pin values explicitly, build the struct
/// literally (`Tuning { .. }`) or start from [`Tuning::serial`].
///
/// These values affect speed only, never scores — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tuning {
    /// Minimum 8-column groups per vertical-blur band — **0 disables banding
    /// entirely** (the vertical pass always runs serial).
    ///
    /// The vertical Gaussian is a per-column IIR, so a band of columns is
    /// independent work — but each band walks the full plane height reading a
    /// narrow vertical strip, and how many strided walks a prefetcher
    /// tolerates differs per machine. Each band is at least
    /// `min_groups_per_band * 32` bytes of each row; the count is then capped
    /// by the thread count. 64 groups = 512 columns = 2 KiB/row, giving 7
    /// bands at 4K and 3 at 1080p; 4–8 bands measured −7% to −12% on Zen 3,
    /// so `64`–`120` is the range worth sweeping on x86.
    ///
    /// Environment override: `FAST_SSIM2_VBAND_MIN_GROUPS` (usize).
    pub min_groups_per_band: usize,

    /// Plane samples below which `rayon` is not engaged for the per-plane
    /// kernels (blur horizontal pass, `image_multiply`, `ssim_map`, XYB
    /// conversion). `0` means always parallel; `usize::MAX` means never.
    ///
    /// Environment override: `FAST_SSIM2_PAR_MIN_SAMPLES` (usize).
    pub par_min_samples: usize,
}

impl Tuning {
    /// Adaptive defaults for the current machine: the per-ISA table, then any
    /// `FAST_SSIM2_*` environment overrides. Unparseable env values are
    /// ignored. See [`ISA_MIN_GROUPS_PER_BAND`] for the measurements behind
    /// the table.
    pub fn detect() -> Self {
        let mut t = Self {
            min_groups_per_band: ISA_MIN_GROUPS_PER_BAND,
            par_min_samples: PAR_MIN_SAMPLES,
        };
        if let Some(v) = env_usize("FAST_SSIM2_VBAND_MIN_GROUPS") {
            t.min_groups_per_band = v;
        }
        if let Some(v) = env_usize("FAST_SSIM2_PAR_MIN_SAMPLES") {
            t.par_min_samples = v;
        }
        t
    }

    /// Fully serial tuning: no banding, `rayon` never engaged. Deterministic
    /// across machines — useful for benchmarking and A/B isolation.
    pub const fn serial() -> Self {
        Self {
            min_groups_per_band: 0,
            par_min_samples: usize::MAX,
        }
    }
}

impl Default for Tuning {
    fn default() -> Self {
        Self::detect()
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_disables_everything() {
        let t = Tuning::serial();
        assert_eq!(t.min_groups_per_band, 0);
        assert_eq!(t.par_min_samples, usize::MAX);
    }

    #[test]
    fn detect_is_isa_default_unless_env_overrides() {
        // Read env the same way detect() does, so the expectation holds
        // whether or not the overrides happen to be set in the test env.
        let t = Tuning::detect();
        assert_eq!(
            t.min_groups_per_band,
            env_usize("FAST_SSIM2_VBAND_MIN_GROUPS").unwrap_or(ISA_MIN_GROUPS_PER_BAND)
        );
        assert_eq!(
            t.par_min_samples,
            env_usize("FAST_SSIM2_PAR_MIN_SAMPLES").unwrap_or(PAR_MIN_SAMPLES)
        );
    }
}
