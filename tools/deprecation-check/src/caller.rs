// Observe warning-level diagnostics under CI's -Dwarnings. The separate
// -Ddeprecated probe must still reject these exact entry points.
#[warn(warnings)]
fn main() {
    let _ = fast_ssim2::compute_ssimulacra2_pu;
    let _ = fast_ssim2::compute_ssimulacra2_pu_with_config;
}
