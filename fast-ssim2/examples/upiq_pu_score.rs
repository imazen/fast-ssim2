//! Score (ref, dist) EXR pairs with the PU21-native SSIMULACRA2 path.
//!
//! Validation harness for the `hdr-pu` feature on the UPIQ HDR subset:
//! reads a TSV (`ref_path<TAB>dist_path`, header row) of absolute-luminance
//! EXRs, scores each pair via `compute_ssimulacra2_pu_nits`, writes
//! `ref_path<TAB>dist_path<TAB>pu_ssim2` to the output TSV.
//!
//!   cargo run --release --features hdr-pu --example upiq_pu_score -- \
//!     /tmp/upiq_pairs.tsv /tmp/pu_native_scores.tsv

use std::collections::HashMap;
use std::io::Write;

use fast_ssim2::{LinearRgbImage, compute_ssimulacra2_pu_nits};

fn load_exr_nits(path: &str) -> Result<LinearRgbImage, String> {
    let img = image::open(path)
        .map_err(|e| format!("{path}: {e}"))?
        .to_rgb32f();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data: Vec<[f32; 3]> = img
        .into_raw()
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    Ok(LinearRgbImage::new(data, w, h))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pairs = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/upiq_pairs.tsv".into());
    let out_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/pu_native_scores.tsv".into());

    let body = std::fs::read_to_string(&pairs).expect("read pairs tsv");
    let mut cache: HashMap<String, LinearRgbImage> = HashMap::new();
    let mut out = String::from("ref_path\tdist_path\tpu_ssim2\n");
    let (mut ok, mut err) = (0usize, 0usize);

    for line in body.lines().skip(1) {
        let mut it = line.split('\t');
        let (Some(rp), Some(dp)) = (it.next(), it.next()) else {
            continue;
        };
        for p in [rp, dp] {
            if !cache.contains_key(p) {
                match load_exr_nits(p) {
                    Ok(img) => {
                        cache.insert(p.to_string(), img);
                    }
                    Err(e) => eprintln!("LOAD FAIL {e}"),
                }
            }
        }
        let (Some(r), Some(d)) = (cache.get(rp), cache.get(dp)) else {
            err += 1;
            continue;
        };
        match compute_ssimulacra2_pu_nits(r.clone(), d.clone()) {
            Ok(s) => {
                out.push_str(&format!("{rp}\t{dp}\t{s}\n"));
                ok += 1;
                if ok % 50 == 0 {
                    eprintln!("scored {ok}…");
                }
            }
            Err(e) => {
                eprintln!("SCORE FAIL {rp}|{dp}: {e:?}");
                err += 1;
            }
        }
    }

    let mut f = std::fs::File::create(&out_path).expect("create out");
    f.write_all(out.as_bytes()).expect("write out");
    eprintln!("done: {ok} scored, {err} errored -> {out_path}");
}
