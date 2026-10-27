#![forbid(unsafe_code)]

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_HDR_PU").is_some() {
        println!(
            "cargo:warning=feature `hdr-pu` is deprecated; native HDR uses the standard \
             SSIMULACRA2 APIs without this feature. PU21 scores are a different metric; \
             preserve separate score baselines when migrating."
        );
    }
}
