# fast-ssim2 dev commands

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt -p fast-ssim2 -p fast-ssim2-cli
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Full local test gate
test:
    cargo test --all-targets
    cargo test --doc

# Input-layout regression tests, including HDR conversion.
test-input:
    cargo test -p fast-ssim2 --lib --features hdr-pu source::tests

# Library regression gate (all library features, without CLI video dependencies).
test-lib:
    cargo test -p fast-ssim2 --lib --all-features

# Public contract regressions across all library modes.
test-api:
    cargo test -p fast-ssim2 --test api_contracts --all-features

# Lint the complete library surface, including development tooling.
clippy-lib:
    cargo clippy -p fast-ssim2 --all-targets --all-features -- -D warnings

# Test every library feature and executable documentation.
check-library:
    cargo test -p fast-ssim2 --all-features --lib --tests
    cargo test -p fast-ssim2 --all-features --doc
    cargo test -p fast-ssim2 --no-default-features --lib --tests
    cargo test -p fast-ssim2 --no-default-features --doc

# Strict API docs and the default CLI gate.
check-doc-cli:
    RUSTDOCFLAGS="-D warnings" cargo doc -p fast-ssim2 --all-features --no-deps
    cargo test -p fast-ssim2-cli
    cargo clippy -p fast-ssim2-cli --all-targets -- -D warnings

# README.md is canonical; the crate-local copy is included in rustdoc and packages.
docs-sync:
    cp README.md README.crates.md
    cp README.md fast-ssim2/README.md

docs-check:
    cmp README.md README.crates.md
    cmp README.md fast-ssim2/README.md

# Compile and execute the guide's examples with the library's actual dependencies.
check-native-hdr-guide:
    python3 tools/check_native_hdr_guide.py

# Check feature warnings, caller diagnostics, deny(deprecated), and explicit compatibility opt-in.
check-deprecations:
    python3 tools/check_deprecations.py

# Workspace lint gate with every library feature (CLI video requires system libraries).
clippy-workspace:
    cargo clippy --workspace --all-targets --features fast-ssim2/unstable-internals,fast-ssim2/hdr-pu,fast-ssim2/imgref,fast-ssim2/rayon -- -D warnings

check-compare:
    cargo check --manifest-path compare_tool/Cargo.toml

# Interleaved RGB8 timings against published 0.7.1 and 0.8.2.
version-audit:
    cargo run --release --manifest-path tools/version-audit/Cargo.toml

# Interleaved pre-HDR/current and initial-HDR/current comparisons in memory.
native-hdr-performance *args:
    AUDIT_NATIVE_HDR=1 cargo run --release --manifest-path tools/version-audit/Cargo.toml -- {{args}}

clippy-version-audit:
    cargo clippy --manifest-path tools/version-audit/Cargo.toml -- -D warnings

build-version-audit: clippy-version-audit
    cargo build --release --manifest-path tools/version-audit/Cargo.toml

# Invoke only the current kernel for a profiler; no timing instrumentation.
profile-native-hdr kind output:
    AUDIT_NATIVE_HDR=1 AUDIT_PROFILE={{kind}} perf record -F 499 -g -o {{output}} -- tools/version-audit/target/release/ssim2-version-audit

native-hdr-performance-summary input output commit binary host:
    python3 tools/native_hdr_performance_report.py {{input}} {{output}} --source-commit {{commit}} --binary-sha256 {{binary}} --host-label {{host}}

# Inventory breaking changes relative to the latest crates.io release.
semver-audit:
    cargo semver-checks check-release -p fast-ssim2 --baseline-version 0.8.2 --release-type patch

# Reproduce SIMD128 conformance failures without relaxing assertions.
test-wasm:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=/" RUSTFLAGS="-C target-feature=+simd128 -Dwarnings" cargo test --target wasm32-wasip1 -p fast-ssim2 --features imgref --lib -- --nocapture

# Conformance for targets without hardware FMA.
test-i686:
    cross test -p fast-ssim2 --features imgref --target i686-unknown-linux-gnu --lib

# Compare published scores with hashed, captured C++ RGB8 fixtures.
version-parity:
    AUDIT_PARITY=1 cargo run --release --manifest-path tools/version-audit/Cargo.toml

# Package and verify library contents without uploading.
package-check:
    cargo publish -p fast-ssim2 --dry-run

# Permit the audited breaking transition from the published 0.8.2 baseline.
# Rust 1.98.1 emits the rustdoc JSON format supported by semver-checks 0.49.
semver-major-audit:
    RUSTUP_TOOLCHAIN=1.98.1 cargo semver-checks check-release -p fast-ssim2 --baseline-version 0.8.2 --release-type major

# Test-only Cloudinary 2.1 frontend oracle; source and build are outside this repo.
# Run this recipe through run-heavy, as for other compilation commands.
build-native-hdr-oracle source build output:
    bash tools/build-native-hdr-oracle.sh "{{source}}" "{{build}}" "{{output}}"

capture-native-hdr oracle output:
    python3 tools/native_hdr_fixtures.py {{oracle}} {{output}}

# Full native-HDR score parity; optional capture must name a new file.
native-hdr-parity oracle *capture:
    cargo run -p fast-ssim2 --example native_hdr_parity -- {{oracle}} {{capture}}

capture-native-hdr-gray oracle output:
    python3 tools/native_hdr_fixtures.py {{oracle}} {{output}} --gray

# Opt-in independent oracle: every u16 code, all native transfers and gamuts.
native-hdr-dense oracle:
    SSIM2_HDR_ORACLE={{oracle}} cargo test -p fast-ssim2 --lib native_hdr_dense_frontend_matches_cpp -- --ignored --nocapture

test-native-hdr:
    cargo test -p fast-ssim2 --all-features --lib native_hdr_
    cargo test -p fast-ssim2 --all-features --test native_hdr

# Native HDR frontend and integration conformance under WASM SIMD128.
test-native-hdr-wasm:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=/" RUSTFLAGS="-C target-feature=+simd128 -Dwarnings" cargo test --target wasm32-wasip1 -p fast-ssim2 --features imgref --test native_hdr -- --nocapture

# Isolate platform libm behavior in the score's final nonlinear transform.
native-hdr-transform bits:
    cargo run -p fast-ssim2 --features unstable-internals --example native_hdr_parity -- unused --transform {{bits}}

native-hdr-transform-wasm bits:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=/" RUSTFLAGS="-C target-feature=+simd128 -Dwarnings" cargo run --target wasm32-wasip1 -p fast-ssim2 --features unstable-internals --example native_hdr_parity -- unused --transform {{bits}}

# Keep cross-container scratch on disk and cap its memory/CPU separately.
test-native-hdr-i686:
    mkdir -p "$HOME/tmp/fast-ssim2-native-hdr/container-scratch"
    CROSS_CONTAINER_OPTS="--network=host --env TMPDIR=/scratch --volume $HOME/tmp/fast-ssim2-native-hdr/container-scratch:/scratch --memory=6g --memory-swap=6g --cpus=4 --cpu-shares=256" cross test -p fast-ssim2 --features imgref --target i686-unknown-linux-gnu --lib --test native_hdr

capture-native-hdr-signed oracle output:
    python3 tools/native_hdr_fixtures.py {{oracle}} {{output}} --signed-zero

test-native-hdr-frontend-wasm:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir=/" RUSTFLAGS="-C target-feature=+simd128 -Dwarnings" cargo test --target wasm32-wasip1 -p fast-ssim2 --features imgref --lib native_hdr_frontend_matches_cpp

test-native-hdr-frontend-i686:
    mkdir -p "$HOME/tmp/fast-ssim2-native-hdr/container-scratch"
    CROSS_CONTAINER_OPTS="--network=host --env TMPDIR=/scratch --volume $HOME/tmp/fast-ssim2-native-hdr/container-scratch:/scratch --memory=6g --memory-swap=6g --cpus=4 --cpu-shares=256" cross test -p fast-ssim2 --features imgref --target i686-unknown-linux-gnu --lib native_hdr_frontend_matches_cpp

test-native-hdr-frontend:
    cargo test -p fast-ssim2 --lib native_hdr_frontend_matches_cpp

test-input-alpha:
    cargo test -p fast-ssim2 --all-features --test native_hdr alpha_metadata_without_a_stored_channel_is_rejected

capture-native-hdr-low-peak oracle rgb_output gray_output:
    python3 tools/native_hdr_fixtures.py {{oracle}} {{rgb_output}} --low-peak
    python3 tools/native_hdr_fixtures.py {{oracle}} {{gray_output}} --low-peak --gray
