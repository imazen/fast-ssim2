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
