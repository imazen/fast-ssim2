# fast-ssim2 dev commands

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
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
