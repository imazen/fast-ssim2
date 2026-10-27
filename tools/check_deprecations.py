#!/usr/bin/env python3
"""Check the public PU21 deprecation diagnostics with real downstream crates."""
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tools/deprecation-check/Cargo.toml"
FEATURE_WARNING = "feature `hdr-pu` is deprecated"
ENTRY_POINTS = {"compute_ssimulacra2_pu", "compute_ssimulacra2_pu_with_config"}


def check(binary, feature, lint):
    env = os.environ.copy()
    encoded = env.get("CARGO_ENCODED_RUSTFLAGS")
    flags = encoded.split("\x1f") if encoded else shlex.split(env.get("RUSTFLAGS", ""))
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags + [lint])
    command = ["cargo", "check", "--manifest-path", str(MANIFEST),
               "--bin", binary, "--message-format=json"]
    if feature:
        command.extend(["--features", "hdr-pu"])
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    print(f"{binary}: feature={feature}, lint={lint}, exit={result.returncode}", flush=True)
    print(result.stderr, end="", flush=True)
    diagnostics = []
    for line in result.stdout.splitlines():
        item = json.loads(line)
        if item.get("reason") == "compiler-message":
            message = item["message"]
            if (message.get("code") or {}).get("code") == "deprecated":
                diagnostics.append(message)
                print(message["rendered"], end="", flush=True)
    return result, diagnostics


def main():
    plain, messages = check("feature-only", False, "-Wdeprecated")
    assert plain.returncode == 0 and not messages, plain.stdout
    assert FEATURE_WARNING not in plain.stderr

    enabled, messages = check("feature-only", True, "-Wdeprecated")
    assert enabled.returncode == 0 and not messages, enabled.stdout
    assert FEATURE_WARNING in enabled.stderr, enabled.stderr

    for lint, level, exit_code in [("-Wdeprecated", "warning", 0),
                                  ("-Ddeprecated", "error", 101)]:
        caller, messages = check("caller", True, lint)
        assert caller.returncode == exit_code, caller.stdout
        assert len(messages) == len(ENTRY_POINTS), caller.stdout
        for name in ENTRY_POINTS:
            assert any(f"`fast_ssim2::{name}`" in message["message"]
                       and message["level"] == level for message in messages), caller.stdout

    legacy, messages = check("legacy", True, "-Ddeprecated")
    assert legacy.returncode == 0 and not messages, legacy.stdout
    print("PU21 warning and compatibility contracts pass.")


if __name__ == "__main__":
    main()
