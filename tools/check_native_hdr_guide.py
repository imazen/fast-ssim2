#!/usr/bin/env python3
"""Run the native HDR guide's Rust examples against the current library."""
import json
from pathlib import Path
import subprocess


def main():
    build = subprocess.run(
        ["cargo", "build", "-p", "fast-ssim2", "--all-features", "--message-format=json"],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    libraries = {}
    for line in build.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") != "compiler-artifact":
            continue
        name = message["target"]["name"]
        for filename in message["filenames"]:
            if filename.endswith(".rlib"):
                libraries[name] = Path(filename)
    dependencies = libraries["zenpixels"].parent
    command = ["rustdoc", "--test", "docs/native-hdr.md", "--edition=2024", "-Dwarnings",
               "-L", f"dependency={dependencies}"]
    for name in ["fast_ssim2", "zenpixels", "bytemuck"]:
        command.extend(["--extern", f"{name}={libraries[name]}"])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
