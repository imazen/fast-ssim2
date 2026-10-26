#!/usr/bin/env python3
"""Capture frontend f32 bits from the independent C++ oracle (test data only)."""
import argparse
import hashlib
from pathlib import Path
import struct
import subprocess


def bits(value):
    return f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Saturated colors, darks, highlights, the HLG knee and extended code values.
    # No smooth image gradients: these are isolated frontend probes.
    pixels = [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0),
              (0.0, 1.0, 0.0, 0.25), (0.0, 0.0, 1.0, 0.75),
              (0.2, 0.4, 0.6, 0.5), (-0.01, 0.5, 1.1, 1.0)]
    rows = ["# Cloudinary SSIMULACRA2 2.1, 81feacf6a5b0bbd7ec11b4a881a4615033ab0b10",
            "# Oracle sha256: " + hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
            "# transfer primaries target background alpha r g b a linear_r linear_g linear_b x y b"]
    for tf, target in [(8, 255), (16, 10000), (18, 1000), (8, 203), (16, 255), (18, 300)]:
        for primaries in [1, 9, 11]:
            for alpha in [0, 1]:
                for bg in ([0.1, 0.9] if alpha else [0.5]):
                    header = f"{tf} {primaries} {target} {bg} {alpha}"
                    samples = [" ".join(map(bits, px)) for px in pixels]
                    request = f"frontend\n{header} {len(pixels)} 1\n" + "\n".join(samples) + "\n"
                    result = subprocess.run([str(args.oracle.resolve())], input=request,
                                            text=True, capture_output=True, check=True)
                    output = result.stdout.splitlines()
                    assert len(output) == len(samples), result
                    for sample, expected in zip(samples, output):
                        assert len(expected.split()) == 6, expected
                        rows.append(f"{header} {sample} {expected}")
    content = "\n".join(rows) + "\n"
    if args.output.exists():
        if args.output.read_text() == content:
            return
        backup = args.output.with_suffix(args.output.suffix + ".bak")
        if backup.exists():
            raise RuntimeError(f"preserve existing backup first: {backup}")
        args.output.rename(backup)
    args.output.write_text(content)
    print(f"Captured {len(rows) - 3} pixel probes, {len(content)} bytes")


if __name__ == "__main__":
    main()
