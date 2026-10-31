#!/usr/bin/env python3
"""Replay captured linear RGB through the pinned Cloudinary metric.

The target is explicit so the reference's observed luminance interpretation
can be reproduced independently of its ICC profile parser.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("height", type=int)
    parser.add_argument("target", type=float)
    parser.add_argument("--expect", required=True, help="expected final f64 bits (hex)")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    rows = trace["rows"]
    assert args.height > 0 and len(rows) == 2 * args.height
    width = rows[0]["width"]
    request = ["score"]
    for image in [rows[:args.height], rows[args.height:]]:
        request.append(f"8 1 {args.target} 0.5 0 {width} {args.height}")
        for row in image:
            assert row["width"] == width
            bits = struct.unpack("<" + "I" * (width * 3),
                                 bytes.fromhex(row["output_le_f32"]))
            for i in range(0, len(bits), 3):
                request.append(f"{bits[i]:08x} {bits[i+1]:08x} {bits[i+2]:08x} 3f800000")
    output = subprocess.check_output([str(args.oracle)], input="\n".join(request) + "\n",
                                     text=True).split()
    actual = int(output[0], 16)
    assert actual == int(args.expect, 16), (output, args.expect)
    print(json.dumps({"trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                      "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
                      "width": width, "height": args.height, "target": args.target,
                      "score_bits": f"{actual:016x}", "score": float(output[1]),
                      "reference_bits_match": True}))


if __name__ == "__main__":
    main()
