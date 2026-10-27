#!/usr/bin/env python3
"""Copy a reference CLI with only its %.8f score format changed to %.17g.

The replacement has the same byte length. Arithmetic and every other byte
remain unchanged. Keep the original binary and record both hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = args.input.read_bytes()
    old, new = b"%.8f\n\0", b"%.17g\0"
    assert len(old) == len(new)
    assert original.count(old) == 1, "require one unambiguous score format"
    offset = original.index(old)
    patched = original[:offset] + new + original[offset + len(old):]
    assert len(patched) == len(original)
    with args.output.open("xb") as out:
        out.write(patched)
    args.output.chmod(args.input.stat().st_mode | 0o100)
    metadata = {
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "precision_copy_sha256": hashlib.sha256(patched).hexdigest(),
        "offset": offset, "old_hex": old.hex(), "new_hex": new.hex(),
        "purpose": "17 significant digits for round-tripping the CLI's f64 score",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
