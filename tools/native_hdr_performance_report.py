#!/usr/bin/env python3
"""Extract a compact, reproducible summary from a zenbench audit object."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args()
    content = args.input.read_bytes()
    if args.input.suffix == ".gz":
        content = gzip.decompress(content)
    run = json.loads(content)
    if run["incomplete"]:
        raise ValueError("refusing to summarize an incomplete run")
    groups = run["comparisons"]
    clean = sum(sample["gate_clean"] for group in groups for sample in group["samples"])
    total = sum(len(group["samples"]) for group in groups)
    if args.output.exists():
        raise FileExistsError(f"preserve the previous report first: {args.output}")
    with args.output.open("w") as out:
        out.write(f"# source_commit={args.source_commit}\n")
        out.write(f"# binary_sha256={args.binary_sha256}\n")
        out.write(f"# host_label={args.host_label}\n")
        out.write(f"# zenbench_object_sha256={hashlib.sha256(content).hexdigest()}\n")
        out.write(f"# clean_rounds={clean}/{total}; unreliable={run['unreliable']}\n")
        out.write("# Paired analysis after zenbench outlier filtering; time in milliseconds.\n")
        out.write("# Positive delta means slower. Significance includes zenbench's 1% practical threshold.\n")
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["group", "baseline", "candidate", "baseline_ms", "candidate_ms",
                         "delta_percent", "ci95_low_percent", "ci95_high_percent",
                         "kept_rounds", "practical_significant"])
        for group in groups:
            for baseline, candidate, analysis in group["analyses"]:
                base = analysis["baseline"]["mean"]
                writer.writerow([group["group_name"], baseline, candidate,
                                 f"{base / 1e6:.6f}",
                                 f"{analysis['candidate']['mean'] / 1e6:.6f}",
                                 f"{analysis['pct_change']:.4f}",
                                 f"{analysis['ci_lower'] / base * 100:.4f}",
                                 f"{analysis['ci_upper'] / base * 100:.4f}",
                                 analysis["baseline"]["n"], analysis["significant"]])


if __name__ == "__main__":
    main()
