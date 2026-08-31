#!/usr/bin/env python3
"""Paired bootstrap on |variant - C++| per cell."""
import csv, random, statistics, sys, pathlib

p = pathlib.Path.home()/"tmp/ssim2-parity-study/cbrt_attribution.tsv"
rows = list(csv.DictReader(open(p), delimiter="\t"))
names = ["v071","v082","head","repro","cbrt64","cppcbrt","cppcbrtu","cppmax"]
err = {n: [abs(float(r[n]) - float(r["cpp"])) for r in rows] for n in names}
n = len(rows)
print(f"{n} cells\n")
print("mean|delta| vs C++ with 95% bootstrap CI (paired resampling of cells):")
random.seed(20260909)
idx_sets = [[random.randrange(n) for _ in range(n)] for _ in range(4000)]
means = {}
for name in names:
    e = err[name]
    bs = sorted(sum(e[i] for i in s)/n for s in idx_sets)
    means[name] = statistics.fmean(e)
    print(f"  {name:<9} {means[name]:.6f}   [{bs[100]:.6f}, {bs[3899]:.6f}]")

print("\npaired differences in mean|delta| (negative = first is closer to C++):")
def paired(a, b):
    d = [err[a][i] - err[b][i] for i in range(n)]
    bs = sorted(sum(d[i] for i in s)/n for s in idx_sets)
    m = statistics.fmean(d)
    sig = "significant" if (bs[100] < 0) == (bs[3899] < 0) else "not significant"
    wins = sum(1 for x in d if x < 0)
    print(f"  {a:<9} - {b:<9} {m:>+.6f}  95% CI [{bs[100]:>+.6f}, {bs[3899]:>+.6f}]  {sig:<15} {a} closer in {wins}/{n}")
for a in ["cbrt64","cppcbrt","cppcbrtu","cppmax"]:
    for b in ["v071","v082","head"]:
        paired(a,b)
print()
paired("v071","v082")
paired("cppcbrtu","cbrt64")
paired("cppcbrtu","cppcbrt")
paired("cppmax","cppcbrt")
paired("cppmax","cppcbrtu")

print("\nsigned mean (variant - C++), i.e. systematic bias:")
for name in names:
    s = [float(r[name]) - float(r["cpp"]) for r in rows]
    bs = sorted(sum(s[i] for i in st)/n for st in idx_sets)
    print(f"  {name:<9} {statistics.fmean(s):>+.6f}  [{bs[100]:>+.6f}, {bs[3899]:>+.6f}]")

print("\nby distortion family, mean|delta| vs C++:")
fams = sorted({r["distortion"] for r in rows})
print("  " + "family".ljust(16) + "".join(f"{x:>10}" for x in names))
for f in fams:
    sel = [i for i,r in enumerate(rows) if r["distortion"]==f]
    print("  " + f.ljust(16) + "".join(f"{statistics.fmean([err[x][i] for i in sel]):>10.5f}" for x in names))
print("\nby size:")
for sz in sorted({r["size"] for r in rows}):
    sel = [i for i,r in enumerate(rows) if r["size"]==sz]
    print("  " + sz.ljust(16) + "".join(f"{statistics.fmean([err[x][i] for i in sel]):>10.5f}" for x in names))
