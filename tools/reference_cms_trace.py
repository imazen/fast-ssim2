#!/usr/bin/env python3
"""Capture x86-64 SysV C++ reference CMS row inputs/outputs without changing binaries.

Usage: reference_cms_trace.py ORIGINAL_CLI SOURCE_PNG DISTORTED_PNG NEW_JSON
Requires GDB with Python support and unstripped reference symbols. RGB only.
The trace stores raw little-endian f32 bytes, one event per CMS row call.
"""
import json
import os
from pathlib import Path
import sys


def trace():
    import gdb

    records = []
    targets = []
    profiles = []
    pending = set()
    gdb.execute("set pagination off")
    gdb.execute("set confirm off")
    gdb.execute("set history save off")
    gdb.execute("set disable-randomization off")

    class Return(gdb.FinishBreakpoint):
        def __init__(self, record, output, size):
            super().__init__(gdb.newest_frame(), internal=True)
            self.record, self.output, self.size = record, output, size
            pending.add(self)

        def stop(self):
            self.record["output_le_f32"] = bytes(
                gdb.selected_inferior().read_memory(self.output, self.size)
            ).hex()
            pending.remove(self)
            return False

    class Entry(gdb.Breakpoint):
        def stop(self):
            # x86-64 SysV ABI: t, thread, input, output, width.
            pointer = lambda reg: int(gdb.parse_and_eval("$" + reg))
            width = pointer("r8")
            assert 0 < width <= 16384, width
            size = width * 3 * 4
            record = {
                "width": width,
                "input_le_f32": bytes(gdb.selected_inferior().read_memory(
                    pointer("rdx"), size)).hex(),
            }
            records.append(record)
            Return(record, pointer("rcx"), size)
            return False

    class Target(gdb.Breakpoint):
        def stop(self):
            targets.append({"function": self.location,
                            "intensity_target": float(gdb.parse_and_eval("$xmm0.v4_float[0]"))})
            return False

    class ProfileReturn(gdb.FinishBreakpoint):
        def __init__(self, address):
            super().__init__(gdb.newest_frame(), internal=True)
            self.address = address
            pending.add(self)

        def stop(self):
            data = bytes(gdb.selected_inferior().read_memory(self.address, 104))
            profiles.append({"color_space": int.from_bytes(data[:4], "little"),
                             "primaries": int.from_bytes(data[24:28], "little"),
                             "transfer": int.from_bytes(data[80:84], "little")})
            pending.remove(self)
            return False

    class Profile(gdb.Breakpoint):
        def stop(self):
            ProfileReturn(int(gdb.parse_and_eval("$rcx")))
            return False

    Entry("jxl::(anonymous namespace)::DoColorSpaceTransform(void*, unsigned long, float const*, float*, unsigned long)")
    Target("jxl::(anonymous namespace)::JxlCmsInit(void*, unsigned long, unsigned long, JxlColorProfile const*, JxlColorProfile const*, float)")
    if os.environ.get("SSIM2_TRACE_MODERN_XYB") == "1":
        Profile("jxl::(anonymous namespace)::JxlCmsSetFieldsFromICC(void*, unsigned char const*, unsigned long, JxlColorEncoding*, int*)")
        Target("jxl::ToXYB(jxl::ColorEncoding const&, float, jxl::Plane<float> const*, jxl::ThreadPool*, jxl::Image3<float>*, JxlCmsInterface const&, jxl::Image3<float>*)")
    gdb.execute("run")
    assert records and not pending, "no completed CMS calls"
    assert int(gdb.parse_and_eval("$_exitcode")) == 0
    with open(os.environ["SSIM2_CMS_TRACE"], "x") as out:
        json.dump({"abi": "x86-64 SysV, RGB f32", "rows": records,
                   "targets": targets, "profiles": profiles}, out)
        out.write("\n")


def main():
    import hashlib
    import subprocess

    assert len(sys.argv) == 5, __doc__
    binary, source, distorted, output = map(Path, sys.argv[1:])
    assert not output.exists(), "output must be new"
    env = dict(os.environ, SSIM2_CMS_TRACE=str(output.resolve()))
    subprocess.run(["gdb", "-nx", "-q", "-batch", "-x", str(Path(__file__).resolve()),
                    "--args", str(binary), str(source), str(distorted)], env=env, check=True)
    data = json.loads(output.read_text())
    data["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    data["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    data["distorted_sha256"] = hashlib.sha256(distorted.read_bytes()).hexdigest()
    output.write_text(json.dumps(data) + "\n")
    print(f"Captured {len(data['rows'])} CMS rows in {output}")


if "gdb" in sys.modules:
    trace()
elif __name__ == "__main__":
    main()
