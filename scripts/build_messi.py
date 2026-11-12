#!/usr/bin/env python3
"""Build the retained MESSI-DTW runner against a supplied DaiSy checkout."""

import argparse, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--daisy", required=True)
p.add_argument("--output", default="outputs/bin/messi_run")
a = p.parse_args()
d = Path(a.daisy).resolve()
out = Path(a.output).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["cmake", "-S", str(d), "-B", str(d / "build"), "-DCMAKE_BUILD_TYPE=Release"],
    check=True,
)
subprocess.run(["cmake", "--build", str(d / "build"), "-j", "2"], check=True)
s = Path(__file__).resolve().parents[1] / "experiments/native/messi_run.cpp"
source = s.read_text().replace(
    '#include "../lib/algos/Messi.hpp"',
    '#include "' + str(d / "lib/algos/Messi.hpp") + '"',
)
cpp = out.with_suffix(".cpp")
cpp.write_text(source)
subprocess.run(
    [
        "g++",
        "-std=c++17",
        "-O3",
        "-fopenmp",
        str(cpp),
        str(d / "build/lib/libdino_lib.a"),
        str(d / "build/commons/libcommons_lib.a"),
        "-lpthread",
        "-o",
        str(out),
    ],
    check=True,
)
print(out)
