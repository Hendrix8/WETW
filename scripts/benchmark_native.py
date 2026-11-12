#!/usr/bin/env python3
"""Compile/run the paper's UCR+EAP, LBKeogh-only, or LBWebb-only engine."""

import argparse, json, os, subprocess
from pathlib import Path
import numpy as np
from wetw.data import cross_distances

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
p.add_argument("--mode", choices=["usp", "lbk", "lbw"], default="usp")
p.add_argument("--window", type=float, default=0.1)
p.add_argument("--k", type=int, default=10)
p.add_argument(
    "--check",
    action="store_true",
    help="Compare all returned distances against dtaidistance",
)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
data = Path(a.data).resolve()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
binary = root / "outputs/bin/ucr_bounds"
binary.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "gcc",
        "-O3",
        "-fopenmp",
        str(root / "experiments/native/ucr_bounds.c"),
        "-lm",
        "-o",
        str(binary),
    ],
    check=True,
)
db = np.load(data / "database.npy", mmap_mode="r")
q = np.load(data / "queries.npy", mmap_mode="r")
if a.k > len(db):
    raise ValueError("k exceeds database size")
prefix = out / a.mode
subprocess.run(
    [
        str(binary),
        "exact",
        str(data / "database.npy"),
        str(data / "queries.npy"),
        str(len(q)),
        str(a.window),
        str(a.k),
        str(prefix),
        str(len(db)),
        "1",
    ],
    check=True,
    env={**os.environ, "CONTROL_LB": a.mode, "OMP_NUM_THREADS": "1"},
)
base = Path(str(prefix) + f"_N{len(db)}")
if a.check:
    d = np.fromfile(str(base) + ".dist.bin", dtype=np.float64).reshape(len(q), a.k)
    expected = np.sort(cross_distances(q, db, a.window), axis=1)[:, : a.k]
    np.testing.assert_allclose(np.sqrt(d / db.shape[1]), expected, rtol=1e-6, atol=1e-8)
    print("Exact distances agree with dtaidistance for every query.")
