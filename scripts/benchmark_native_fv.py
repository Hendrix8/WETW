#!/usr/bin/env python3
"""Measured WETW/ED ranking with the paper's native exact verification cascade."""

import argparse, json, math, os, subprocess
from pathlib import Path
import numpy as np
import torch
from wetw.experiment import load_checkpoint
from wetw.models import WETW
from wetw.search import SearchIndex
from wetw.data import write_json, fingerprint
from wetw.evaluation import parse_kappa, kappa_for

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
p.add_argument(
    "--rhos",
    nargs="+",
    type=float,
    default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
)
p.add_argument("--k", type=int, default=10)
p.add_argument("--kappa", default="5", help="prefilter size factor: an integer (0 = whole database) or the prefilter_selection.json of the run")
p.add_argument("--window", type=float, default=0.1)
p.add_argument("--device", default="cpu")
a = p.parse_args()
a.kappa = parse_kappa(a.kappa)
torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
data = Path(a.data).resolve()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
signature = {
    "checkpoint": fingerprint(a.checkpoint),
    "database": fingerprint(data / "database.npy"),
    "queries": fingerprint(data / "queries.npy"),
    "k": a.k,
    "window": a.window,
    "kappa": a.kappa,
    "device": a.device,
}
if (out / "signature.json").exists() and json.loads(
    (out / "signature.json").read_text()
) != signature:
    raise ValueError(
        "Output belongs to different inputs/settings. Use a new output directory."
    )
write_json(out / "signature.json", signature)
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
n, nq = len(db), len(q)
if a.k > n:
    raise ValueError("k exceeds database size")
env = {**os.environ, "CONTROL_LB": "usp", "OMP_NUM_THREADS": "1"}
base = [
    str(binary),
    "exact",
    str(data / "database.npy"),
    str(data / "queries.npy"),
    str(nq),
    str(a.window),
    str(a.k),
    str(out / "exact"),
    str(n),
    "1",
]
subprocess.run(base, check=True, env=env)
truth = np.fromfile(str(out / "exact") + f"_N{n}.nn.bin", np.int32).reshape(nq, a.k)
exact = json.loads(Path(str(out / "exact") + f"_N{n}.json").read_text())
e, h, _ = load_checkpoint(a.checkpoint, a.device)
rows = {}
for method, head, kappa in [
    ("wetw", h, a.kappa),
    ("euclidean", WETW(e.dimension, kind="ed"), 1),
]:
    index = SearchIndex(e, head, db, a.window, a.device, normalized=True)
    curves = []
    for rho in a.rhos:
        m = min(n, max(a.k, math.ceil(rho * n)))
        orders = np.empty((nq, n), np.int32)
        encoding = []
        ranking = []
        for i, query in enumerate(q):
            ids, et, rt, _ = index.ranking(query, a.k, rho, kappa_for(kappa, rho))
            orders[i, :m] = ids
            orders[i, m:] = np.setdiff1d(np.arange(n), ids, assume_unique=True)
            encoding.append(et)
            ranking.append(rt)
        order_file = out / f"{method}_{rho:g}.order.bin"
        orders.tofile(order_file)
        prefix = out / f"{method}_{rho:g}"
        cmd = [
            str(binary),
            "verify",
            str(data / "database.npy"),
            str(data / "queries.npy"),
            str(nq),
            str(a.window),
            str(a.k),
            str(prefix),
            str(m),
            "1",
            str(order_file),
        ]
        subprocess.run(cmd, check=True, env=env)
        ids = np.fromfile(str(prefix) + f"_m{m}.nn.bin", np.int32).reshape(nq, a.k)
        timing = json.loads(Path(str(prefix) + f"_m{m}.json").read_text())
        recall = float(
            np.mean([len(set(x) & set(y)) / a.k for x, y in zip(ids, truth)])
        )
        curves.append(
            {
                "rho": rho,
                f"recall@{a.k}": recall,
                "encoding_seconds": float(np.mean(encoding)),
                "ranking_seconds": float(np.mean(ranking)),
                "verification_seconds": float(np.mean(timing["seconds"])),
                "seconds": float(
                    np.mean(encoding) + np.mean(ranking) + np.mean(timing["seconds"])
                ),
            }
        )
        order_file.unlink()
    rows[method] = curves
write_json(
    out / "benchmark.json",
    {
        "methods": rows,
        "exact_seconds": float(np.mean(exact["seconds"])),
        "verification": "UCR Suite + EAPrunedDTW",
        "ranking_device": a.device,
        "threads_per_query": 1,
        "measured": True,
        "scaled": False,
    },
)
print(out / "benchmark.json")
