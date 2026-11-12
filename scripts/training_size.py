#!/usr/bin/env python3
"""Nested training subsets; keep every held-out split fixed."""

import argparse, subprocess, sys
from pathlib import Path
import numpy as np
from wetw.data import condensed_index, write_json

p = argparse.ArgumentParser()
p.add_argument("--data-root", required=True)
p.add_argument("--dataset", required=True)
p.add_argument(
    "--sizes", type=int, nargs="+", default=[250, 500, 1000, 2000, 5000, 10000]
)
p.add_argument("--output", default="outputs/training_size")
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
src = Path(a.data_root).resolve() / a.dataset
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
x = np.load(src / "train.npy", mmap_mode="r")
order = np.random.RandomState(a.seed).permutation(len(x))
for n in a.sizes:
    if n > len(x):
        continue
    dest = out / f"n{n}" / "data" / a.dataset
    if not a.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "train.npy", np.asarray(x[order[:n]]))
        for split in ["valid", "test", "database", "queries"]:
            target = dest / f"{split}.npy"
            if not target.exists():
                target.symlink_to(src / f"{split}.npy")
        vc = dest / "valid_dists.npy"
        if not vc.exists():
            vc.symlink_to(src / "valid_dists.npy")
        cache = src / "train_dists.npy"
        if cache.exists():
            d = np.load(cache, mmap_mode="r")
            i, j = np.triu_indices(n, 1)
            ii, jj = order[:n][i], order[:n][j]
            np.save(dest / "train_dists.npy", d[condensed_index(ii, jj, len(x))])
        write_json(
            dest / "subset.json",
            {
                "seed": a.seed,
                "train_rows": order[:n].tolist(),
                "held_out_splits": "unchanged",
            },
        )
    cmd = [
        sys.executable,
        "-m",
        "wetw.experiment",
        f"dataset={a.dataset}",
        f"data.root={dest.parent}",
        f"output={out / f'n{n}' / 'run'}",
        f"device={a.device}",
    ]
    print(" ".join(cmd), flush=True)
    if not a.dry_run:
        subprocess.run(cmd, check=True)
