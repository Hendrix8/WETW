#!/usr/bin/env python3
"""Run the retained Figure 6 fine-tuning protocol with saved pilot choices.

Initial encoder checkpoints come from wetw-train with the matching L preset.
Data must contain explicit splits and squared-DTW caches (wetw targets).
"""

import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data-root", required=True)
p.add_argument("--initial-root", required=True)
p.add_argument("--output", default="outputs/scalability")
p.add_argument("--device", default="cuda")
p.add_argument("--task", type=int, nargs="*")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
data = Path(a.data_root).resolve()
initial = Path(a.initial_root).resolve()
out = Path(a.output).resolve()
m = json.loads((root / "configs/paper/scalability.json").read_text())
out.mkdir(parents=True, exist_ok=True)
for name in ["encoder_selection", "head_selection"]:
    (out / (name + ".json")).write_text(
        (root / f"configs/paper/scalability_{name}.json").read_text()
    )
for t in m["tasks"]:
    d = data / t["dataset"]
    staged = out / "input" / t["dataset"]
    fit = initial / t["dataset"] / "fit"
    t.update(
        data_dir=str(staged),
        database=str(d / "database.npy"),
        queries=str(d / "queries.npy"),
        exact=str(d / "exact_neighbors.bin"),
        config=str(fit / "config.yaml"),
        encoder=str(fit / "encoder.pt"),
        wetw=str(fit / "wetw.pt"),
    )
    if not a.dry_run:
        staged.mkdir(parents=True, exist_ok=True)
        for split in ["train", "valid"]:
            link = staged / f"{split}_sample.npy"
            if not link.exists():
                link.symlink_to(d / f"{split}.npy")
            # The original Figure 6 follow-up consumes sqrt-DTW distances (not squared caches).
            src = np.load(d / f"{split}_dists.npy", mmap_mode="r")
            np.save(staged / f"{split}_dists.npy", np.sqrt(src))
(out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
for i in a.task if a.task is not None else range(len(m["tasks"])):
    cmd = [
        sys.executable,
        str(root / "experiments/reference/scalability.py"),
        "production",
        str(i),
    ]
    print(" ".join(cmd), flush=True)
    if not a.dry_run:
        subprocess.run(
            cmd,
            check=True,
            env={
                **os.environ,
                "FIG6_ROOT": str(out),
                "FIG6_DEVICE": a.device,
                "PIECE_SECONDS": "864000",
            },
        )
