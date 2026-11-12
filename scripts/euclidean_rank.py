#!/usr/bin/env python3
"""Ranking-trained encoder with Euclidean distance (paper Table VI, WESee Euclidean row, column L).

Euclidean distance has no generator to refine, so the ranking term fine-tunes the
encoder itself. Protocol of the paper's control:

* warm start from a trained WESee encoder (``wetw-train ... stage=fit`` or a full model);
* Adam, learning rate 1e-4, 3,000 updates; each update uses 32 random regression
  pairs (relative Huber, delta 0.1) and 16 ranking triplets (log hinge, weight 1,
  margin 0.1); positives are one of the anchor's ten DTW neighbours, negatives are
  random training series; normalization statistics stay fixed (eval mode);
* the checkpoint among updates 0, 300, 1,000 and 3,000 with the best validation
  recall@1 at a 2% shortlist is kept (100 queries against up to 1,900 validation
  series; ties keep the earlier checkpoint);
* the scalar scale is calibrated on validation pairs; mean |RE| uses 20K test
  pairs and recall uses the database/query split against exact DTW neighbours.

Example::

    python scripts/euclidean_rank.py --checkpoint outputs/lendb/model.pt \
        --data data/lendb --output outputs/lendb/euclidean_rank --device cuda

``--truth`` accepts a (queries, 10) integer array of exact neighbour indices (for
example from ``wetw benchmark``); without it the neighbours are computed with DTW.
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from wetw.data import Targets, cross_distances, write_json
from wetw.encoder_ranking import condensed, encode as encode_with, refine_encoder
from wetw.experiment import load_checkpoint

RHOS = (0.005, 0.01, 0.02, 0.05, 0.1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--window", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--truth", default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out, data, dev = Path(a.output), Path(a.data), a.device
    encoder, _, _ = load_checkpoint(a.checkpoint, dev)
    encoder, info = refine_encoder(encoder, data, out, dev, a.window, a.steps, a.lr, a.seed)
    best = {"step": info["selected_step"], "score": info["validation_recall_at_1"]}
    raw = {s: np.load(data / f"{s}.npy", mmap_mode="r") for s in ("train", "valid", "test")}
    batch = 8 if raw["train"].shape[1] >= 2500 else 64

    def encode(x):
        return encode_with(encoder, x, dev, batch)

    with torch.no_grad():
        emb = {s: encode(np.load(data / f"{s}.npy", mmap_mode="r")).cpu().numpy()
               for s in ("valid", "test", "database", "queries")}

    def pairs(split, count, seed):
        r = np.random.RandomState(seed)
        m = len(emb[split])
        i, j = r.randint(m, size=count), r.randint(m, size=count)
        k = i != j
        i, j = i[k], j[k]
        t = Targets(raw[split], a.window, condensed(data, split), squared_cache=True).pairs(i, j)
        k = t > 1e-6
        return np.linalg.norm(emb[split][i[k]] - emb[split][j[k]], axis=1), t[k]

    pv, tv = pairs("valid", 10_000, 0)
    scale = float(np.mean(pv / tv))
    ptest, ttest = pairs("test", 20_000, 1)
    error = float(np.mean(np.abs(ptest / scale - ttest) / ttest))
    if a.truth:
        truth = np.load(a.truth)[:, :10]
    else:
        db = np.load(data / "database.npy", mmap_mode="r")
        qs = np.load(data / "queries.npy", mmap_mode="r")
        truth = np.argsort(cross_distances(qs, db, a.window), axis=1, kind="stable")[:, :10]
    hits1, hits10 = [], []
    for qi, q in enumerate(emb["queries"]):
        order = np.argsort(np.linalg.norm(emb["database"] - q, axis=1), kind="stable")
        cut = [max(10, int(r * len(order))) for r in RHOS]
        hits1.append([float(truth[qi, 0] in order[:c]) for c in cut])
        hits10.append([len(set(truth[qi]) & set(order[:c])) / 10 for c in cut])
    result = {
        "complete": True,
        "selected_step": best["step"],
        "validation_recall_at_1": best["score"],
        "mean_abs_re": error,
        "validation_scale": scale,
        "recall_at_1": dict(zip(map(str, RHOS), np.mean(hits1, 0).tolist())),
        "recall_at_10": dict(zip(map(str, RHOS), np.mean(hits10, 0).tolist())),
        "protocol": "Euclidean distance, encoder fine-tuned with relative Huber + log hinge (Table VI)",
    }
    write_json(out / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
