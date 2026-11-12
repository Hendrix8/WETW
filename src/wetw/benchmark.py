"""Fresh, measured search curves on one machine, without time scaling."""

from pathlib import Path
import math, time
import numpy as np
import torch
from .data import cross_distances, distance, write_json
from .experiment import load_checkpoint
from .models import WETW
from .search import SearchIndex
from .evaluation import evaluate_search, parse_kappa


def paa(x, factor):
    n = x.shape[1] // factor
    if n < 1:
        raise ValueError("PAA factor exceeds the series length")
    return x[:, : n * factor].reshape(len(x), n, factor).mean(-1)


def benchmark(args):
    torch.set_num_threads(1)
    data = Path(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    db = np.load(data / "database.npy")
    queries = np.load(data / "queries.npy")
    start = time.perf_counter()
    truth_dist = cross_distances(queries, db, args.window)
    exact_seconds = (time.perf_counter() - start) / len(queries)
    truth = np.argsort(truth_dist, axis=1, kind="stable")[:, : args.k]
    results = {}
    encoder, head, _ = load_checkpoint(args.checkpoint, args.device)
    for method in args.methods:
        if method in ("wetw", "euclidean"):
            h = head if method == "wetw" else WETW(encoder.dimension, kind="ed")
            index = SearchIndex(
                encoder, h, db, args.window, args.device, normalized=True
            )
            results[method] = evaluate_search(
                index,
                queries,
                args.rhos,
                args.k,
                parse_kappa(args.kappa) if method == "wetw" else 1,
                truth,
            )
        elif method == "exact":
            results[method] = [
                {
                    "recall@1": 1.0,
                    f"recall@{args.k}": 1.0,
                    "seconds": exact_seconds,
                    "backend": "dtaidistance C",
                }
            ]
        elif method.startswith("ds"):
            factor = int(method[2:])
            dx, qx = paa(db, factor), paa(queries, factor)
            start = time.perf_counter()
            scores = cross_distances(qx, dx, args.window)
            rank_seconds = (time.perf_counter() - start) / len(queries)
            order = np.argsort(scores, axis=1, kind="stable")
            rows = []
            for rho in args.rhos:
                m = min(len(db), max(args.k, math.ceil(rho * len(db))))
                recall = []
                elapsed = []
                for i, q in enumerate(queries):
                    ids = order[i, :m]
                    start = time.perf_counter()
                    d = np.array([distance(q, db[j], args.window) for j in ids])
                    selected = ids[np.lexsort((ids, d))[: args.k]]
                    elapsed.append(time.perf_counter() - start)
                    recall.append(len(set(selected) & set(truth[i])) / args.k)
                rows.append(
                    {
                        "rho": rho,
                        f"recall@{args.k}": float(np.mean(recall)),
                        "seconds": rank_seconds + float(np.mean(elapsed)),
                    }
                )
            results[method] = rows
        else:
            raise ValueError(f"Unknown benchmark method: {method}")
    write_json(
        out / "benchmark.json",
        {
            "methods": results,
            "window": args.window,
            "threads": 1,
            "device": args.device,
            "measured": True,
            "scaled": False,
        },
    )
    print(f"Wrote {out}/benchmark.json")
