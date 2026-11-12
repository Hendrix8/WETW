import argparse
from pathlib import Path
import json
import numpy as np


def main():
    p = argparse.ArgumentParser(
        description="WETW dataset, search and reproducibility tools. Training: wetw-train --help"
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("prepare", help="Import five disjoint NPZ splits")
    d.add_argument("--input", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--already-normalized", action="store_true")
    t = sub.add_parser("targets", help="Build quadratic condensed DTW target caches")
    t.add_argument("--data", required=True)
    t.add_argument("--window", type=float, default=0.1)
    t.add_argument("--squared", action="store_true")
    t.add_argument("--splits", nargs="+", default=["train", "valid"])
    q = sub.add_parser("search", help="Search new queries with a trained model")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--database", required=True)
    q.add_argument("--queries", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--rho", type=float, default=0.02)
    q.add_argument("--kappa", default="5", help="prefilter size factor: an integer (0 = whole database) or the prefilter_selection.json of the run")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--window", type=float, default=0.1)
    r = sub.add_parser(
        "paper", help="Regenerate paper figures and tables from saved evidence"
    )
    r.add_argument("--results", default="results/paper")
    r.add_argument("--output", default="outputs/paper")
    b = sub.add_parser(
        "benchmark", help="Benchmark exact and learned search on supplied splits"
    )
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--data", required=True)
    b.add_argument("--output", required=True)
    b.add_argument(
        "--methods", nargs="+", default=["wetw", "euclidean", "exact", "ds4"]
    )
    b.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    )
    b.add_argument("--k", type=int, default=10)
    b.add_argument("--window", type=float, default=0.1)
    b.add_argument("--device", default="cpu")
    b.add_argument("--kappa", default="5", help="prefilter size factor: an integer (0 = whole database) or the prefilter_selection.json of the run")
    args = p.parse_args()
    if args.command == "prepare":
        from .data import prepare_npz

        print(
            json.dumps(
                prepare_npz(args.input, args.output, not args.already_normalized),
                indent=2,
            )
        )
    elif args.command == "targets":
        from .reference_training import make_condensed

        for split in args.splits:
            src = Path(args.data) / f"{split}.npy"
            suffix = "_dists.npy" if args.squared else "_targets.npy"
            make_condensed(
                np.load(src, mmap_mode="r"),
                Path(args.data) / (split + suffix),
                args.window,
                args.squared,
            )
    elif args.command == "search":
        from .experiment import load_checkpoint
        from .search import SearchIndex
        from .data import write_json
        from .evaluation import parse_kappa, kappa_for

        encoder, head, _ = load_checkpoint(args.checkpoint, args.device)
        index = SearchIndex(
            encoder, head, np.load(args.database), args.window, args.device
        )
        rows = []
        for q in np.load(args.queries):
            result = index.query(q, args.k, args.rho, kappa_for(parse_kappa(args.kappa), args.rho))
            rows.append(
                {
                    "indices": result.indices.tolist(),
                    "distances": result.distances.tolist(),
                    "seconds": result.seconds,
                }
            )
        write_json(args.output, rows)
    elif args.command == "paper":
        from .paper import reproduce

        reproduce(Path(args.results), Path(args.output))
    elif args.command == "benchmark":
        from .benchmark import benchmark

        benchmark(args)
