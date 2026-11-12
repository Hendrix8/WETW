"""Validation-only model/prefilter selection and independent test evaluation."""

import math
import numpy as np
import torch
from scipy.stats import spearmanr
from .data import Targets, cross_distances
from .search import SearchIndex


def rank_embeddings(emb, head, query_ids, device="cpu", batch=4096):
    head.eval()
    emb = torch.as_tensor(emb, dtype=torch.float32, device=device)
    rows = []
    with torch.inference_mode():
        for q in query_ids:
            parts = []
            for st in range(0, len(emb), batch):
                e = emb[st : st + batch]
                parts.append(head(emb[q].expand(len(e), -1), e))
            scores = torch.cat(parts)
            scores[q] = torch.inf
            rows.append(torch.argsort(scores, stable=True).cpu().numpy())
    return np.stack(rows)


def validation_recall(
    emb, head, truth, query_ids, rhos=(0.005, 0.002, 0.01), device="cpu"
):
    order = rank_embeddings(emb, head, query_ids, device)
    result = {}
    for rho in rhos:
        m = min(len(emb) - 1, max(1, math.ceil(rho * len(emb))))
        result[str(rho)] = float(
            np.mean([truth[i, 0] in order[i, :m] for i in range(len(order))])
        )
    return result


def validation_truth(series, window, queries=300, seed=1):
    qids = np.sort(
        np.random.RandomState(seed).choice(
            len(series), min(queries, len(series)), replace=False
        )
    )
    d = cross_distances(series[qids], series, window)
    d[np.arange(len(qids)), qids] = np.inf
    return qids, np.argsort(d, axis=1, kind="stable")[:, : min(10, len(series) - 1)]


def pair_fidelity(
    emb, head, series, window, pairs=20000, seed=0, calibration=1.0, device="cpu"
):
    rng = np.random.RandomState(seed)
    n = len(series)
    i = rng.randint(n, size=pairs)
    j = rng.randint(n - 1, size=pairs)
    j += j >= i
    true = Targets(series, window).pairs(i, j)
    e = torch.as_tensor(emb, dtype=torch.float32, device=device)
    prediction = []
    with torch.inference_mode():
        for st in range(0, pairs, 4096):
            prediction.extend(
                head(e[i[st : st + 4096]], e[j[st : st + 4096]]).cpu().numpy()
            )
    pred = np.asarray(prediction)
    keep = true > 1e-6
    if not keep.any():
        raise ValueError("No positive-distance pairs for fidelity evaluation")
    re = (pred[keep] / calibration - true[keep]) / true[keep]
    corr = spearmanr(pred[keep], true[keep]).statistic
    return {
        "mean_abs_re": float(np.abs(re).mean()),
        "mean_signed_re": float(re.mean()),
        "spearman": float(corr) if np.isfinite(corr) else None,
        "pairs": int(keep.sum()),
    }


def parse_kappa(value):
    """An integer, or the path of a prefilter_selection.json written by wetw-train."""
    try:
        return int(value)
    except (TypeError, ValueError):
        import json
        from pathlib import Path

        return {str(float(r)): int(k) for r, k in json.loads(Path(value).read_text())["kappa"].items()}


def kappa_for(kappa, rho):
    """kappa is one integer for every shortlist, or a {rho: kappa} map from select_prefilter."""
    if isinstance(kappa, dict):
        return int(kappa[str(float(rho))] if str(float(rho)) in kappa else kappa[rho])
    return int(kappa)


def evaluate_search(index, queries, rhos, k=10, kappa=5, truth=None):
    if truth is None:
        truth = np.argsort(
            cross_distances(queries, index.database, index.window),
            axis=1,
            kind="stable",
        )[:, :k]
    rows = []
    for rho in rhos:
        kr = kappa_for(kappa, rho)
        results = [index.query(q, k, rho, kr, normalized=True) for q in queries]
        recall1 = np.mean(
            [int(truth[i, 0] in r.indices) for i, r in enumerate(results)]
        )
        recallk = np.mean(
            [len(set(truth[i, :k]) & set(r.indices)) / k for i, r in enumerate(results)]
        )
        rows.append(
            {
                "rho": float(rho),
                "kappa": kr,
                "recall@1": float(recall1),
                f"recall@{k}": float(recallk),
                "seconds": float(np.mean([r.seconds for r in results])),
                "encoding_seconds": float(
                    np.mean([r.encoding_seconds for r in results])
                ),
                "ranking_seconds": float(np.mean([r.ranking_seconds for r in results])),
                "verification_seconds": float(
                    np.mean([r.verification_seconds for r in results])
                ),
            }
        )
    return rows


def choose_kappa(records, tolerance=0.005):
    """The paper's rule: the smallest kappa whose validation recall@1 and recall@10 are both
    within ``tolerance`` of the best, else the best recall@1 + recall@10 (0 = whole database,
    ordered last)."""
    size = lambda r: math.inf if r["kappa"] == 0 else r["kappa"]
    b1 = max(r["recall@1"] for r in records)
    b10 = max(r["recall@10"] for r in records)
    close = [
        r
        for r in records
        if r["recall@1"] >= b1 - tolerance and r["recall@10"] >= b10 - tolerance
    ]
    if close:
        return min(close, key=size)["kappa"]
    return max(records, key=lambda r: (r["recall@1"] + r["recall@10"], -size(r)))["kappa"]


def select_prefilter(emb, head, series, window, rhos, grid, queries=300, device="cpu", tolerance=0.005):
    """Choose kappa for every shortlist fraction on a validation-only pool.

    Each validation query ranks the rest of the validation split: Euclidean prefilter
    of kappa * rho * n candidates, WETW rescoring, top rho * n kept. Test, database and
    query arrays are not read.
    """
    qids, truth = validation_truth(series, window, queries)
    e = torch.as_tensor(emb, dtype=torch.float32, device=device)
    head.eval()
    n = len(e) - 1  # candidates per query (the query itself is excluded)
    hits = {(kp, r): ([], []) for kp in grid for r in rhos}
    with torch.inference_mode():
        for i, q in enumerate(qids):
            base = (e - e[q]).norm(dim=-1)
            base[q] = torch.inf
            order = torch.argsort(base, stable=True)[:n]
            # one WETW score per candidate; every (kappa, rho) rescoring is a slice of it
            scores = torch.cat(
                [head(e[q].expand(len(c), -1), c) for c in e[order].split(4096)]
            )
            for kp in grid:
                for r in rhos:
                    m = min(n, max(1, math.ceil(r * n)))
                    p = n if kp == 0 else min(n, max(m, math.ceil(kp * r * n)))
                    if p == m:
                        keep = order[:m]
                    else:
                        keep = order[:p][torch.argsort(scores[:p], stable=True)[:m]]
                    keep = set(keep.cpu().numpy().tolist())
                    hits[(kp, r)][0].append(float(truth[i, 0] in keep))
                    hits[(kp, r)][1].append(len(set(truth[i].tolist()) & keep) / truth.shape[1])
    candidates, kappa = {}, {}
    for r in rhos:
        records = [
            {
                "kappa": int(kp),
                "recall@1": float(np.mean(hits[(kp, r)][0])),
                "recall@10": float(np.mean(hits[(kp, r)][1])),
            }
            for kp in grid
        ]
        candidates[str(float(r))] = records
        kappa[str(float(r))] = int(choose_kappa(records, tolerance))
    return {
        "kappa": kappa,
        "candidates": candidates,
        "rule": f"smallest kappa with validation recall@1 and recall@10 within {tolerance} of the best (0 = whole database)",
        "selection_split": "valid",
    }
