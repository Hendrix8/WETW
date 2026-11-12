"""Warm-started head refinement with the saved dataset-specific tuning recipe."""

from pathlib import Path
import copy
import math
import time
import numpy as np
import torch
from .data import Targets, write_json
from .models import WETW
from .losses import warpdistill, log_ranking
from .evaluation import validation_truth, validation_recall


def refine(cfg, encoder, head, data, embeddings, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = str(cfg.device)
    raw = np.load(Path(data) / "train.npy", mmap_mode="r")
    valid = np.load(Path(data) / "valid.npy", mmap_mode="r")
    cache = Path(data) / "train_dists.npy"
    targets = Targets(
        raw,
        cfg.training.warping_window,
        np.load(cache, mmap_mode="r") if cache.exists() else None,
        squared_cache=True,
    )
    rng = np.random.RandomState(0)
    torch.manual_seed(0)
    e = torch.as_tensor(embeddings["train"], dtype=torch.float32, device=device)
    count = int(cfg.refinement.pairs)
    i = rng.randint(len(raw), size=count)
    j = rng.randint(len(raw) - 1, size=count)
    j += j >= i
    t = targets.pairs(i, j)
    keep = t > 1e-6
    i, j, t = i[keep], j[keep], t[keep]
    if not len(t):
        raise ValueError("No positive training pairs")
    rc = cfg.dataset.refine
    anchors = rng.choice(len(raw), min(int(rc.anchors), len(raw)), replace=False)
    limit = min(int(rc.hard_rank), len(raw) - 1)
    positives = min(10, max(1, limit // 2))
    if cfg.objective == "ranking" and limit <= positives:
        raise ValueError("Ranking requires more training series")
    neighbors = []
    if cfg.objective == "ranking":
        for a in anchors:
            row = targets.row(a)
            row[a] = np.inf
            neighbors.append(np.argsort(row, kind="stable")[:limit])
        neighbors = np.stack(neighbors)
    qids, truth = validation_truth(
        valid, cfg.training.warping_window, cfg.search.validation_queries
    )
    head = head.to(device)
    # Match the original median scale initialization before refinement.
    head.train()
    with torch.no_grad():
        ratios = []
        for st in range(0, min(20000, len(t)), 4096):
            pred = head(e[i[st : st + 4096]], e[j[st : st + 4096]])
            ratios.append(
                torch.as_tensor(t[st : st + 4096], device=device) / (pred + 1e-9)
            )
        head.log_s.add_(torch.cat(ratios).median().log())
    opt = torch.optim.Adam(head.parameters(), lr=cfg.refinement.lr)
    steps = int(rc.steps)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    start = 0
    best_key = None
    best_state = None
    history = []
    resume = out / "resume.pt"
    if cfg.resume and resume.exists():
        ck = torch.load(resume, map_location=device, weights_only=False)
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["optimizer"])
        schedule.load_state_dict(ck["scheduler"])
        rng.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start = ck["step"]
        best_key = ck["best_key"]
        best_state = ck["best_state"]
        history = ck["history"]
    started = time.perf_counter()
    for step in range(start, steps):
        head.train()
        ids = rng.randint(len(t), size=cfg.refinement.batch_size)
        prediction = head(e[i[ids]], e[j[ids]])
        regression = warpdistill(prediction, torch.as_tensor(t[ids], device=device))
        loss = cfg.refinement.beta * regression
        if cfg.objective == "ranking":
            ai = rng.randint(len(anchors), size=cfg.refinement.triplets)
            a = anchors[ai]
            p = neighbors[ai, rng.randint(positives, size=len(ai))]
            n = rng.randint(len(raw) - 1, size=len(ai))
            n += n >= a
            hard = neighbors[ai, rng.randint(positives, limit, size=len(ai))]
            n = np.where(rng.rand(len(ai)) < rc.hard_fraction, hard, n)
            loss = loss + log_ranking(
                head(e[a], e[p]), head(e[a], e[n]), cfg.refinement.margin
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite refinement loss at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()
        if (step + 1) % cfg.refinement.validation_every == 0 or step + 1 == steps:
            head.eval()
            scores = validation_recall(
                embeddings["valid"], head, truth, qids, device=device
            )
            key = tuple(scores[str(r)] for r in [0.005, 0.002, 0.01])
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(head.state_dict())
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "validation_recall": scores,
                }
            )
            torch.save(
                {
                    "step": step + 1,
                    "head": head.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": schedule.state_dict(),
                    "numpy_rng": rng.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.startswith("cuda")
                    else None,
                    "best_key": best_key,
                    "best_state": best_state,
                    "history": history,
                },
                resume,
            )
    if best_state is None:
        raise RuntimeError("No validated refinement checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    torch.save(head.state_dict(), out / "best.pt")
    write_json(
        out / "history.json",
        {
            "history": history,
            "seconds": time.perf_counter() - started,
            "selection_split": "valid",
            "objective": str(cfg.objective),
            "steps": steps,
            "best_validation_key": list(best_key),
        },
    )
    return head
