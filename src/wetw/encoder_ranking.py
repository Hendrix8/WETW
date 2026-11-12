"""Stage (iii), first half: ranking refinement of the encoder (the ranking-refined encoder of WETW-FV).

Protocol of the paper:

* warm start from the encoder trained in stages (i) and (ii);
* Adam, learning rate 1e-4, 3,000 updates; each update uses 32 random regression
  pairs (relative Huber, delta 0.1) and 16 ranking triplets (log hinge, margin 0.1)
  on the Euclidean distance of the embeddings; positives are one of the anchor's ten
  DTW neighbours, negatives are random training series; normalization statistics
  stay fixed (eval mode);
* the checkpoint among updates 0, 300, 1,000 and 3,000 with the best validation
  recall@1 at a 2% shortlist is kept (100 queries against up to 1,900 validation
  series; ties keep the earlier checkpoint).

The selected encoder serves both the Euclidean prefilter and, after the generator is
refined on its frozen embeddings, the WETW rescoring. Training state is saved at every
validation checkpoint, so an interrupted run resumes from the last one.
"""

from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data import Targets, write_json

CHECKPOINTS = (0, 300, 1000, 3000)


def condensed(data, split):
    p = Path(data) / f"{split}_dists.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def encode(encoder, x, device, batch):
    x = np.asarray(x, dtype=np.float32)
    return torch.cat(
        [encoder(torch.as_tensor(x[i : i + batch], device=device)) for i in range(0, len(x), batch)]
    )


def refine_encoder(encoder, data, out, device, window=0.1, steps=3000, lr=1e-4, seed=42, resume=True):
    """Fine-tune ``encoder`` in place and return it loaded with the selected checkpoint.

    Writes ``selected.pt`` (encoder state and Euclidean scale), ``validation.jsonl`` and
    ``encoder_ranking.json`` to ``out``.
    """
    out, data = Path(out), Path(data)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "encoder_ranking.json"
    if resume and done.exists() and (out / "selected.pt").exists():
        ck = torch.load(out / "selected.pt", map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        return encoder.eval(), json.loads(done.read_text())
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder.eval()  # gradients enabled, normalization statistics fixed
    train = np.load(data / "train.npy", mmap_mode="r")
    valid = np.load(data / "valid.npy", mmap_mode="r")
    n, length = train.shape
    batch = 8 if length >= 2500 else 64
    targets = Targets(train, window, condensed(data, "train"), squared_cache=True)
    rng = np.random.RandomState(0)
    pa, pb = rng.randint(n, size=200_000), rng.randint(n, size=200_000)
    keep = pa != pb
    pa, pb = pa[keep], pb[keep]
    pt = targets.pairs(pa, pb)
    keep = pt > 1e-6
    pa, pb, pt = pa[keep], pb[keep], pt[keep]
    anchors = rng.choice(n, min(4000, n), replace=False)
    nearest = []
    for anchor in anchors:
        row = targets.row(anchor)
        row[anchor] = np.inf
        nearest.append(np.argsort(row, kind="stable")[:10])
    nearest = np.stack(nearest)

    # Fixed validation pool: the first 100 series are queries, the rest candidates.
    nv = len(valid)
    ids = np.sort(np.random.RandomState(2).choice(nv, min(nv, 2000), replace=False))
    nq = min(100, len(ids) // 4)
    vq, vpool = ids[:nq], ids[nq:]
    vt = Targets(valid, window, condensed(data, "valid"), squared_cache=True)
    vtruth = np.array([np.argmin(vt.pairs(np.full(len(vpool), q), vpool)) for q in vq])

    log_scale = torch.nn.Parameter(torch.zeros((), device=device))
    with torch.no_grad():
        sel = np.arange(min(128, len(pt)))
        d = (encode(encoder, train[pa[sel]], device, batch) - encode(encoder, train[pb[sel]], device, batch)).norm(dim=1)
        log_scale.copy_((torch.as_tensor(pt[sel], device=device) / d.clamp_min(1e-6)).median().log())
    params = list(encoder.parameters()) + [log_scale]
    opt = torch.optim.Adam(params, lr=lr)
    srng = np.random.RandomState(seed)
    best = {"score": -1.0, "step": None}
    start = 1
    state = out / "resume.pt"
    if resume and state.exists():
        ck = torch.load(state, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        log_scale.data.copy_(ck["log_scale"])
        opt.load_state_dict(ck["optimizer"])
        srng.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        best, start = ck["best"], ck["step"] + 1

    @torch.no_grad()
    def validate(step):
        e = encode(encoder, valid[ids], device, batch)
        ranks = []
        for q, g in enumerate(vtruth):
            s = (e[q : q + 1] - e[nq:]).norm(dim=1)
            ranks.append(int((s <= s[int(g)]).sum()) - 1)  # pessimistic ties
        score = float(np.mean(np.asarray(ranks) < max(1, int(0.02 * len(vpool)))))
        with (out / "validation.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "validation_recall_at_1": score}) + "\n")
        if score > best["score"]:
            best.update(score=score, step=step)
            torch.save({"encoder": encoder.state_dict(), "log_scale": log_scale.detach()}, out / "selected.pt")
        torch.save(
            {
                "step": step,
                "encoder": encoder.state_dict(),
                "log_scale": log_scale.detach(),
                "optimizer": opt.state_dict(),
                "numpy_rng": srng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "best": dict(best),
            },
            state,
        )

    if start == 1:
        validate(0)
    t0 = time.time()
    for step in range(start, steps + 1):
        opt.zero_grad(set_to_none=True)
        draws = srng.randint(len(pt), size=32)
        ai = srng.randint(len(anchors), size=16)
        anchor, positive = anchors[ai], nearest[ai, srng.randint(10, size=16)]
        negative = srng.randint(n, size=16)
        bad = (negative == anchor) | (negative == positive)
        while bad.any():
            negative[bad] = srng.randint(n, size=int(bad.sum()))
            bad = (negative == anchor) | (negative == positive)
        for st in range(0, 32, 8):  # accumulate to bound memory on long series
            k = draws[st : st + 8]
            pred = log_scale.exp() * (
                encode(encoder, train[pa[k]], device, batch) - encode(encoder, train[pb[k]], device, batch)
            ).norm(dim=1)
            truth = torch.as_tensor(pt[k], device=device)
            rel = (pred - truth) / truth.clamp_min(1e-6)
            (F.huber_loss(rel, torch.zeros_like(rel), delta=0.1) / 4).backward()
        for st in range(0, 16, 4):
            ea = encode(encoder, train[anchor[st : st + 4]], device, batch)
            dp = (ea - encode(encoder, train[positive[st : st + 4]], device, batch)).norm(dim=1).clamp_min(1e-6)
            dn = (ea - encode(encoder, train[negative[st : st + 4]], device, batch)).norm(dim=1).clamp_min(1e-6)
            (F.relu(0.1 + dp.log() - dn.log()).mean() / 4).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        if step in CHECKPOINTS or step == steps:
            validate(step)
            print(f"encoder ranking step {step} best {best} {time.time() - t0:.0f}s", flush=True)

    ck = torch.load(out / "selected.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(ck["encoder"])
    info = {"selected_step": best["step"], "validation_recall_at_1": best["score"], "steps": steps, "lr": lr}
    write_json(done, info)
    state.unlink(missing_ok=True)
    return encoder.eval(), info
