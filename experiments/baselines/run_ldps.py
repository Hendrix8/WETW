#!/usr/bin/env python3
"""LDPS baseline (Lods et al.) on one collection, with the bl_v2 protocol of the other learned baselines.

Model: LDPS_GPU (mimic_models_gpu.py next to this file, a pinned copy of 1-dtwrl/lib/LDPS on thalia): shapelets of
lengths {0.15, 0.30, 0.45} L as in LDPS, 128 shapelets in all (embedding size d = 128 as RF-DTAN / CDPS-128),
k-means initialisation, Adagrad (lr 0.01), prediction beta * ||ST(x_i) - ST(x_j)||.
Objective (LDPS_LOSS):
  native : LDPS's own squared error between the prediction and the band-0.1L DTW distance
  wd     : WarpDistill, the relative Huber error (delta 0.1, eps 1e-6) of the same prediction
Training pairs are sampled uniformly from the training split (condensed DTW targets held on the GPU), 64 pairs per
step, for the 6-h budget (LDPS_HOURS) or LDPS_MAX_ITER steps. Model selection on validation (valid_select = 2000):
the lowest value of the method's own objective on 2000 fixed validation pairs, checked every LDPS_VAL_EVERY steps.
Resumable: resume.pt (model, optimiser, step, used budget, best state, RNG) every 10 min and at the end.
LDPS_PIECE_HOURS (optional, short dev-queue pieces): stop training after this many hours of this invocation,
checkpoint and exit 3 without outputs; the next invocation resumes and the total budget stays LDPS_HOURS.
Writes <out>/emb_{train,valid,test,db,q}.npy of the best model (LDPS_EMBED: other splits, "" for none), model.pt,
train_meta.json, fit_seconds.txt.
usage: run_ldps.py <dataset> <out_dir>    (data: LDPS_SD samples dir, LDPS_DB, LDPS_Q; LDPS_WINDOW: band fraction)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mimic_models_gpu import LDPS_GPU  # noqa: E402

# conv1d computes ||w||^2 - 2 w.s + ||s||^2; TF32 (cuDNN default on A100) loses this cancellation
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

ds, out = sys.argv[1], sys.argv[2]
LOSS = os.environ.get("LDPS_LOSS", "native")
assert LOSS in ("native", "wd"), LOSS
HOURS = float(os.environ.get("LDPS_HOURS", 6))
PIECE = float(os.environ.get("LDPS_PIECE_HOURS", "inf"))
MAX_ITER = int(os.environ.get("LDPS_MAX_ITER", 2_000_000))
VAL_EVERY = int(os.environ.get("LDPS_VAL_EVERY", 2000))
WINDOW = float(os.environ.get("LDPS_WINDOW", 0.1))
EMBED = [s for s in os.environ.get("LDPS_EMBED", "train,valid,test,db,q").split(",") if s]
BATCH, LR, N_SHP, N_VAL = 64, 0.01, 128, 2000
S = os.environ.get("LDPS_SD", os.environ["LDPS_SD"])
SPLITS = {"train": f"{S}/train_sample.npy", "valid": f"{S}/valid_sample.npy", "test": f"{S}/test_sample.npy",
          "db": os.environ.get("LDPS_DB", f"{S}/db.npy"),
          "q": os.environ.get("LDPS_Q", os.environ["LDPS_Q"])}
os.makedirs(out, exist_ok=True)
dev = torch.device(os.environ.get("WETW_DEVICE","cuda"))
torch.manual_seed(42)
np.random.seed(42)


def condensed(n, a, b):
    a, b = torch.minimum(a, b), torch.maximum(a, b)
    return n * a - a * (a + 1) // 2 + b - a - 1


def objective(pred, target):
    if LOSS == "native":
        return F.mse_loss(pred, target)
    return F.huber_loss((pred - target) / (target + 1e-6), torch.zeros_like(target), delta=0.1)


Xtr = np.asarray(np.load(SPLITS["train"], mmap_mode="r"), dtype=np.float32)
n, L = Xtr.shape
counts = [N_SHP // 3 + (k < N_SHP % 3) for k in range(3)]  # 43 / 43 / 42
lengths = {max(2, int(p * L)): c for p, c in zip((0.15, 0.30, 0.45), counts)}
model = LDPS_GPU(lengths, d=1, convergence_rate=LR, device=str(dev), init_method="kmeans")
opt = torch.optim.Adagrad(model.parameters(), lr=LR)
X = torch.from_numpy(Xtr).to(dev)[:, :, None]
Dtr = torch.from_numpy(np.asarray(np.load(f"{S}/train_dists.npy", mmap_mode="r"), dtype=np.float32)).to(dev)
assert Dtr.numel() == n * (n - 1) // 2, (Dtr.numel(), n)
# Same targets as every other method: spot-check plain band DTW (dtaidistance, window max(1, int(WINDOW L))).
from dtaidistance import dtw  # noqa: E402
for i, j in ((0, 1), (1, 2), (0, n - 1)):
    ref = dtw.distance_fast(Xtr[i].astype(np.float64), Xtr[j].astype(np.float64), window=max(1, int(WINDOW * L)))
    got = float(Dtr[condensed(n, torch.tensor(i), torch.tensor(j))])
    assert np.isclose(got, ref, rtol=1e-4, atol=1e-4), (i, j, got, ref)
print("LDPS_TARGETS_VERIFIED", flush=True)
Xva = np.load(SPLITS["valid"], mmap_mode="r")
nv = Xva.shape[0]
rv = np.random.RandomState(7)
va, vb = rv.randint(0, nv, 4 * N_VAL), rv.randint(0, nv, 4 * N_VAL)
keep = va != vb
va, vb = va[keep][:N_VAL], vb[keep][:N_VAL]
Dva = np.load(f"{S}/valid_dists.npy", mmap_mode="r")
lo, hi = np.minimum(va, vb), np.maximum(va, vb)
tva = torch.from_numpy(np.asarray(Dva[nv * lo - lo * (lo + 1) // 2 + hi - lo - 1], dtype=np.float32)).to(dev)
Xva_a = torch.from_numpy(np.asarray(Xva[va], dtype=np.float32)).to(dev)[:, :, None]
Xva_b = torch.from_numpy(np.asarray(Xva[vb], dtype=np.float32)).to(dev)[:, :, None]


@torch.no_grad()
def validate():
    model.eval()
    pred = torch.cat([model.forward_dist(Xva_a[s:s + 256], Xva_b[s:s + 256]) for s in range(0, N_VAL, 256)])
    model.train()
    return float(objective(pred, tva))


ck = f"{out}/resume.pt"
if os.path.exists(ck):
    st = torch.load(ck, map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    step, used, best, best_state, history = st["step"], st["used"], st["best"], st["best_state"], st["history"]
    torch.set_rng_state(st["rng"].cpu())  # map_location moved the saved states to the GPU
    if torch.cuda.is_available() and st["cuda_rng"] is not None:
        torch.cuda.set_rng_state(st["cuda_rng"].cpu())
    print(f"resumed at step {step}, {used / 3600:.2f} h used, best validation {best:.5g}", flush=True)
else:
    t = time.time()
    model.init_shapelets(Xtr[:, :, None])
    print(f"k-means init {time.time() - t:.0f} s", flush=True)
    step, used, best, best_state, history = 0, 0.0, float("inf"), None, []


def checkpoint():
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, used=used, best=best,
                    best_state=best_state, history=history, rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None), ck + ".tmp")
    os.replace(ck + ".tmp", ck)


model.train()
t_last, t_ck = time.time(), time.time()
t_piece = t_last
while step < MAX_ITER and used < HOURS * 3600:
    if time.time() - t_piece > PIECE * 3600:
        used += time.time() - t_last
        checkpoint()
        print(f"LDPS_PIECE_END step {step} {used / 3600:.3f} of {HOURS} h used", flush=True)
        sys.exit(3)
    i = torch.randint(0, n, (BATCH,), device=dev)
    j = torch.randint(0, n - 1, (BATCH,), device=dev)
    j = j + (j >= i)  # uniform over j != i
    loss = objective(model.forward_dist(X[i], X[j]), Dtr[condensed(n, i, j)])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        model.beta.clamp_(min=1e-6)
    step += 1
    if not torch.isfinite(loss):
        raise SystemExit(f"non-finite loss at step {step}")
    if step % VAL_EVERY == 0 or step == MAX_ITER:
        v = validate()
        history.append((step, float(loss), v))
        if v < best:
            best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
        now = time.time()
        used += now - t_last
        t_last = now
        print(f"step {step} loss {float(loss):.5g} valid {v:.5g} best {best:.5g} beta {float(model.beta):.4g} "
              f"{used / 3600:.3f} h", flush=True)
        if now - t_ck > 600:
            checkpoint()
            t_ck = now
used += time.time() - t_last
if best_state is None:  # budget ended before the first validation
    v = validate()
    best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    history.append((step, float("nan"), v))
checkpoint()
model.load_state_dict(best_state)
model.save(f"{out}/model.pt")
del X, Dtr
torch.cuda.empty_cache()


@torch.no_grad()
def embed(fp):
    A = np.load(fp, mmap_mode="r")
    model.eval()
    E = [model.shapelet_transform(torch.from_numpy(np.asarray(A[s:s + 512], dtype=np.float32)).to(dev)[:, :, None]).cpu()
         for s in range(0, A.shape[0], 512)]
    return torch.cat(E).numpy().astype(np.float32)


for name in EMBED:
    fp = SPLITS[name]
    if os.path.exists(f"{out}/emb_{name}.npy"):
        continue
    E = embed(fp)
    assert E.shape[1] == N_SHP and np.isfinite(E).all(), (name, E.shape)
    np.save(f"{out}/emb_{name}.tmp.npy", E)
    os.replace(f"{out}/emb_{name}.tmp.npy", f"{out}/emb_{name}.npy")  # a killed job never leaves a partial split
    print(name, E.shape, flush=True)
src = open(f"{HERE}/mimic_models_gpu.py", "rb").read()
json.dump(dict(method="ldps" if LOSS == "native" else "ldps_wd", objective=LOSS, d=N_SHP, n_shapelets=N_SHP,
               shapelet_lengths={str(k): v for k, v in lengths.items()}, batch_pairs=BATCH, lr=LR, optimizer="Adagrad",
               steps=step, budget_hours=HOURS, used_hours=used / 3600, best_valid=best, valid_pairs=int(N_VAL),
               val_every=VAL_EVERY, history=history, n_train=n, L=L, beta=float(model.beta), seed=42,
               gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", targets=f"verified plain band DTW, window fraction {WINDOW:g}",
               mimic_models_gpu_md5=hashlib.md5(src).hexdigest()), open(f"{out}/train_meta.json", "w"), indent=1)
open(f"{out}/fit_seconds.txt", "w").write(f"{used:.1f}\n")
print("LDPS_DONE", ds, LOSS, "steps", step, "best valid", best, flush=True)
