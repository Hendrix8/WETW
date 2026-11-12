import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from dtaidistance import dtw
import time
import wandb

# Add lib/RF-DTAN to path
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.environ["WETW_RFDTAN_ROOT"]
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from models.RFDTAN import RFDTAN
from models.train_utils import one_batch
from loss.alignment_loss import ICAE
from loss.smoothness_prior import PriorLoss
from helper.UCR_loader import UCRDataset

def spread_penalty(E):
    """1 - PR/d of a batch of embeddings, PR = (sum v)^2 / sum v^2 over the per-coordinate variances v:
    0 when every coordinate varies equally, ~1 when one dominates (scale-free; encourages effective dims)."""
    v = E.float().var(0)
    return 1.0 - (v.sum() ** 2 / (v.pow(2).sum() + 1e-12)) / E.shape[1]


def sanitize_time_series(X, dim_series):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X)
    if X.ndim == 1:
        X = X.reshape(-1, dim_series)
    elif X.ndim == 2:
        if X.shape[1] != dim_series:
            # Simple truncate/pad to match dim_series if needed
            if X.shape[1] > dim_series:
                X = X[:, :dim_series]
            else:
                pad = np.zeros((X.shape[0], dim_series - X.shape[1]))
                X = np.hstack([X, pad])
    return X.astype(np.float32)

def compact_to_square(compact, n):
    """Compact upper-triangle vector (length n*(n-1)/2) -> symmetric n×n, zero diagonal."""
    D = np.zeros((n, n), dtype=np.float32)
    compact = np.asarray(compact).ravel().astype(np.float32)
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            D[i, j] = compact[idx]
            D[j, i] = compact[idx]
            idx += 1
    return D

class ScaledPairwiseDistDiff(torch.nn.Module):
    def __init__(self, dim_series, dim_emb, device="cuda", precomputed_dists=None):
        super().__init__()
        self.dim_series = dim_series
        self.dim_emb = dim_emb
        self.device = device
        self.precomputed_dists = precomputed_dists # Square matrix if provided

    def forward(self, embeddings, batch_series, batch_indices=None, warping_window=0.1):
        B = embeddings.size(0)
        nlogn = int(B * np.log(B))
        num_pairs = min(nlogn, B * (B - 1) // 2)

        i_indices, j_indices = np.triu_indices(B, k=1)
        selected_idx = np.random.choice(len(i_indices), num_pairs, replace=False)
        pairs = np.stack((i_indices[selected_idx], j_indices[selected_idx]), axis=1)

        dtw_dists = []
        euclidean_dists = []

        window = int(warping_window * batch_series.shape[-1]) if warping_window < 1.0 else None

        for i, j in pairs:
            # batch_series is (B, C, L) or (B, L)
            if self.precomputed_dists is not None and batch_indices is not None:
                # Use precomputed distance
                idx_i = batch_indices[i].item()
                idx_j = batch_indices[j].item()
                dtw_dist = self.precomputed_dists[idx_i, idx_j]
            else:
                # On-the-fly DTW
                ts_i = batch_series[i].detach().cpu().numpy().squeeze()
                ts_j = batch_series[j].detach().cpu().numpy().squeeze()
                dtw_dist = dtw.distance_fast(ts_i.astype(np.double), ts_j.astype(np.double), window=window)
            
            # WETW scaling convention: DTW / sqrt(dim_series)
            target_scaled = dtw_dist / np.sqrt(self.dim_series)
            
            # Prediction-side scaling: emb_dist / sqrt(dim_emb)
            emb_dist = torch.norm(embeddings[i] - embeddings[j], p=2) 
            pred_scaled = emb_dist / np.sqrt(self.dim_emb)

            dtw_dists.append(target_scaled)
            euclidean_dists.append(pred_scaled)

        dtw_dists = torch.tensor(dtw_dists, dtype=torch.float32, device=self.device)
        euclidean_dists = torch.stack(euclidean_dists)

        # Loss matching WETW style (relative huber on relative error)
        eps = 1e-6
        r = (euclidean_dists - dtw_dists) / (dtw_dists + eps)
        beta = 0.1
        loss = torch.mean(torch.where(
            torch.abs(r) < beta,
            0.5 * r**2 / beta,
            torch.abs(r) - 0.5 * beta
        ))
        return loss

@hydra.main(config_path=".", config_name="config_rfdtan", version_base=None)
def main(cfg: DictConfig):
    # Setup
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.system.seed)
    np.random.seed(cfg.system.seed)

    # Load Data using WETW convention
    if not os.path.exists(cfg.dataset.dataset_fp):
        raise FileNotFoundError(f"Dataset not found at {cfg.dataset.dataset_fp}")
    
    # Check for samples_npy
    ds_dir = os.path.dirname(cfg.dataset.dataset_fp)
    snpy_dir = os.path.join(ds_dir, "samples_npy")
    precomputed_dists = None
    
    if os.path.isdir(snpy_dir):
        print(f"Loading samples_npy from {snpy_dir}")
        train_sample_p = os.path.join(snpy_dir, "train_sample.npy")
        train_dists_p = os.path.join(snpy_dir, "train_dists.npy")
        native = getattr(cfg.training, "objective", "dtw") == "native"  # no DTW targets needed
        if os.path.isfile(train_sample_p) and (native or os.path.isfile(train_dists_p)):
            X_train = np.load(train_sample_p).astype(np.float32)
            train_dists_compact = None if native else np.load(train_dists_p).astype(np.float32)
            precomputed_dists = (compact_to_square(train_dists_compact, len(X_train))
                                 if getattr(cfg.training, "objective", "dtw") == "dtw" else None)
            y_train = np.zeros(len(X_train)) # Not really needed for DTW imitation
            print(f"Using precomputed distances for {len(X_train)} samples")
        else:
            # Fallback to NPZ
            data = np.load(cfg.dataset.dataset_fp)
            X_train = sanitize_time_series(data["X_train"], cfg.dataset.dim_series)
            y_train = data["y_train"]
    else:
        # Fallback to NPZ
        data = np.load(cfg.dataset.dataset_fp)
        X_train = sanitize_time_series(data["X_train"], cfg.dataset.dim_series)
        y_train = data["y_train"]
    
    labels_fp = getattr(cfg.training, "labels_fp", None)
    if labels_fp:  # class labels for the native per-class objective
        y_lab = np.load(labels_fp)
        if y_lab.dtype == np.float32 and np.abs(y_lab).max() < 1e-30:
            y_lab = y_lab.view(np.int32)  # int32 class ids stored in a float32 container
        y_train = y_lab.astype(np.int64)
        assert len(y_train) == len(X_train), (len(y_train), len(X_train))
        print(f"class labels: {np.unique(y_train).size} classes", flush=True)

    # Reshape for RF-DTAN (N, C, L)
    X_train = X_train.reshape(X_train.shape[0], 1, -1)
    
    # Use indices in Dataset to lookup precomputed distances
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], idx

    dataset = IndexedDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Model
    tess = int(cfg.model.tess_size[0]) if hasattr(cfg.model.tess_size, "__iter__") else int(cfg.model.tess_size)
        
    model = RFDTAN(
        signal_len=cfg.dataset.dim_series,
        channels=1,
        tess=tess,
        n_recurrence=cfg.model.n_recurrences,
        zero_boundary=cfg.model.zero_boundary,
        device=device,
        backbone=cfg.model.backbone,
        output_dim=cfg.model.dim_emb
    ).to(device)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    loss_fn = ScaledPairwiseDistDiff(dim_series=cfg.dataset.dim_series, dim_emb=cfg.model.dim_emb, device=device, precomputed_dists=precomputed_dists)

    os.makedirs(cfg.paths.exps_dir, exist_ok=True)
    exp_path = os.path.join(cfg.paths.exps_dir, cfg.dataset.dataset_name)
    os.makedirs(exp_path, exist_ok=True)

    wandb_mode = getattr(cfg.system, "wandb_mode", "online")
    use_wandb = wandb_mode != "disabled"
    if use_wandb:
        wandb.init(
            project=getattr(cfg.system, "wandb_project", "dtwrl_baselines"),
            name=f"rfdtan_{cfg.dataset.dataset_name}",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
            reinit=True,
        )

    objective = getattr(cfg.training, "objective", "dtw")  # "dtw" (DTW-preserving) or "native"
    native_losses = {"ICAE_loss": ICAE()}
    spread_w = float(getattr(cfg.training, "spread_weight", 0.0))
    print(f"spread_weight: {spread_w}")
    prior = PriorLoss(model.T, lambda_smooth=0.5, lambda_var=0.01) if getattr(cfg.training, "smooth_prior", False) else None
    print(f"smoothness prior: {prior is not None}", flush=True)
    print(f"objective: {objective}")
    model.train()
    best_loss = float('inf')
    wall_start = time.time()
    time_limit_h = float(getattr(cfg.training, "time_limit_hours", 0))
    time_limit_sec = time_limit_h * 3600 if time_limit_h > 0 else None
    stopped_by_time_limit = False

    # resume: epoch-level checkpoint; the elapsed training time counts against the time budget
    ckpt_fp = os.path.join(exp_path, "resume.pt")
    start_epoch = 0
    if os.path.exists(ckpt_fp):
        ck = torch.load(ckpt_fp, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng"].cpu())  # saved state may load onto the GPU (map_location)
        start_epoch, best_loss = ck["epoch"], ck["best_loss"]
        wall_start = time.time() - ck["elapsed"]
        print(f"Resumed from epoch {start_epoch} ({ck['elapsed']:.0f}s of budget used)")

    for epoch in range(start_epoch, cfg.training.n_epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.n_epochs}")
        for batch_x, batch_y, batch_idx in pbar:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            if objective == "native":  # upstream RF-DTAN objective: ICAE, one class (label-free)
                yb = batch_y.long() if labels_fp else torch.zeros(len(batch_x), dtype=torch.long)
                loss = one_batch(batch_x, yb, model, device,
                                 native_losses, {"ICAE_loss": 1.0})["loss"]  # upstream config weight
            else:
                outputs = model(batch_x)
                embeddings = outputs["embedding"]
                loss = loss_fn(embeddings, batch_x, batch_indices=batch_idx, warping_window=cfg.dataset.warping_window)
                if prior is not None:  # RF-DTAN's own smoothness prior on the warps
                    loss = loss + prior(thetas=outputs["theta"])
                if spread_w > 0:  # optional spread regulariser (effective-dimensionality test)
                    loss = loss + spread_w * spread_penalty(embeddings)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        if use_wandb:
            wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(exp_path, "best_model.pth"))
            # Also save config
            with open(os.path.join(exp_path, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                    "epoch": epoch + 1, "best_loss": best_loss, "elapsed": time.time() - wall_start}, ckpt_fp + ".tmp")
        os.replace(ckpt_fp + ".tmp", ckpt_fp)

        if time_limit_sec is not None and (time.time() - wall_start) >= time_limit_sec:
            stopped_by_time_limit = True
            torch.save(model.state_dict(), os.path.join(exp_path, "last_model.pth"))
            print(f"Time limit reached ({time_limit_h}h). Stopping.")
            break

    meta = {
        "dataset": cfg.dataset.dataset_name,
        "best_loss": float(best_loss),
        "stopped_by_time_limit": stopped_by_time_limit,
        "time_limit_sec": time_limit_sec,
        "train_time_sec": float(time.time() - wall_start),
    }
    with open(os.path.join(exp_path, "train_meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)
    if use_wandb:
        wandb.log({"best_loss": float(best_loss), "stopped_by_time_limit": int(stopped_by_time_limit)})
        wandb.finish()

if __name__ == "__main__":
    main()
