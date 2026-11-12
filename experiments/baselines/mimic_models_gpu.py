import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import logging
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

class LDPS_GPU(nn.Module):
    def __init__(self, shapelet_lengths, d=1, convergence_rate=0.01, device='cuda',
                 init_method="kmeans", print_loss_every=1000, stride=1):
        super(LDPS_GPU, self).__init__()
        self.shapelet_lengths = shapelet_lengths # dict {length: count}
        self.d = d
        self.convergence_rate = convergence_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.init_method = init_method
        self.print_loss_every = print_loss_every
        
        self.sorted_lengths = sorted(shapelet_lengths.keys())
        self.n_shapelets = sum(shapelet_lengths.values())
        self.stride = stride
        
        # Parameters
        self.shapelets = nn.ParameterList()
        for slen in self.sorted_lengths:
            count = shapelet_lengths[slen]
            # Initialize with zeros, will be overwritten by fit/init_shapelets
            shp = nn.Parameter(torch.randn(count, slen, d))
            self.shapelets.append(shp)
            
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.to(self.device)
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def shapelet_transform(self, X):
        """
        X: (Batch, T, D)
        returns: (Batch, n_shapelets)
        """
        batch_size, T, D = X.shape
        all_transforms = []
        
        x = X.transpose(1, 2)  # (N, D, T)
        for i, slen in enumerate(self.sorted_lengths):
            # shapelets[i]: (count, slen, D)
            shp = self.shapelets[i]

            # Squared distance to every window as ||w||^2 - 2 w.s + ||s||^2 (w.s by conv1d): same values as
            # the (window - shapelet) difference, but O(N * windows * count) memory instead of
            # O(N * windows * count * slen * D), which ran out of GPU memory on long series
            ws = nn.functional.conv1d(x, shp.permute(0, 2, 1), stride=self.stride)  # (N, count, num_windows)
            w2 = nn.functional.conv1d(x * x, torch.ones(1, D, slen, device=x.device, dtype=x.dtype), stride=self.stride)
            s2 = (shp ** 2).sum(dim=(1, 2))[None, :, None]
            dist_sq = (w2 - 2 * ws + s2).clamp_min(0).transpose(1, 2) / slen  # (N, num_windows, count)

            # Min over windows: (N, count)
            min_dist_sq, _ = dist_sq.min(dim=1)
            all_transforms.append(min_dist_sq)
            
        return torch.cat(all_transforms, dim=1)

    def forward_dist(self, Xi, Xj):
        """
        Computes predicted distance between Xi and Xj
        Xi, Xj: (Batch, T, D)
        """
        ti = self.shapelet_transform(Xi)
        tj = self.shapelet_transform(Xj)
        # Prediction: beta * ||ti - tj||_2
        # Note: original LDPS uses beta * ||ST(Xi) - ST(Xj)||_2
        # where ST(Xi) is (n_shapelets,)
        dist = torch.norm(ti - tj, p=2, dim=1)
        return self.beta * dist

    def init_shapelets(self, X_train):
        """
        X_train: (N_train, T, D) - numpy array
        """
        if self.init_method == "random":
            for i, slen in enumerate(self.sorted_lengths):
                count = self.shapelet_lengths[slen]
                self.shapelets[i].data = torch.randn(count, slen, self.d).to(self.device)
        elif self.init_method == "kmeans":
            n_draw = 10000
            n_ts, sz, d = X_train.shape
            for i, slen in enumerate(self.sorted_lengths):
                count = self.shapelet_lengths[slen]
                indices_ts = np.random.choice(n_ts, size=n_draw, replace=True)
                indices_time = np.random.choice(sz - slen + 1, size=n_draw, replace=True)
                subseries = np.zeros((n_draw, slen, d))
                for j in range(n_draw):
                    subseries[j] = X_train[indices_ts[j], indices_time[j]:indices_time[j] + slen]
                subseries = subseries.reshape((n_draw, slen * d))
                kmeans_centers = KMeans(n_clusters=count, n_init=10).fit(subseries).cluster_centers_
                kmeans_centers = kmeans_centers.reshape((count, slen, d))
                self.shapelets[i].data = torch.from_numpy(kmeans_centers).float().to(self.device)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

    def fit(
        self,
        X_train,
        y_matrix=None,
        niter=100000,
        batch_size=32,
        lr=None,
        time_limit_sec=None,
        use_wandb=False,
        wandb_project_name="",
        wandb_team_name="",
        wandb_run_name="",
        wandb_mode="online",
        wandb_log_every=100,
        wandb_config=None,
    ):
        """
        X_train: (N, T, D) numpy array
        y_matrix: (N, N) precomputed DTW distances
        time_limit_sec: if set, stop training after this many seconds and save the model
        """
        if lr is None:
            lr = self.convergence_rate
            
        self.init_shapelets(X_train)
        
        optimizer = optim.Adagrad(self.parameters(), lr=lr)
        
        X_tensor = torch.from_numpy(X_train).float().to(self.device)
        N = X_train.shape[0]
        
        if y_matrix is not None:
            y_matrix = torch.from_numpy(y_matrix).float().to(self.device)

        wandb_module = None
        if use_wandb:
            if not wandb_project_name:
                raise ValueError("wandb_project_name must be provided when use_wandb=True")
            try:
                import wandb as wandb_module
            except Exception as exc:
                raise RuntimeError("Failed to import wandb. Install wandb or disable --use_wandb") from exc
            cfg = {
                "niter": niter,
                "batch_size": batch_size,
                "lr": lr,
                "device": str(self.device),
                "n_shapelets": self.n_shapelets,
                "shapelet_lengths": dict(self.shapelet_lengths),
                "time_limit_sec": time_limit_sec,
            }
            if wandb_config:
                cfg.update(wandb_config)
            init_kwargs = {"project": wandb_project_name, "config": cfg, "mode": wandb_mode}
            if wandb_team_name:
                init_kwargs["entity"] = wandb_team_name
            wandb_module.init(**init_kwargs)
            if wandb_run_name:
                wandb_module.run.name = wandb_run_name

        self.train()
        wall_start = time.time()
        t0 = wall_start
        self.training_completed = False
        self.iterations_completed = 0
        self.training_stopped_by_time_limit = False
        try:
            for it in range(niter):
                idx_i = torch.randint(0, N, (batch_size,), device=self.device)
                idx_j = torch.randint(0, N, (batch_size,), device=self.device)

                mask = idx_i == idx_j
                while mask.any():
                    idx_j[mask] = torch.randint(0, N, (mask.sum(),), device=self.device)
                    mask = idx_i == idx_j

                if y_matrix is not None:
                    y_targets = y_matrix[idx_i, idx_j]
                else:
                    raise ValueError("y_matrix (precomputed DTW) must be provided for GPU training")

                optimizer.zero_grad()
                y_hat = self.forward_dist(X_tensor[idx_i], X_tensor[idx_j])

                loss = nn.MSELoss()(y_hat, y_targets)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    self.beta.clamp_(min=1e-6)

                self.iterations_completed = it + 1

                if wandb_module is not None and ((it + 1) % max(1, wandb_log_every) == 0 or it == 0):
                    wandb_module.log(
                        {
                            "iteration": it + 1,
                            "loss": float(loss.item()),
                            "beta": float(self.beta.item()),
                            "lr": float(lr),
                        }
                    )

                if (it + 1) % self.print_loss_every == 0:
                    elapsed = time.time() - t0
                    print(f"Iteration {it+1}/{niter}, Loss: {loss.item():.6f}, Beta: {self.beta.item():.4f}, Time: {elapsed:.2f}s")
                    t0 = time.time()

                if time_limit_sec is not None and (it + 1) % self.print_loss_every == 0:
                    wall_elapsed = time.time() - wall_start
                    if wall_elapsed >= time_limit_sec:
                        self.training_stopped_by_time_limit = True
                        self.logger.warning(
                            "Time limit reached (%.1f / %.1f sec) at iteration %d/%d. Stopping training.",
                            wall_elapsed, time_limit_sec, it + 1, niter,
                        )
                        break

            else:
                self.training_completed = True
        finally:
            if wandb_module is not None:
                wandb_module.finish()

    @torch.no_grad()
    def embed(self, X, batch_size=128):
        self.eval()
        N = X.shape[0]
        X_tensor = torch.from_numpy(X).float().to(self.device)
        embs = []
        for i in range(0, N, batch_size):
            batch_X = X_tensor[i:i+batch_size]
            emb = self.shapelet_transform(batch_X)
            # Apply beta scaling if needed? 
            # In original script, _embed returns raw ST, and beta is used in pred.
            # But original LDPS's ST includes beta? 
            # Let's check mimic_models.py:388
            # _shapelet_transform returns ret[k] = norm**2 / sz
            # _compute_approximate_dist returns beta * norm(STi - STj)
            # So beta is NOT in the transform.
            embs.append(emb.cpu().numpy())
        return np.concatenate(embs, axis=0)

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'shapelet_lengths': self.shapelet_lengths,
            'd': self.d,
            'convergence_rate': self.convergence_rate,
            'init_method': self.init_method,
            'stride': self.stride,
            'training_completed': getattr(self, 'training_completed', None),
            'iterations_completed': getattr(self, 'iterations_completed', None),
            'training_stopped_by_time_limit': getattr(self, 'training_stopped_by_time_limit', False),
        }, path)

    @classmethod
    def load(cls, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(shapelet_lengths=checkpoint['shapelet_lengths'], 
                    d=checkpoint['d'], 
                    convergence_rate=checkpoint['convergence_rate'],
                    device=device,
                    init_method=checkpoint['init_method'],
                    stride=checkpoint.get('stride', 1))
        model.load_state_dict(checkpoint['state_dict'])
        model.training_completed = checkpoint.get('training_completed', None)
        model.iterations_completed = checkpoint.get('iterations_completed', None)
        model.training_stopped_by_time_limit = checkpoint.get('training_stopped_by_time_limit', False)
        return model
