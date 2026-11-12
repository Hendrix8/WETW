#!/usr/bin/env python3
"""
CDPS GPU — Optimized version with vectorized inner loops and GPU support.
Train + Transform both GPU-accelerated.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn.functional as F
from typing import Optional


class CDPS_GPU(BaseEstimator, TransformerMixin):
    """
    GPU-accelerated CDPS with vectorized inner loops.

    Parameters
    ----------
    n_shapelets : int
    shapelet_length : float or int (fraction of series length if < 1)
    max_iter : int
    learning_rate : float
    lambda_ortho : float
    lambda_reg : float
    device : str ("cuda" or "cpu")
    """
    def __init__(self, n_shapelets: int = 10, shapelet_length: float = 0.2,
                 max_iter: int = 500, learning_rate: float = 1e-2,
                 lambda_ortho: float = 0.1, lambda_reg: float = 1e-4,
                 use_fixed_length: bool = False, random_state: int = 42,
                 device: str = "cuda", stride: int = 1):
        self.n_shapelets = n_shapelets
        self.shapelet_length = shapelet_length
        self.max_iter = max_iter
        self.stride = stride
        self.learning_rate = learning_rate
        self.lambda_ortho = lambda_ortho
        self.lambda_reg = lambda_reg
        self.use_fixed_length = use_fixed_length
        self.random_state = random_state
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.shapelets_ = None
        self._fitted = False

    def _resolve_length(self, series_length: int) -> int:
        if self.use_fixed_length:
            return int(self.shapelet_length)
        else:
            return max(3, int(series_length * self.shapelet_length))

    def _initialize_shapelets(self, X: np.ndarray, shapelet_len: int):
        rng = np.random.RandomState(self.random_state)
        n, L = X.shape
        shapelets = []
        for _ in range(self.n_shapelets):
            idx = rng.randint(0, n)
            start = rng.randint(0, max(1, L - shapelet_len))
            s = X[idx, start:start+shapelet_len].copy()
            std = np.std(s)
            if std > 1e-10:
                s = (s - np.mean(s)) / std
            else:
                s = s - np.mean(s)
            shapelets.append(s)
        return np.array(shapelets, dtype=np.float32)

    def _batch_shapelet_dist_gpu(self, shapelets_t: torch.Tensor,
                                   X_norm: torch.Tensor) -> torch.Tensor:
        """
        Vectorized shapelet distance.
        shapelets_t: (n_shapelets, shapelet_len)
        X_norm: (n, L) already z-normalized
        Returns: (n, n_shapelets) min distances
        """
        n, L = X_norm.shape
        n_s, slen = shapelets_t.shape
        if L < slen:
            return torch.full((n, n_s), float("inf"), device=X_norm.device)

        # Normalize shapelets
        s_mean = shapelets_t.mean(dim=1, keepdim=True)
        s_std = shapelets_t.std(dim=1, keepdim=True) + 1e-10
        s_norm = (shapelets_t - s_mean) / s_std  # (n_s, slen)

        # Unfold all series: (n, N_windows, slen)
        windows = X_norm.unfold(1, slen, self.stride)  # (n, N, slen)
        N = windows.shape[1]

        # Compute min distances for each shapelet
        D_s = torch.zeros(n, n_s, device=X_norm.device)
        for k in range(n_s):
            # (n, N, slen) - (1, 1, slen) -> (n, N, slen)
            sq = torch.sum((windows - s_norm[k:k+1, None, :]) ** 2, dim=2)  # (n, N)
            D_s[:, k] = torch.sqrt(sq.min(dim=1).values)
        return D_s

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            D_dtw: Optional[np.ndarray] = None) -> "CDPS_GPU":
        n, L = X.shape
        shapelet_len = self._resolve_length(L)
        device = self.device

        if D_dtw is None:
            from dtaidistance import dtw
            window = max(1, int(0.1 * L))
            D_dtw = dtw.distance_matrix_fast(
                X.astype(np.float64), compact=False, parallel=True, window=window
            )

        init_shapelets = self._initialize_shapelets(X, shapelet_len)

        # Move to device
        shapelets_t = torch.nn.Parameter(
            torch.from_numpy(init_shapelets).float().to(device), requires_grad=True
        )
        X_t = torch.from_numpy(X).float().to(device)
        D_dtw_t = torch.from_numpy(D_dtw).float().to(device)

        # Pre-normalize X
        X_mean = X_t.mean(dim=1, keepdim=True)
        X_std = X_t.std(dim=1, keepdim=True) + 1e-10
        X_norm = (X_t - X_mean) / X_std

        optimizer = torch.optim.Adam([shapelets_t], lr=self.learning_rate)
        best_loss = float("inf")
        best_shapelets = init_shapelets.copy()

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            D_s = self._batch_shapelet_dist_gpu(shapelets_t, X_norm)
            D_pred = torch.cdist(D_s, D_s, p=2)

            mask_t = D_dtw_t > 1e-10
            if mask_t.sum() > 0:
                scale = (D_dtw_t[mask_t] * D_pred[mask_t]).sum() / \
                        (D_pred[mask_t] ** 2).sum() + 1e-10
            else:
                scale = 1.0

            loss_dtw = F.mse_loss(D_pred * scale, D_dtw_t)

            # Orthogonality penalty
            S_norm = shapelets_t / (shapelets_t.norm(dim=1, keepdim=True) + 1e-10)
            gram = S_norm @ S_norm.T
            off_diag = gram - torch.eye(self.n_shapelets, device=device)
            loss_ortho = (off_diag ** 2).sum() / max(1, self.n_shapelets * (self.n_shapelets - 1))

            loss_reg = (shapelets_t ** 2).mean()

            loss = loss_dtw + self.lambda_ortho * loss_ortho + self.lambda_reg * loss_reg
            loss.backward()
            optimizer.step()

            if iteration % 50 == 0:
                print(f"  [CDPS] iter={iteration:4d} loss={loss.item():.4f} "
                      f"dtw={loss_dtw.item():.4f} ortho={loss_ortho.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_shapelets = shapelets_t.detach().cpu().numpy().copy()

        self.shapelets_ = best_shapelets
        self._fitted = True
        self._series_length = L
        self._shapelet_length = shapelet_len
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("CDPS not fitted yet")
        return self._transform_gpu(X)

    def _transform_gpu(self, X: np.ndarray) -> np.ndarray:
        """GPU-accelerated transform."""
        device = self.device
        X_t = torch.from_numpy(X).float().to(device)

        # Normalize
        X_mean = X_t.mean(dim=1, keepdim=True)
        X_std = X_t.std(dim=1, keepdim=True) + 1e-10
        X_norm = (X_t - X_mean) / X_std

        # Shapelets already on CPU; move to GPU
        shapelets_t = torch.from_numpy(self.shapelets_).float().to(device)
        D_s = self._batch_shapelet_dist_gpu(shapelets_t, X_norm)
        return D_s.cpu().numpy()

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                       D_dtw: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y, D_dtw)
        return self.transform(X)
