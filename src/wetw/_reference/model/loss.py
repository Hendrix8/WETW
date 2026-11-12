# coding = utf-8

import logging
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from wetw._reference.util.helper_functions import (
    weighted_pairwise_distance,
    wetw_distance_kwargs_from_cfg,
    training_scale_dists_enabled,
)


class PairwiseDistDiff(torch.nn.Module):
    def __init__(self, cfg, dataset_size, default_batch_size, device='cuda', logger=None):
        super(PairwiseDistDiff, self).__init__()

        self.cfg = cfg
        # Use provided logger or get existing logger (don't reconfigure logging)
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger
        
        self.__device = device

        self.__dataset_size = dataset_size
        self.__targets_size = int(
            self.__dataset_size * (self.__dataset_size - 1) / 2)

        self.__l2 = torch.nn.PairwiseDistance(p=2).to(device)
        self.__l1 = torch.nn.PairwiseDistance(p=1).to(device)
        self.train_all_pairs = cfg.training.train_all_pairs
        
        # Initialize augment_data attribute (default to False if not in config)
        self.augment_data = getattr(cfg.training, 'augment_data', False)

        self.__dist_targets_np = None

        self.__default_batch_size = default_batch_size
        self.__pair_indices_array = [
            None for _ in range(default_batch_size + 1)]
        self.__num_samples_array = np.full(default_batch_size + 1, -1, dtype=int)

        self.__epsilon = 1e-3

        self.__num_example2check = 5

        # Distribution control
        self.lb_penalty_weight = getattr(cfg.training, 'lb_penalty_weight', 0.0)
        self.underestimation_penalty_weight = getattr(cfg.training, 'underestimation_penalty_weight', 0.0)
        self.variance_penalty_weight = getattr(cfg.training, 'variance_penalty_weight', 0.0)

        # Primary distance loss: l1 (default) | relative_l1 | relative_l2 | relative_huber
        self.distance_loss_type = getattr(cfg.training, 'distance_loss_type', 'l1')
        self.relative_error_epsilon = float(getattr(cfg.training, 'relative_error_epsilon', 1e-6))
        self.relative_loss_huber_delta = float(getattr(cfg.training, 'relative_loss_huber_delta', 0.1))
        self.relative_error_batch_var_weight = float(
            getattr(cfg.training, 'relative_error_batch_var_weight', 0.0)
        )
        # Optional: pull batch mean of relative errors toward 0 and reduce Var(r); uses same r as eval
        self.pairwise_rel_error_epsilon = float(
            getattr(cfg.training, 'pairwise_rel_error_epsilon', getattr(cfg.training, 'relative_error_epsilon', 1e-6))
        )
        self.pairwise_rel_error_mean_penalty_weight = float(
            getattr(cfg.training, 'pairwise_rel_error_mean_penalty_weight', 0.0)
        )
        self.pairwise_rel_error_var_penalty_weight = float(
            getattr(cfg.training, 'pairwise_rel_error_var_penalty_weight', 0.0)
        )

        # WETW only: discourage collapsed weights (near-constant w across pairs in the batch).
        self.wetw_weight_diversity_weight = float(
            getattr(cfg.training, 'wetw_weight_diversity_weight', 0.0)
        )
        self.wetw_weight_diversity_mode = str(
            getattr(cfg.training, 'wetw_weight_diversity_mode', 'shortfall')
        ).lower()
        self.wetw_weight_diversity_floor = float(
            getattr(cfg.training, 'wetw_weight_diversity_floor', 1e-6)
        )
        self.wetw_weight_diversity_neglog_cap = float(
            getattr(cfg.training, 'wetw_weight_diversity_neglog_cap', 20.0)
        )

        self.pred_distance_learned_reg_weight = float(
            getattr(cfg.training, 'pred_distance_learned_reg_weight', 0.0)
        )

        # Optional rank-aware term (default off): hinge on triplets (anchor a, pos j, neg k)
        # where DTW targets satisfy T_aj + dtw_margin < T_ak. Encourages pred_aj < pred_ak.
        self.pairwise_rank_margin_weight = float(
            getattr(cfg.training, "pairwise_rank_margin_weight", 0.0)
        )
        self.pairwise_rank_dtw_margin = float(
            getattr(cfg.training, "pairwise_rank_dtw_margin", 0.0)
        )
        self.pairwise_rank_pred_margin = float(
            getattr(cfg.training, "pairwise_rank_pred_margin", 0.05)
        )
        self.pairwise_rank_max_triplets = int(
            getattr(cfg.training, "pairwise_rank_max_triplets", 64)
        )
        self.pairwise_rank_sample_multiplier = int(
            getattr(cfg.training, "pairwise_rank_sample_multiplier", 40)
        )

        # Optional learned pair residual d' = d + scale * tanh(MLP([z_i,z_j])) (default off).
        self.wetw_query_pair_residual_scale = float(
            getattr(cfg.training, "wetw_query_pair_residual_scale", 0.1)
        )

        # Optional listwise KL: match softmax(-tau*DTW) over in-batch candidates to softmax(-tau*pred).
        self.listwise_simsearch_weight = float(
            getattr(cfg.training, "listwise_simsearch_weight", 0.0)
        )
        self.listwise_simsearch_temperature = float(
            getattr(cfg.training, "listwise_simsearch_temperature", 1.0)
        )
        self.listwise_simsearch_n_anchors = int(
            getattr(cfg.training, "listwise_simsearch_n_anchors", 4)
        )
        self.listwise_simsearch_n_candidates = int(
            getattr(cfg.training, "listwise_simsearch_n_candidates", 24)
        )

    def _targets_from_global_pairs(self, gi: torch.Tensor, gj: torch.Tensor) -> torch.Tensor:
        """Upper-triangle compact DTW targets for pairs (gi, gj) (same shape, int indices)."""
        gi = gi.reshape(-1).to(torch.int64)
        gj = gj.reshape(-1).to(torch.int64)
        ds = int(self.__dataset_size)
        ts = int(self.__targets_size)
        global_row_idx = torch.minimum(gi, gj)
        global_column_idx = torch.maximum(gi, gj)
        targets_cutoff_rows = ds - global_row_idx
        targets_cutoff_sizes = torch.bitwise_right_shift(
            targets_cutoff_rows * (targets_cutoff_rows - 1), 1
        )
        targets_row_offsets = global_column_idx - global_row_idx - 1
        target_dist_idx = targets_row_offsets + ts - targets_cutoff_sizes
        idx_cpu = target_dist_idx.detach().cpu().numpy().astype(np.int64)
        vals = self.__dist_targets_np[idx_cpu]
        return torch.as_tensor(vals, dtype=torch.float32, device=self.__device)

    def _pred_for_embedding_pairs(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        wetw_model,
        approximator: str,
        _wdk: dict,
        pred_dist_calib,
        query_pair_residual=None,
    ) -> torch.Tensor:
        """Predicted distance for each row pair (left[i], right[i]); same scaling/calib as main path."""
        if approximator == "euclidean":
            d = self.__l2(left, right)
        else:
            w = wetw_model(left, right)
            d = weighted_pairwise_distance(left, right, w=w, **_wdk)
        if training_scale_dists_enabled(self.cfg):
            dim_emb = int(self.cfg.encoder_model.dim_emb)
            d = d / (dim_emb ** 0.5)
        if pred_dist_calib is not None:
            d = pred_dist_calib(d)
        if (
            approximator != "euclidean"
            and query_pair_residual is not None
            and self.wetw_query_pair_residual_scale > 0.0
        ):
            d = d + self.wetw_query_pair_residual_scale * query_pair_residual(left, right)
        return d

    def _pairwise_rank_triplet_loss(
        self,
        emb_all: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
        wetw_model,
        approximator: str,
        _wdk: dict,
        pred_dist_calib,
        query_pair_residual=None,
    ) -> torch.Tensor:
        """Mean hinge max(0, pred_aj - pred_ak + pred_margin) on DTW-valid triplets (see __init__)."""
        T = int(self.pairwise_rank_max_triplets)
        mult = max(5, int(self.pairwise_rank_sample_multiplier))
        M = min(T * mult, 5000)
        device = emb_all.device
        idx = batch_indices.to(device=device, dtype=torch.long).reshape(-1)
        if idx.numel() != batch_size:
            return torch.tensor(0.0, device=device, dtype=emb_all.dtype)

        a = torch.randint(0, batch_size, (M,), device=device)
        j = torch.randint(0, batch_size, (M,), device=device)
        k = torch.randint(0, batch_size, (M,), device=device)
        ok = (a != j) & (a != k) & (j != k)
        a, j, k = a[ok], j[ok], k[ok]
        if a.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=emb_all.dtype)

        ga = idx[a]
        gj = idx[j]
        gk = idx[k]
        tgt_aj = self._targets_from_global_pairs(ga, gj)
        tgt_ak = self._targets_from_global_pairs(ga, gk)
        dtw_m = float(self.pairwise_rank_dtw_margin)
        valid = (tgt_aj + dtw_m < tgt_ak)
        if not torch.any(valid):
            return torch.tensor(0.0, device=device, dtype=emb_all.dtype)
        a, j, k = a[valid], j[valid], k[valid]
        tgt_aj, tgt_ak = tgt_aj[valid], tgt_ak[valid]
        if a.shape[0] > T:
            sel = torch.randperm(a.shape[0], device=device)[:T]
            a, j, k = a[sel], j[sel], k[sel]
            tgt_aj, tgt_ak = tgt_aj[sel], tgt_ak[sel]

        left_a = emb_all[a]
        pred_aj = self._pred_for_embedding_pairs(
            left_a, emb_all[j], wetw_model, approximator, _wdk, pred_dist_calib, query_pair_residual
        )
        pred_ak = self._pred_for_embedding_pairs(
            left_a, emb_all[k], wetw_model, approximator, _wdk, pred_dist_calib, query_pair_residual
        )
        m_pred = float(self.pairwise_rank_pred_margin)
        hinge = torch.relu(pred_aj - pred_ak + m_pred)
        return hinge.mean()

    def _wetw_weight_diversity_loss(self, warp_weights: torch.Tensor) -> torch.Tensor:
        """Penalty when per-dimension variance of weights across pairs is too low (collapse)."""
        if warp_weights is None:
            return torch.tensor(0.0, device=self.__device)
        if warp_weights.dim() < 2:
            return torch.tensor(0.0, device=warp_weights.device, dtype=warp_weights.dtype)
        p = warp_weights.shape[0]
        if p < 2:
            return torch.tensor(0.0, device=warp_weights.device, dtype=warp_weights.dtype)
        # Mean over embedding dims of Var_across_pairs(w[:, d]); unnormalized var for stability.
        var_per_dim = warp_weights.var(dim=0, unbiased=False)
        var_mean = var_per_dim.mean()
        mode = self.wetw_weight_diversity_mode
        if mode == 'shortfall':
            floor_t = torch.as_tensor(
                self.wetw_weight_diversity_floor, device=var_mean.device, dtype=var_mean.dtype
            )
            return torch.relu(floor_t - var_mean)
        if mode in ('neg_log', 'neglog'):
            # -log(v) is negative when v>1; that would *reduce* total loss (reward high var).
            # Keep a nonnegative penalty: stress only collapses (v small); no bonus for v>1.
            eps = 1e-12
            return torch.clamp(
                torch.relu(-torch.log(var_mean + eps)),
                max=self.wetw_weight_diversity_neglog_cap,
            )
        raise ValueError(
            f"Unknown training.wetw_weight_diversity_mode={mode!r}; use 'shortfall' or 'neg_log'."
        )

    def _listwise_simsearch_row_loss(
        self,
        emb_all: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
        wetw_model,
        approximator: str,
        _wdk: dict,
        pred_dist_calib,
        query_pair_residual,
    ) -> torch.Tensor:
        """Mean KL-like loss: soft targets from DTW row vs student softmax on predicted WETW (+residual)."""
        device = emb_all.device
        dtype = emb_all.dtype
        idx = batch_indices.to(device=device, dtype=torch.long).reshape(-1)
        if idx.numel() != batch_size:
            return torch.tensor(0.0, device=device, dtype=dtype)

        K = int(min(self.listwise_simsearch_n_candidates, batch_size - 1))
        if K < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        n_anchors = int(max(1, min(self.listwise_simsearch_n_anchors, batch_size)))
        tau = float(max(self.listwise_simsearch_temperature, 1e-6))
        total = torch.tensor(0.0, device=device, dtype=dtype)
        used = 0

        all_slots = torch.arange(batch_size, device=device)
        for _ in range(n_anchors):
            a = int(torch.randint(0, batch_size, (1,), device=device).item())
            mask = all_slots != a
            pool = all_slots[mask]
            if pool.numel() < 2:
                continue
            take = min(K, pool.numel())
            perm = torch.randperm(pool.numel(), device=device)[:take]
            slots = pool[perm]
            ga = idx[a].expand(take)
            gj = idx[slots]

            left = emb_all[a].unsqueeze(0).expand(take, -1).reshape(take, -1)
            right = emb_all[slots]
            dtw_vec = self._targets_from_global_pairs(ga, gj)
            pred_vec = self._pred_for_embedding_pairs(
                left, right, wetw_model, approximator, _wdk, pred_dist_calib, query_pair_residual
            )
            dtw_vec = dtw_vec - dtw_vec.min()
            pred_vec = pred_vec - pred_vec.min()
            p = F.softmax(-tau * dtw_vec, dim=0).detach()
            logq = F.log_softmax(-tau * pred_vec, dim=0)
            total = total - (p * logq).sum()
            used += 1

        if used == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return total / float(used)

    def _ensure_pair_indices(self, batch_size):
        """Lazily compute and cache pair indices for batch_size (avoids O(n³) init for large batch sizes)."""
        if batch_size < 2:
            return
        # Extend arrays if batch_size exceeds pre-allocated size (e.g. last batch)
        while batch_size >= len(self.__pair_indices_array):
            self.__pair_indices_array.append(None)
            self.__num_samples_array = np.append(
                self.__num_samples_array, -1)
        if self.__pair_indices_array[batch_size] is not None:
            return
        # Use np.triu_indices for O(n²) instead of Python nested loop
        i, j = np.triu_indices(batch_size, 1)
        self.__pair_indices_array[batch_size] = np.column_stack([i, j])
        num_samples = batch_size * int(np.ceil(np.log2(batch_size)))
        n_pairs = self.__pair_indices_array[batch_size].shape[0]
        self.__num_samples_array[batch_size] = min(num_samples, n_pairs)

    def update_targets(self, dist_ndarray):
        if self.cfg.training.target_dist_function == 'dtw':
            self.__dist_targets_np = np.sqrt(dist_ndarray)
        else:
            self.__dist_targets_np = dist_ndarray

        if training_scale_dists_enabled(self.cfg):
            dim_series = self.cfg.dataset.dim_series
            self.__dist_targets_np = self.__dist_targets_np / np.sqrt(dim_series)

        if int(self.__dataset_size * (self.__dataset_size - 1) / 2) != self.__dist_targets_np.shape[0] and not self.augment_data:
            self.logger.error('dataset_size={:d} - {:d}, dist_targets_np.shape[0]={:d}'.format(
                self.__dataset_size, int(self.__dataset_size * (self.__dataset_size - 1) / 2), self.__dist_targets_np.shape[0]))
            raise ValueError('mismatched targets size')

        if not np.isfinite(self.__dist_targets_np).all():
            raise ValueError(
                "Pairwise target distances contain NaN or Inf after preprocessing. "
                "Regenerate or fix train_dists.npy / valid_dists.npy (wrong format, corrupt file, or bad DTW cache)."
            )

        if training_scale_dists_enabled(self.cfg):
            ds = int(self.cfg.dataset.dim_series)
            tmin = float(np.min(self.__dist_targets_np))
            tmax = float(np.max(self.__dist_targets_np))
            msg = (
                f"Scaled original pairwise target distances (after ÷√dim_series={ds}): "
                f"min={tmin:.6g} max={tmax:.6g}"
            )
            self.logger.info(msg)
            print(msg)

        return self.__dist_targets_np
           
    def forward(self, batch, indices, wetw_model, pred_dist_calib=None, query_pair_residual=None):
        batch = torch.squeeze(batch)
        batch_emb = batch
        batch_size = batch.shape[0]

        # sampling pairs to calculate loss, instead of taking their (n 2) which would be intractable
        self._ensure_pair_indices(batch_size)
        sample_pair_idx = np.random.choice(self.__pair_indices_array[batch_size].shape[0],
                                           size=self.__num_samples_array[batch_size],
                                           replace=False)
        

        if self.train_all_pairs: 
            sample_idx_pairs = self.__pair_indices_array[batch_size]
        else:
            sample_idx_pairs = self.__pair_indices_array[batch_size][sample_pair_idx]

        left_sample_idx = sample_idx_pairs[:, 0]
        left_sample_global_idx = indices[left_sample_idx]
        right_sample_idx = sample_idx_pairs[:, 1]
        right_sample_global_idx = indices[right_sample_idx]

        global_row_idx = torch.minimum(
            left_sample_global_idx, right_sample_global_idx).to(torch.int64)
        global_column_idx = torch.maximum(
            left_sample_global_idx, right_sample_global_idx).to(torch.int64)

        targets_cutoff_rows = self.__dataset_size - global_row_idx

        targets_cutoff_sizes = torch.bitwise_right_shift(
            targets_cutoff_rows * (targets_cutoff_rows - 1), 1)
        targets_row_offsets = global_column_idx - global_row_idx - 1

        target_dist_idx = targets_row_offsets + \
            self.__targets_size - targets_cutoff_sizes
        
        if (torch.any(left_sample_global_idx >= self.__dataset_size) or torch.any(left_sample_global_idx < 0)) and not self.augment_data:
            raise ValueError("left_sample_global_idx contains invalid values")

        if (torch.any(right_sample_global_idx >= self.__dataset_size) or torch.any(right_sample_global_idx < 0)) and not self.augment_data:
            raise ValueError("right_sample_global_idx contains invalid values")


        try:
            batch_dist_targets = torch.Tensor(
                self.__dist_targets_np[target_dist_idx.cpu()]).to(self.__device)
        except IndexError as ie:
            self.logger.error('IndexError: {:s}'.format(str(ie)))
            self.logger.error('left_sample_global_idx: {:s}'.format(
                np.array2string(left_sample_global_idx.cpu().numpy(), separator=', ', suppress_small=True)))
            self.logger.error('right_sample_global_idx: {:s}'.format(
                np.array2string(right_sample_global_idx.cpu().numpy(), separator=', ', suppress_small=True)))
            self.logger.error('target_dist_idx: {:s}'.format(
                np.array2string(target_dist_idx.cpu().numpy(), separator=', ', suppress_small=True)))
            raise IndexError(str(ie))

        approximator = getattr(self.cfg.weight_model, 'approximator', 'wetw')
        _wdk = wetw_distance_kwargs_from_cfg(self.cfg) if approximator != 'euclidean' else {}
        warp_weights = None
        if approximator == 'euclidean':
            bleft, bright = batch[left_sample_idx], batch[right_sample_idx]
            batch_dist_embedded = self.__l2(bleft, bright)
        else:
            bleft, bright = batch[left_sample_idx], batch[right_sample_idx]
            warp_weights = wetw_model(bleft, bright)
            batch_dist_embedded = weighted_pairwise_distance(bleft, bright, w=warp_weights, **_wdk)

        if training_scale_dists_enabled(self.cfg):
            dim_emb = self.cfg.encoder_model.dim_emb
            batch_dist_embedded = batch_dist_embedded / (dim_emb ** 0.5)

        if pred_dist_calib is not None:
            batch_dist_embedded = pred_dist_calib(batch_dist_embedded)

        if (
            query_pair_residual is not None
            and self.wetw_query_pair_residual_scale > 0.0
            and approximator != "euclidean"
        ):
            bres_l = batch_emb[left_sample_idx]
            bres_r = batch_emb[right_sample_idx]
            batch_dist_embedded = batch_dist_embedded + self.wetw_query_pair_residual_scale * query_pair_residual(
                bres_l, bres_r
            )

        # Fix: divide by actual number of pairs used, not sample_pair_idx.shape[0]
        num_pairs_used = sample_idx_pairs.shape[0]
        loss = self._main_distance_loss(
            batch_dist_embedded, batch_dist_targets, num_pairs_used
        )

        # Optional rank-aware triplet hinge (training.pairwise_rank_margin_weight; default 0 = disabled).
        rw = self.pairwise_rank_margin_weight
        if (
            rw > 0.0
            and batch_size >= 3
            and not self.augment_data
            and self.__dist_targets_np is not None
        ):
            emb_all = batch_emb
            rank_term = self._pairwise_rank_triplet_loss(
                emb_all,
                indices,
                batch_size,
                wetw_model,
                approximator,
                _wdk,
                pred_dist_calib,
                query_pair_residual,
            )
            loss = loss + rw * rank_term

        lw = self.listwise_simsearch_weight
        if (
            lw > 0.0
            and batch_size >= 3
            and not self.augment_data
            and self.__dist_targets_np is not None
            and approximator != "euclidean"
            and wetw_model is not None
        ):
            emb_all = batch_emb
            list_term = self._listwise_simsearch_row_loss(
                emb_all,
                indices,
                batch_size,
                wetw_model,
                approximator,
                _wdk,
                pred_dist_calib,
                query_pair_residual,
            )
            loss = loss + lw * list_term

        # Optional: center mean relative error near 0 and reduce batch variance of r=(pred-target)/(target+eps)
        pm = self.pairwise_rel_error_mean_penalty_weight
        pv = self.pairwise_rel_error_var_penalty_weight
        if pm > 0.0 or pv > 0.0:
            eps_r = self.pairwise_rel_error_epsilon
            denom = batch_dist_targets + eps_r
            r = (batch_dist_embedded - batch_dist_targets) / denom
            if pm > 0.0:
                loss = loss + pm * (torch.mean(r) ** 2)
            if pv > 0.0 and r.numel() > 1:
                loss = loss + pv * torch.var(r)

        # Lower-bound penalty: squared hinge on violations (predicted > target)
        # High = tight bound for pruning (few violations), more underestimation
        if self.lb_penalty_weight > 0:
            violations = torch.clamp(batch_dist_embedded - batch_dist_targets, min=0)
            lb_penalty = torch.sum(violations ** 2) / num_pairs_used
            loss = loss + self.lb_penalty_weight * lb_penalty

        # Underestimation penalty: squared hinge when pred < target
        # > 0 pushes predictions up to reduce mean negative relative error
        if self.underestimation_penalty_weight > 0:
            underest = torch.clamp(batch_dist_targets - batch_dist_embedded, min=0)
            underest_penalty = torch.sum(underest ** 2) / num_pairs_used
            loss = loss + self.underestimation_penalty_weight * underest_penalty

        # Variance penalty: tighten the error distribution within batch
        if self.variance_penalty_weight > 0:
            errors = batch_dist_embedded - batch_dist_targets
            var_penalty = torch.var(errors)
            loss = loss + self.variance_penalty_weight * var_penalty

        # WETW anti-collapse: encourage non-trivial spread of w across pairs (same batch).
        if (
            self.wetw_weight_diversity_weight > 0.0
            and approximator != 'euclidean'
            and wetw_model is not None
            and warp_weights is not None
        ):
            loss = loss + self.wetw_weight_diversity_weight * self._wetw_weight_diversity_loss(
                warp_weights
            )

        if self.pred_distance_learned_reg_weight > 0.0 and pred_dist_calib is not None:
            reg = torch.tensor(0.0, device=self.__device, dtype=batch_dist_embedded.dtype)
            for p in pred_dist_calib.parameters():
                reg = reg + (p ** 2).sum()
            loss = loss + self.pred_distance_learned_reg_weight * reg

        return loss

    def _main_distance_loss(self, pred, target, num_pairs_used):
        """Mean loss over sampled pairs. `l1` matches original mean L1 distance; relative_* use (pred-target)/(target+eps)."""
        lt = (self.distance_loss_type or 'l1').lower()
        if lt == 'l1':
            return self.__l1(pred, target) / num_pairs_used

        eps = self.relative_error_epsilon
        denom = target + eps
        rel = (pred - target) / denom

        if lt == 'relative_l1':
            main = torch.mean(torch.abs(rel))
        elif lt == 'relative_l2':
            main = torch.mean(rel ** 2)
        elif lt == 'relative_huber':
            beta = self.relative_loss_huber_delta
            main = F.smooth_l1_loss(
                rel, torch.zeros_like(rel), beta=beta, reduction='mean'
            )
        else:
            raise ValueError(
                f"Unknown training.distance_loss_type={lt!r}. "
                "Use one of: l1, relative_l1, relative_l2, relative_huber."
            )

        if self.relative_error_batch_var_weight > 0.0 and rel.numel() > 1:
            main = main + self.relative_error_batch_var_weight * torch.var(rel)
        return main
