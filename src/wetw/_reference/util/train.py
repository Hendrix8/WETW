# coding = utf-8

import math
import os
import logging
from timeit import default_timer as timer
from wetw._reference.util.timing_utils import compute_timer, compute_time_start, compute_time_elapsed
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import sys
from dtaidistance import dtw
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
try:
    from tslearn.metrics import dtw_path
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    dtw_path = None
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm

from wetw._reference.util.sampleSEAsamEMB import Sampler
from wetw._reference.util.data import DatasetwithIdx
from wetw._reference.util.produce_embeddings import save_embeddings_from_model
from wetw._reference.model.builder import ModelBuilder
from wetw._reference.model.loss import PairwiseDistDiff
from wetw._reference.model.wetw_model import PairConvModel, pair_conv_kwargs_from_cfg
from wetw._reference.model.query_pair_residual import PairQueryDistanceResidual
from wetw._reference.util.helper_functions import (
    weighted_pairwise_distance,
    wetw_distance_kwargs_from_cfg,
    get_data_load_dtype,
    sanitize_time_series_split,
    training_scale_dists_enabled,
)


def get_lr_scheduler(optimizer, cfg, model_type, max_epoch):
    """
    Build LR scheduler based on config.
    model_type: 'encoder' | 'weight' (uses encoder_model or weight_model config)
    Returns: (scheduler, use_metric_for_step)
        use_metric_for_step: True if scheduler.step(metric) should be called, False for scheduler.step()
    """
    if model_type == 'encoder':
        model_cfg = cfg.encoder_model
        sched_name = getattr(getattr(model_cfg, 'scheduler', None), 'name', 'plateau')
        warmup_epochs = getattr(getattr(model_cfg, 'scheduler', None), 'warmup_epochs', 5)
        base_lr = model_cfg.lr_init
        min_lr = model_cfg.min_lr
    else:
        model_cfg = cfg.weight_model
        sched_name = getattr(getattr(model_cfg, 'scheduler', None), 'name', 'plateau')
        warmup_epochs = getattr(getattr(model_cfg, 'scheduler', None), 'warmup_epochs', 5)
        base_lr = model_cfg.lr_init
        min_lr = model_cfg.min_lr

    if sched_name == 'warmup_cosine':
        # Linear warmup for warmup_epochs (training epochs), then cosine decay to min_lr
        eta_min = min_lr / base_lr if base_lr > 0 else 0.0
        total_epochs = max_epoch
        warmup_epochs = min(warmup_epochs, total_epochs - 1)  # ensure cosine phase exists
        # Use ref so lr_lambda sees current training epoch (updated before each step)
        epoch_ref = {'epoch': 0}

        def lr_lambda(_step):
            epoch = epoch_ref['epoch']
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs  # linear warmup from ~0 to 1
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return (scheduler, False, epoch_ref)
    else:
        # plateau (default)
        if model_type == 'encoder':
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=model_cfg.mode,
                factor=model_cfg.lr_reduce_factor,
                patience=model_cfg.sched_pat,
                threshold_mode=model_cfg.threshold_mode,
                threshold=model_cfg.threshold,
                min_lr=model_cfg.min_lr * model_cfg.lr_reduce_factor,
                cooldown=model_cfg.sched_cooldown,
            )
        else:
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=model_cfg.mode,
                factor=model_cfg.lr_reduce_factor,
                patience=model_cfg.lr_scheduler_patience,
                threshold_mode=model_cfg.threshold_mode,
                threshold=model_cfg.threshold,
                min_lr=model_cfg.min_lr * model_cfg.lr_reduce_factor,
                cooldown=model_cfg.sched_cooldown,
            )
        return (scheduler, True, None)


class Train:
    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self.epoch = 0
        self.has_setup = False
        self.use_wandb = getattr(cfg.logging, 'use_wandb', False)
        self.wandb = None
        self.plot_during_training = getattr(cfg.logging, 'plot_during_training', False)
        self.plot_start_epoch = getattr(cfg.logging, 'plot_start_epoch', 1)
        self.plot_frequency = getattr(cfg.logging, 'plot_frequency', 10)
        self.plot_num_samples = getattr(cfg.logging, 'plot_num_samples', 3)
        self.plot_save_dir = None  # Will be set in __setup
        self.plot_fixed_pairs = None  # Will store fixed pair indices
        self.plot_weight_history = {}  # Track weight evolution over epochs
        self._logged_encoder_joint_to_frozen = False
        self._warned_two_phase_eucl = False

    def _encoder_frozen_phase(self) -> bool:
        """True when only the weight model should train (encoder eval, no encoder grads)."""
        n_joint = getattr(self.cfg.training, "encoder_joint_train_epochs", None)
        if n_joint is None:
            return False
        if self.wetw_model is None:
            if not self._warned_two_phase_eucl:
                self.logger.warning(
                    "training.encoder_joint_train_epochs is set but approximator is euclidean; two-phase freeze disabled."
                )
                self._warned_two_phase_eucl = True
            return False
        return self.epoch >= int(n_joint)

    def _pred_calib_enabled(self) -> bool:
        m = getattr(self.cfg.training, "pred_distance_learned_calib", "off")
        if m is None:
            return False
        return str(m).lower() not in ("off", "false", "0", "none", "")

    def _pred_calib_train_when_encoder_frozen(self) -> bool:
        if not self._pred_calib_enabled():
            return False
        return bool(
            getattr(self.cfg.training, "pred_distance_calib_train_when_encoder_frozen", True)
        )

    def _encoder_optimizer_should_run(self, enc_frozen: bool) -> bool:
        if not enc_frozen:
            return True
        cal = getattr(self.encoder_model, "pred_dist_calib", None) if self.encoder_model else None
        if cal is None:
            return False
        if self.wetw_model is not None:
            return False
        return self._pred_calib_train_when_encoder_frozen()

    def _parameters_for_encoder_optimizer(self):
        """Backbone (+ pred_dist_calib only when there is no WETW; else calib is in WETW optimizer)."""
        if self.encoder_model.pred_dist_calib is None or self.wetw_model is None:
            return list(self.encoder_model.parameters())
        return [
            p
            for name, p in self.encoder_model.named_parameters()
            if not name.startswith("pred_dist_calib.")
        ]

    def _parameters_for_wetw_optimizer(self):
        if self.wetw_model is None:
            return []
        params = list(self.wetw_model.parameters())
        if self.encoder_model.pred_dist_calib is not None:
            params.extend(self.encoder_model.pred_dist_calib.parameters())
        if getattr(self, "query_pair_residual", None) is not None:
            params.extend(self.query_pair_residual.parameters())
        return params

    def __setup(self) -> None:
        if not self.has_setup:
            self.has_setup = True

            # Use existing logger if configured, otherwise get root logger
            self.logger = logging.getLogger(self.__class__.__name__)
            
            # Logger is already configured in run_exp.py to only write to file
            # No need to modify handlers here
            
            # Configure anomaly detection based on config
            if hasattr(self.cfg.system, 'detect_anomaly') and self.cfg.system.detect_anomaly:
                torch.autograd.set_detect_anomaly(True)
                self.logger.warning("Anomaly detection is enabled - this will slow down training significantly!")
            else:
                torch.autograd.set_detect_anomaly(False)

            self.device = self.cfg.system.device

            torch.manual_seed(self.cfg.system.torch_manual_seed)
            if str(self.device).startswith('cuda'):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.manual_seed_all(
                        self.cfg.system.torch_cuda_manual_seed)
                else:
                    raise ValueError('cuda is not available')
            self.max_epoch = self.cfg.training.num_epoch

            self.dim_series = self.cfg.dataset.dim_series
            self.dim_embedding = self.cfg.encoder_model.dim_emb
            
            # Load data with error handling
            try:
                processed_fp = getattr(self.cfg.dataset, 'processed_fp', None)
                if processed_fp and os.path.exists(processed_fp) and processed_fp.endswith('.npz'):
                    d = np.load(processed_fp)
                    self.db = sanitize_time_series_split(d["X_train"], self.dim_series, "X_train")
                    self.test_db = sanitize_time_series_split(d["X_test"], self.dim_series, "X_test")
                elif (hasattr(self.cfg.dataset, 'processed_train_fp') and hasattr(self.cfg.dataset, 'processed_test_fp') and
                      os.path.exists(self.cfg.dataset.processed_train_fp) and os.path.exists(self.cfg.dataset.processed_test_fp)):
                    train_fp = self.cfg.dataset.processed_train_fp
                    test_fp = self.cfg.dataset.processed_test_fp
                    load_dtype = get_data_load_dtype(self.cfg)
                    if train_fp.endswith('.npy'):
                        train_data = np.load(train_fp).astype(np.float32)
                    else:
                        train_data = np.fromfile(train_fp, dtype=load_dtype).reshape(-1, self.dim_series).astype(np.float32)
                    if test_fp.endswith('.npy'):
                        self.test_db = np.load(test_fp).astype(np.float32)
                    else:
                        self.test_db = np.fromfile(test_fp, dtype=load_dtype).reshape(-1, self.dim_series).astype(np.float32)
                    # Pre-split: processed_valid_fp provides valid; db = train + valid for Sampler
                    processed_valid_fp = getattr(self.cfg.dataset, 'processed_valid_fp', None)
                    if processed_valid_fp and os.path.exists(processed_valid_fp):
                        if str(processed_valid_fp).endswith('.npy'):
                            valid_data = np.load(processed_valid_fp).astype(np.float32)
                        else:
                            valid_data = np.fromfile(processed_valid_fp, dtype=load_dtype).reshape(-1, self.dim_series).astype(np.float32)
                        train_data = sanitize_time_series_split(train_data, self.dim_series, "train_sample")
                        valid_data = sanitize_time_series_split(valid_data, self.dim_series, "valid_sample")
                        self.test_db = sanitize_time_series_split(self.test_db, self.dim_series, "test_sample")
                        self.db = np.concatenate([train_data, valid_data], axis=0)
                    else:
                        train_data = sanitize_time_series_split(train_data, self.dim_series, "train_sample")
                        self.test_db = sanitize_time_series_split(self.test_db, self.dim_series, "test_sample")
                        self.db = train_data
                else:
                    raise FileNotFoundError("No processed data found. Set processed_fp (.npz) or processed_train_fp/processed_test_fp")
                self.size_db = self.db.shape[0]
                self.size_test_db = self.test_db.shape[0]
                
                if self.size_db == 0:
                    raise ValueError("Training dataset is empty")
                if self.size_test_db == 0:
                    raise ValueError("Test dataset is empty")
                    
                self.logger.info(f"Loaded training data: {self.size_db} samples")
                self.logger.info(f"Loaded test data: {self.size_test_db} samples")
            except Exception as e:
                self.logger.error(f"Failed to load data: {str(e)}")
                raise

            # When processed_valid_fp is provided, use pre-split sizes (bypass valid_split)
            # train_size from processed_train_fp, valid_size from processed_valid_fp (each its own size)
            processed_valid_fp = getattr(self.cfg.dataset, 'processed_valid_fp', None)
            if processed_valid_fp and os.path.exists(processed_valid_fp):
                train_fp = self.cfg.dataset.processed_train_fp
                if str(train_fp).endswith('.npy'):
                    tr = np.load(train_fp, mmap_mode='r')
                    self.train_size = int(tr.shape[0])
                    n_feat = int(tr.shape[1]) if tr.ndim == 2 else self.dim_series
                    if n_feat != self.dim_series:
                        self.logger.warning(
                            "Train .npy second dim %s != dim_series %s; sizes use row count only (data will be sanitized in Sampler).",
                            n_feat, self.dim_series
                        )
                else:
                    n_bytes = os.path.getsize(train_fp)
                    dtype = get_data_load_dtype(self.cfg)
                    self.train_size = n_bytes // (self.dim_series * np.dtype(dtype).itemsize)
                if str(processed_valid_fp).endswith('.npy'):
                    va = np.load(processed_valid_fp, mmap_mode='r')
                    self.valid_size = int(va.shape[0])
                else:
                    n_bytes = os.path.getsize(processed_valid_fp)
                    dtype = get_data_load_dtype(self.cfg)
                    self.valid_size = n_bytes // (self.dim_series * np.dtype(dtype).itemsize)
                self.logger.info(f"Pre-split data: train={self.train_size}, valid={self.valid_size} (from processed_train_fp, processed_valid_fp)")
            else:
                self.valid_size = int(self.cfg.training.valid_split * self.size_db)
                self.train_size = int(self.size_db - self.valid_size)
                # Cap sample sizes if max limits are configured (useful for large datasets)
                if hasattr(self.cfg.training, 'max_train_samples') and self.cfg.training.max_train_samples is not None:
                    self.train_size = min(self.train_size, self.cfg.training.max_train_samples)
                if hasattr(self.cfg.training, 'max_valid_samples') and self.cfg.training.max_valid_samples is not None:
                    self.valid_size = min(self.valid_size, self.cfg.training.max_valid_samples)
            self.batch_size = self.cfg.training.size_batch

            self.train_data_np = None
            self.train_dist_np = None
            self.train_dataloader = None
            self.valid_data_np = None
            self.valid_dist_np = None
            self.valid_dataloader = None


            # Pass pre-loaded db when pre-split (train+valid concatenated); Sampler uses it instead of loading from file
            sampler_data = self.db if (processed_valid_fp and os.path.exists(processed_valid_fp)) else None
            self.sampler = Sampler(self.cfg, 
                                   self.size_db, 
                                   self.train_size,
                                   self.valid_size,
                                   data_array=sampler_data)  # if scheduler is used then resampling happens independently of this variable


            self.encoder_model = None
            self.wetw_model = None

            self.lossF_distances_train = PairwiseDistDiff(  self.cfg,
                                                            dataset_size=self.train_size,
                                                            default_batch_size=self.batch_size,
                                                            device=self.device)

            self.lossF_distances_valid = PairwiseDistDiff(  self.cfg,
                                                            dataset_size=self.valid_size,
                                                            default_batch_size=self.batch_size,
                                                            device=self.device)
            
                
            # encoder Model ======================================================
            self.encoder_model = ModelBuilder(self.cfg)
            if self.encoder_model.pred_dist_calib is not None:
                self.logger.info(
                    "Learned pred-distance calibration: mode=%s (bias_init=%s)",
                    self.encoder_model.pred_dist_calib.mode,
                    getattr(self.cfg.training, "pred_distance_learned_bias_init", 0.0),
                )

            # wetw model =========================================================
            self.approximator = getattr(self.cfg.weight_model, 'approximator', 'wetw')
            self.query_pair_residual = None

            if self.approximator == 'euclidean':
                self.wetw_model = None
                self.wetw_optimizer = None
                self.wetw_scheduler = None
                self.logger.info("Approximator: euclidean (no weight model)")
            else:
                self.wetw_model = PairConvModel(
                    **pair_conv_kwargs_from_cfg(self.cfg, self.dim_embedding)
                ).to(self.device)

                if bool(getattr(self.cfg.training, "wetw_query_pair_residual_enabled", False)):
                    qh = int(getattr(self.cfg.training, "wetw_query_pair_residual_hidden", 128))
                    self.query_pair_residual = PairQueryDistanceResidual(
                        int(self.dim_embedding), hidden_dim=qh
                    ).to(self.device)
                    self.logger.info(
                        "Pair query-distance residual enabled (hidden=%d, scale=%s)",
                        qh,
                        getattr(self.cfg.training, "wetw_query_pair_residual_scale", 0.1),
                    )

                wetw_opt = str(self.cfg.weight_model.optimizer).lower()
                wetw_params = self._parameters_for_wetw_optimizer()
                wetw_lr = self.cfg.weight_model.lr_init
                wetw_wd = float(getattr(self.cfg.weight_model, "weight_decay", 0.0))
                if wetw_opt == 'adam':
                    self.wetw_optimizer = optim.Adam(wetw_params, lr=wetw_lr, weight_decay=wetw_wd)
                elif wetw_opt == 'adamw':
                    self.wetw_optimizer = optim.AdamW(wetw_params, lr=wetw_lr, weight_decay=wetw_wd)
                else:
                    raise ValueError(f"Unsupported weight_model.optimizer: {self.cfg.weight_model.optimizer!r}")

                self.wetw_scheduler, self.wetw_scheduler_use_metric, self.wetw_scheduler_epoch_ref = get_lr_scheduler(
                    self.wetw_optimizer, self.cfg, 'weight', self.max_epoch
                )

            # encoder optimizer (after WETW: pred_dist_calib updates with WETW Adam when WETW exists)
            enc_opt = str(self.cfg.encoder_model.optimizer).lower()
            enc_params = self._parameters_for_encoder_optimizer()
            enc_lr = self.cfg.encoder_model.lr_init
            enc_wd = float(self.cfg.encoder_model.weight_decay)
            if enc_opt == 'sgd':
                self.encoder_optimizer = torch.optim.SGD(
                    enc_params,
                    lr=enc_lr,
                    momentum=self.cfg.encoder_model.momentum,
                    weight_decay=enc_wd,
                )
            elif enc_opt == 'adamw':
                self.encoder_optimizer = optim.AdamW(enc_params, lr=enc_lr, weight_decay=enc_wd)
            else:
                raise ValueError(f"Unsupported encoder_model.optimizer: {self.cfg.encoder_model.optimizer!r}")

            self.encoder_scheduler, self.encoder_scheduler_use_metric, self.encoder_scheduler_epoch_ref = get_lr_scheduler(
                self.encoder_optimizer, self.cfg, 'encoder', self.max_epoch
            )

            if (
                self.encoder_model.pred_dist_calib is not None
                and self.wetw_model is not None
            ):
                self.logger.info(
                    "pred_dist_calib parameters use the weight (WETW) optimizer; encoder optimizer is backbone-only."
                )

            # ===================================================================

            print(self.encoder_model)
            if self.wetw_model is not None:
                print(self.wetw_model)
  
            # -------------------------------------------------------
            # Count parameters 
            # -------------------------------------------------------

            encoder_params = sum(p.numel() for p in self.encoder_model.parameters() if p.requires_grad)
            wetw_params    = sum(p.numel() for p in self.wetw_model.parameters() if p.requires_grad) if self.wetw_model is not None else 0

            parameters_num = encoder_params + wetw_params

            self.logger.info(f"Encoder parameters: {encoder_params:,}")
            self.logger.info(f"WETW parameters: {wetw_params:,}")
            self.logger.info(f"Total trainable parameters: {parameters_num:,}")

            print("--------------------------------------------------")
            print(f"Encoder parameters: {encoder_params:,}")
            print(f"WETW parameters: {wetw_params:,}")
            print(f"Total trainable parameters: {parameters_num:,}")
            print("--------------------------------------------------")
            
            # Initialize wandb if enabled
            if self.use_wandb:
                if not WANDB_AVAILABLE:
                    self.logger.warning("wandb is requested but not installed. Install with: pip install wandb")
                    print("Warning: wandb is requested but not installed. Install with: pip install wandb")
                    self.use_wandb = False
                    self.wandb = None
                else:
                    try:
                        wandb_config = {
                            'use_wandb': True,
                            'wandb_project': getattr(self.cfg.logging, 'wandb_project', 'wetw'),
                            'wandb_entity': getattr(self.cfg.logging, 'wandb_entity', None),
                            'wandb_run_name': getattr(self.cfg.logging, 'wandb_run_name', None),
                            'wandb_tags': getattr(self.cfg.logging, 'wandb_tags', []),
                            'wandb_mode': getattr(self.cfg.logging, 'wandb_mode', 'offline')
                        }
                        
                        # If wandb_run_name is not specified, use the experiment directory name
                        run_name = wandb_config['wandb_run_name']
                        if run_name is None or run_name == "":
                            # Extract directory name from exps_dir (e.g., "SimBinEqual-2026-01-12-14-40-04")
                            run_name = os.path.basename(self.cfg.exps_dir)
                            self.logger.info(f"Using experiment directory name as wandb run name: {run_name}")
                        
                        # Prepare wandb init parameters
                        init_kwargs = {
                            'project': wandb_config['wandb_project'],
                            'config': OmegaConf.to_container(self.cfg, resolve=True),
                            'tags': wandb_config['wandb_tags'] if wandb_config['wandb_tags'] else None,
                            'name': run_name,
                            'mode': wandb_config['wandb_mode']
                        }
                        
                        # Remove None values (but keep 'name' even if it was originally None, since we set it above)
                        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
                        
                        # Add entity if specified
                        if wandb_config['wandb_entity']:
                            init_kwargs['entity'] = wandb_config['wandb_entity']
                        
                        wandb.init(**init_kwargs)
                        self.wandb = wandb
                        self.logger.info(f"wandb initialized successfully with run name: {run_name}")
                        print(f"wandb initialized successfully with run name: {run_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize wandb: {e}")
                        print(f"Warning: Failed to initialize wandb: {e}")
                        self.use_wandb = False
                        self.wandb = None
            
            # Setup plotting if enabled
            if self.plot_during_training:
                if not MATPLOTLIB_AVAILABLE:
                    self.logger.warning("Plotting requested but matplotlib not installed. Install with: pip install matplotlib")
                    print("Warning: Plotting requested but matplotlib not installed. Install with: pip install matplotlib")
                    self.plot_during_training = False
                elif not TSLEARN_AVAILABLE:
                    self.logger.warning("Plotting requested but tslearn not installed. Install with: pip install tslearn")
                    print("Warning: Plotting requested but tslearn not installed. Install with: pip install tslearn")
                    self.plot_during_training = False
                else:
                    # Create plot directory in the model/experiment directory
                    plot_dir = getattr(self.cfg.logging, 'plot_save_dir', 'plots')
                    if plot_dir == "." or plot_dir == "":
                        # Save directly in model directory
                        self.plot_save_dir = self.cfg.exps_dir
                    else:
                        # Save in subdirectory of model directory
                        self.plot_save_dir = os.path.join(self.cfg.exps_dir, plot_dir)
                    os.makedirs(self.plot_save_dir, exist_ok=True)
                    self.logger.info(f"Plotting enabled: start_epoch={self.plot_start_epoch}, frequency={self.plot_frequency}, samples={self.plot_num_samples}")
                    self.logger.info(f"Plots will be saved to: {self.plot_save_dir}")
                    print(f"Plots will be saved to: {self.plot_save_dir}")
                    # Initialize weight history tracking
                    self.plot_weight_history = {i: [] for i in range(self.plot_num_samples)}
                    self.plot_fixed_pairs_data = None  # Will store fixed time series pairs

    def _plot_dtw_warping_path(self, ts1, ts2, warping_path, epoch, pair_idx, save_path):
        """Plot DTW warping path between two time series."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Time series alignment
        ax1 = axes[0]
        ax1.plot(ts1, label='Time Series 1', linewidth=2)
        ax1.plot(ts2, label='Time Series 2', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title(f'DTW Alignment - Epoch {epoch}, Pair {pair_idx}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Warping path matrix
        ax2 = axes[1]
        if warping_path is not None and len(warping_path) > 0:
            path_array = np.array(warping_path)
            # Create warping path visualization
            warp_matrix = np.zeros((len(ts1), len(ts2)))
            for i, j in path_array:
                if 0 <= i < len(ts1) and 0 <= j < len(ts2):
                    warp_matrix[int(i), int(j)] = 1
            
            im = ax2.imshow(warp_matrix, aspect='auto', origin='lower', cmap='YlOrRd', alpha=0.7)
            ax2.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2, label='Warping Path')
            ax2.set_xlabel('Time Series 2 Index')
            ax2.set_ylabel('Time Series 1 Index')
            ax2.set_title('DTW Warping Path')
            ax2.legend()
            plt.colorbar(im, ax=ax2, label='Path')
        else:
            ax2.text(0.5, 0.5, 'No warping path available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('DTW Warping Path (Not Available)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_weight_distribution(self, weights, epoch, save_path):
        """Plot weight distribution from weight model."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        weights_np = weights.detach().cpu().numpy() if torch.is_tensor(weights) else weights
        n_pairs_aggr = int(weights_np.shape[0]) if weights_np.ndim >= 2 else 1
        
        # Plot 1: Weight distribution histogram
        ax1 = axes[0]
        ax1.hist(weights_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(
            f'Weight Distribution - Epoch {epoch} (n_pairs={n_pairs_aggr}, histogram valid for all weights from those pairs)'
        )
        ax1.grid(True, alpha=0.3)
        ax1.axvline(weights_np.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {weights_np.mean():.4f}')
        ax1.axvline(np.median(weights_np), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(weights_np):.4f}')
        ax1.legend()
        
        # Plot 2: Weight values across dimensions
        ax2 = axes[1]
        if weights_np.ndim == 1:
            ax2.plot(weights_np, 'o-', markersize=4)
            ax2.set_xlabel('Dimension')
        else:
            # If 2D, plot mean across batch
            mean_weights = weights_np.mean(axis=0) if weights_np.ndim == 2 else weights_np
            std_weights = weights_np.std(axis=0) if weights_np.ndim == 2 else np.zeros_like(mean_weights)
            ax2.plot(mean_weights, 'o-', markersize=4, label='Mean')
            ax2.fill_between(range(len(mean_weights)), 
                           mean_weights - std_weights, 
                           mean_weights + std_weights, 
                           alpha=0.3, label='±1 Std')
            ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Weight Value')
        ax2.set_title(
            f'Weight values across dimensions (mean ± std over {n_pairs_aggr} fixed valid pair(s))'
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(
            f"[plotting] Saved weight_distribution epoch {epoch}: "
            f"aggregated over {n_pairs_aggr} pair(s) → {save_path}"
        )

    @staticmethod
    def _welford_merge_batches(n_prev, mean_prev, m2_prev, w_batch):
        """Merge Welford aggregates with a new batch w_batch (B, D). Returns (n, mean, M2)."""
        B, _D = w_batch.shape
        if B == 0:
            return n_prev, mean_prev, m2_prev
        n_b = float(B)
        mean_b = w_batch.mean(dim=0)
        m2_b = ((w_batch - mean_b) ** 2).sum(dim=0)
        if n_prev == 0 or mean_prev is None:
            return n_b, mean_b, m2_b
        n = n_prev + n_b
        delta = mean_b - mean_prev
        mean = mean_prev + delta * (n_b / n)
        m2 = m2_prev + m2_b + delta * delta * n_prev * n_b / n
        return n, mean, m2

    def _encode_series_matrix(self, series_np: np.ndarray) -> torch.Tensor:
        """series_np: (N, dim_series) -> embeddings (N, dim_emb) on self.device."""
        self.encoder_model.eval()
        x = torch.from_numpy(series_np.reshape(-1, 1, self.dim_series)).float().to(self.device)
        batch = int(getattr(self.cfg.training, "size_batch", 256))
        outs = []
        with torch.no_grad():
            for s in range(0, x.shape[0], batch):
                chunk = x[s : s + batch]
                enc = self.encoder_model.encode(chunk)
                emb = enc[0] if isinstance(enc, tuple) else enc
                emb = torch.squeeze(emb)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                outs.append(emb)
        return torch.cat(outs, dim=0)

    def _export_split_weight_dim_pdf(self, split: str, data_np: np.ndarray) -> None:
        """
        Mean ±1 std of WETW weights vs embedding dim + histogram for ``split`` (train|valid|test).
        Caller must have loaded best encoder/wetw and set eval mode.
        """
        split_l = split.lower()
        pair_bs = int(getattr(self.cfg.logging, "valid_full_weight_dim_pair_batch", 4096))
        pair_cap = int(getattr(self.cfg.logging, "valid_full_weight_dim_pair_cap", 2_500_000))
        max_pairs_cfg = getattr(self.cfg.logging, "valid_full_weight_dim_max_pairs", None)
        seed = int(getattr(self.cfg.system, "torch_manual_seed", 0))
        rng = np.random.default_rng(seed + {"train": 101, "valid": 202, "test": 303}[split_l])

        emb = self._encode_series_matrix(data_np)
        N = int(emb.shape[0])
        n_all = N * (N - 1) // 2
        use_random = False
        n_target = n_all
        if max_pairs_cfg is not None:
            n_target = min(n_all, int(max_pairs_cfg))
            use_random = n_target < n_all
        elif n_all > pair_cap:
            n_target = pair_cap
            use_random = True
            self.logger.info(
                "%s weight PDF: C(%d,2)=%d pairs; subsampling %d random (cap logging.valid_full_weight_dim_pair_cap).",
                split_l,
                N,
                n_all,
                n_target,
            )

        n_acc, mean_w, m2 = 0.0, None, None
        hist_lo = float(getattr(self.cfg.logging, "valid_full_weight_dim_hist_lo", -5.0))
        hist_hi = float(getattr(self.cfg.logging, "valid_full_weight_dim_hist_hi", 5.0))
        hist_edges = np.linspace(hist_lo, hist_hi, 51)
        hist_counts = np.zeros(50, dtype=np.float64)
        self.logger.info(
            "%s weight PDF: N=%d, pairs_target=%d (all=%d, random=%s), pair_batch=%d",
            split_l,
            N,
            n_target,
            n_all,
            use_random,
            pair_bs,
        )

        def accum_batch(w_tensor: torch.Tensor) -> None:
            nonlocal n_acc, mean_w, m2, hist_counts
            n_acc, mean_w, m2 = self._welford_merge_batches(n_acc, mean_w, m2, w_tensor)
            flat = w_tensor.detach().float().cpu().numpy().ravel()
            c, _ = np.histogram(flat, bins=hist_edges)
            hist_counts += c.astype(np.float64)

        with torch.no_grad():
            if use_random:
                done = 0
                while done < n_target:
                    b = min(pair_bs, n_target - done)
                    ii = rng.integers(0, N, size=b, dtype=np.int64)
                    jj = rng.integers(0, N, size=b, dtype=np.int64)
                    same = ii == jj
                    while np.any(same):
                        jj[same] = rng.integers(0, N, size=int(same.sum()), dtype=np.int64)
                        same = ii == jj
                    lo = np.minimum(ii, jj)
                    hi = np.maximum(ii, jj)
                    ei = emb[torch.from_numpy(lo).to(self.device)]
                    ej = emb[torch.from_numpy(hi).to(self.device)]
                    accum_batch(self.wetw_model(ei, ej))
                    done += b
            else:
                for i in range(N - 1):
                    js = torch.arange(i + 1, N, device=self.device, dtype=torch.long)
                    for s in range(0, js.numel(), pair_bs):
                        j_batch = js[s : s + pair_bs]
                        ei = emb[i].unsqueeze(0).expand(j_batch.numel(), -1)
                        ej = emb[j_batch]
                        accum_batch(self.wetw_model(ei, ej))

        if mean_w is None:
            self.logger.warning("%s weight PDF: no pairs accumulated; skip.", split_l)
            return

        mean_np = mean_w.detach().cpu().numpy()
        std_np = torch.sqrt(torch.clamp(m2 / max(n_acc, 1.0), min=0)).detach().cpu().numpy()
        dim_range = np.arange(mean_np.shape[0])

        plot_dir = os.path.join(self.cfg.exps_dir, getattr(self.cfg.logging, "plot_save_dir", "plots"))
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, f"{split_l}_full_weight_dimensions.pdf")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        ax0 = axes[0]
        bc = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        ax0.bar(bc, hist_counts, width=(hist_edges[1] - hist_edges[0]) * 0.95, align="center", edgecolor="black", alpha=0.75)
        ax0.set_xlabel("Weight Value")
        ax0.set_ylabel("Count")
        ax0.set_title(
            f"[{split_l}] Weight distribution ([{hist_lo:.1f}, {hist_hi:.1f}]; pair updates={int(n_acc)}, N={N})"
        )
        ax0.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(mean_np, "o-", markersize=4, label="Mean")
        ax2.fill_between(
            dim_range,
            mean_np - std_np,
            mean_np + std_np,
            alpha=0.3,
            label="±1 Std",
        )
        ax2.set_xlabel("Dimension")
        ax2.set_ylabel("Weight Value")
        mode = f"all unordered {split_l} pairs" if not use_random else f"random subsample (~{int(n_acc)} pairs)"
        ax2.set_title(f"[{split_l}] Weight values across dimensions (mean ± std; {mode})")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Saved %s split weight dimension PDF: %s", split_l, out_path)
        print(f"[plotting] Saved {split_l}_full_weight_dimensions.pdf ({int(n_acc)} pairs) → {out_path}")

    def _maybe_save_full_weight_dim_pdfs(self) -> None:
        """After training: optional PDFs for train / valid / test (separate toggles)."""
        log = self.cfg.logging
        specs = (
            ("valid", "save_valid_full_weight_dim_pdf", "valid_data_np"),
            ("train", "save_train_full_weight_dim_pdf", "train_data_np"),
            ("test", "save_test_full_weight_dim_pdf", "test_db"),
        )
        if not any(getattr(log, flag, False) for _sp, flag, _a in specs):
            return
        if self.wetw_model is None or str(getattr(self.cfg.weight_model, "approximator", "wetw")).lower() == "euclidean":
            return
        if not MATPLOTLIB_AVAILABLE or plt is None:
            self.logger.warning("full weight dim PDF: matplotlib not available; skip all.")
            return
        enc_fp = self.cfg.paths.best_encoder_sd_fp
        wetw_fp = self.cfg.paths.best_wetw_sd_fp
        if not enc_fp or not os.path.isfile(enc_fp):
            self.logger.warning("full weight dim PDF: missing %s; skip all.", enc_fp)
            return
        if not wetw_fp or not os.path.isfile(wetw_fp):
            self.logger.warning("full weight dim PDF: missing %s; skip all.", wetw_fp)
            return

        self.encoder_model.load_state_dict(torch.load(enc_fp, map_location=self.device))
        self.wetw_model.load_state_dict(torch.load(wetw_fp, map_location=self.device))
        self.encoder_model.eval()
        self.wetw_model.eval()

        for split, flag, attr in specs:
            if not getattr(log, flag, False):
                continue
            data = getattr(self, attr, None)
            if data is None or (hasattr(data, "shape") and data.shape[0] < 2):
                self.logger.warning("full weight dim PDF (%s): need >=2 rows (attr=%s); skip.", split, attr)
                continue
            try:
                self._export_split_weight_dim_pdf(split, np.asarray(data, dtype=np.float32))
            except Exception as e:
                self.logger.warning("full weight dim PDF (%s) failed: %s", split, e, exc_info=True)
    
    def _initialize_fixed_pairs(self):
        """Initialize fixed pairs from validation set for consistent plotting."""
        if self.valid_dataloader is None:
            self.logger.warning("Cannot initialize fixed pairs: valid_dataloader is None")
            return False
        
        if len(self.valid_dataloader) == 0:
            self.logger.warning("Cannot initialize fixed pairs: valid_dataloader is empty")
            return False
        
        try:
            self.logger.info(f"Attempting to initialize fixed pairs from valid_dataloader (size={len(self.valid_dataloader)})")
            # Get a batch of validation data
            valid_batch = next(iter(self.valid_dataloader))
            batch_data, batch_indices = valid_batch
            # Handle shape: (batch, 1, length) -> (batch, length)
            if batch_data.ndim == 3:
                batch_data = batch_data.squeeze(1)
            elif batch_data.ndim == 2 and batch_data.shape[1] == 1:
                batch_data = batch_data.squeeze(1)
            
            # Limit number of samples
            num_samples = min(self.plot_num_samples, batch_data.shape[0] // 2)
            
            # Store fixed pairs data
            self.plot_fixed_pairs_data = []
            for pair_idx in range(num_samples):
                idx1 = pair_idx * 2
                idx2 = pair_idx * 2 + 1
                
                if idx2 >= batch_data.shape[0]:
                    break
                
                # Store original time series (on CPU)
                ts1 = batch_data[idx1].cpu().clone()
                ts2 = batch_data[idx2].cpu().clone()
                
                # Store global indices for reference
                global_idx1 = batch_indices[idx1].item() if torch.is_tensor(batch_indices) else batch_indices[idx1]
                global_idx2 = batch_indices[idx2].item() if torch.is_tensor(batch_indices) else batch_indices[idx2]
                
                self.plot_fixed_pairs_data.append({
                    'ts1': ts1,
                    'ts2': ts2,
                    'global_idx1': global_idx1,
                    'global_idx2': global_idx2,
                    'pair_idx': pair_idx
                })
            
            n_pairs_fixed = len(self.plot_fixed_pairs_data)
            batch_rows = int(batch_data.shape[0])
            pair_cap = batch_rows // 2
            msg = (
                f"Initialized {n_pairs_fixed} fixed pair(s) for plotting "
                f"(plot_num_samples={self.plot_num_samples}, "
                f"first valid batch rows={batch_rows} → max {pair_cap} pair(s))"
            )
            self.logger.info(msg)
            print(f"[plotting] {msg}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize fixed pairs: {e}")
            return False
    
    def _plot_weight_evolution(self, epoch):
        """Plot weight evolution over epochs for fixed pairs."""
        if not self.plot_weight_history or len(self.plot_weight_history) == 0:
            return
        
        fig, axes = plt.subplots(len(self.plot_weight_history), 1, figsize=(12, 4 * len(self.plot_weight_history)))
        if len(self.plot_weight_history) == 1:
            axes = [axes]
        
        for pair_idx, (pair_key, weight_history) in enumerate(self.plot_weight_history.items()):
            if len(weight_history) == 0:
                continue
            
            ax = axes[pair_idx]
            epochs = [entry['epoch'] for entry in weight_history]
            weights_list = [entry['weights'] for entry in weight_history]
            
            # Convert to numpy if needed
            weights_array = np.array([w.cpu().numpy() if torch.is_tensor(w) else w for w in weights_list])
            
            # Plot mean weight over time
            mean_weights = weights_array.mean(axis=1)
            std_weights = weights_array.std(axis=1)
            
            ax.plot(epochs, mean_weights, 'o-', linewidth=2, markersize=4, label='Mean Weight')
            ax.fill_between(epochs, mean_weights - std_weights, mean_weights + std_weights, 
                          alpha=0.3, label='±1 Std')
            
            # Plot individual dimension weights (sample a few)
            if weights_array.shape[1] > 0:
                num_dims_to_show = min(5, weights_array.shape[1])
                dim_indices = np.linspace(0, weights_array.shape[1] - 1, num_dims_to_show, dtype=int)
                for dim_idx in dim_indices:
                    ax.plot(epochs, weights_array[:, dim_idx], '--', alpha=0.5, linewidth=1,
                           label=f'Dim {dim_idx}' if dim_idx < 3 else '')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Weight Value')
            ax.set_title(f'Weight Evolution - Pair {pair_key}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Use PNG for wandb (PIL/wandb.Image cannot open PDFs)
        plot_ext = '.png' if (self.use_wandb and self.wandb) else '.pdf'
        evolution_plot_path = os.path.join(self.plot_save_dir, f'weight_evolution_epoch_{epoch}{plot_ext}')
        plt.savefig(evolution_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Log to wandb if enabled
        if self.use_wandb and self.wandb:
            self.wandb.log({
                'plots/weight_evolution': wandb.Image(evolution_plot_path)
            }, step=epoch)
    
    def _create_plots(self, epoch):
        """Create DTW warping path and weight distribution plots using fixed pairs."""
        if not self.plot_during_training:
            self.logger.warning(f"Plotting requested but plot_during_training is False at epoch {epoch}")
            return
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning(f"Plotting requested but matplotlib not available at epoch {epoch}")
            return
        
        try:
            # Initialize fixed pairs on first call
            if self.plot_fixed_pairs_data is None:
                self.logger.info(f"Initializing fixed pairs for plotting at epoch {epoch}")
                if not self._initialize_fixed_pairs():
                    self.logger.warning(f"Failed to initialize fixed pairs at epoch {epoch}")
                    return
                self.logger.info(f"Successfully initialized {len(self.plot_fixed_pairs_data)} fixed pairs")
            
            if len(self.plot_fixed_pairs_data) == 0:
                self.logger.warning(f"No fixed pairs data available at epoch {epoch}")
                return
            
            self.encoder_model.eval()
            if self.wetw_model is not None:
                self.wetw_model.eval()
            
            all_weights = []
            
            with torch.no_grad():
                # Process each fixed pair
                for pair_data in self.plot_fixed_pairs_data:
                    ts1 = pair_data['ts1']
                    ts2 = pair_data['ts2']
                    pair_idx = pair_data['pair_idx']
                    
                    # Move to device
                    ts1_device = ts1.to(self.device)
                    ts2_device = ts2.to(self.device)
                    
                    # Ensure proper shape for encoder (add batch and channel dims)
                    if ts1_device.ndim == 1:
                        ts1_device = ts1_device.unsqueeze(0).unsqueeze(0)  # (1, 1, length)
                    elif ts1_device.ndim == 2:
                        ts1_device = ts1_device.unsqueeze(0)  # (1, 1, length)
                    
                    if ts2_device.ndim == 1:
                        ts2_device = ts2_device.unsqueeze(0).unsqueeze(0)
                    elif ts2_device.ndim == 2:
                        ts2_device = ts2_device.unsqueeze(0)
                    
                    # Get embeddings
                    batch_encoded1 = self.encoder_model.encode(ts1_device)
                    batch_encoded2 = self.encoder_model.encode(ts2_device)
                    
                    if isinstance(batch_encoded1, tuple):
                        emb1 = batch_encoded1[0]
                    else:
                        emb1 = batch_encoded1
                    
                    if isinstance(batch_encoded2, tuple):
                        emb2 = batch_encoded2[0]
                    else:
                        emb2 = batch_encoded2
                    
                    # Ensure embeddings have correct shape [1, embed_dim] for weight model
                    # The weight model expects (B, embed_dim) where embed_dim = num_tokens * token_dim
                    # Handle various possible shapes from encoder
                    if emb1.ndim == 1:
                        # 1D tensor [embed_dim] -> add batch dimension
                        emb1 = emb1.unsqueeze(0)  # [embed_dim] -> [1, embed_dim]
                    elif emb1.ndim == 2:
                        # 2D tensor - check if first dim is batch or if it's [1, embed_dim] already
                        if emb1.shape[0] != 1:
                            # If batch size > 1, take first sample
                            emb1 = emb1[:1]
                    elif emb1.ndim > 2:
                        # More than 2D - flatten but keep first dim as batch
                        emb1 = emb1.view(emb1.shape[0], -1)
                        if emb1.shape[0] != 1:
                            emb1 = emb1[:1]
                    
                    if emb2.ndim == 1:
                        # 1D tensor [embed_dim] -> add batch dimension
                        emb2 = emb2.unsqueeze(0)  # [embed_dim] -> [1, embed_dim]
                    elif emb2.ndim == 2:
                        # 2D tensor - check if first dim is batch or if it's [1, embed_dim] already
                        if emb2.shape[0] != 1:
                            # If batch size > 1, take first sample
                            emb2 = emb2[:1]
                    elif emb2.ndim > 2:
                        # More than 2D - flatten but keep first dim as batch
                        emb2 = emb2.view(emb2.shape[0], -1)
                        if emb2.shape[0] != 1:
                            emb2 = emb2[:1]
                    
                    # Final check: ensure both are exactly [1, embed_dim]
                    assert emb1.ndim == 2 and emb1.shape[0] == 1, f"emb1 shape should be [1, embed_dim], got {emb1.shape}"
                    assert emb2.ndim == 2 and emb2.shape[0] == 1, f"emb2 shape should be [1, embed_dim], got {emb2.shape}"
                    
                    # Get weights from weight model (skip when approximator=euclidean)
                    if self.wetw_model is not None:
                        weights = self.wetw_model(emb1, emb2)
                        # Store weights for evolution tracking
                        if pair_idx not in self.plot_weight_history:
                            self.plot_weight_history[pair_idx] = []
                        self.plot_weight_history[pair_idx].append({
                            'epoch': epoch,
                            'weights': weights[0].clone().cpu()  # Store first (and only) sample
                        })
                        all_weights.append(weights)
                    
                    # Get time series as numpy for plotting
                    ts1_np = ts1.numpy() if torch.is_tensor(ts1) else ts1
                    ts2_np = ts2.numpy() if torch.is_tensor(ts2) else ts2
                    
                    # Ensure 1D arrays for DTW
                    if ts1_np.ndim > 1:
                        ts1_np = ts1_np.flatten()
                    if ts2_np.ndim > 1:
                        ts2_np = ts2_np.flatten()
                    
                    # Compute DTW warping path (only once, it's the same for fixed pairs)
                    warping_path = None
                    if TSLEARN_AVAILABLE and epoch == 1:  # Compute once at first epoch
                        try:
                            path, dist = dtw_path(ts1_np, ts2_np)
                            warping_path = np.array(path)
                            # Store for future use
                            pair_data['warping_path'] = warping_path
                        except Exception as e:
                            self.logger.warning(f"Failed to compute DTW path for pair {pair_idx}: {e}")
                    elif 'warping_path' in pair_data:
                        warping_path = pair_data['warping_path']
                    
                    # Plot DTW warping path
                    # Use PNG for wandb (PIL/wandb.Image cannot open PDFs)
                    plot_ext = '.png' if (self.use_wandb and self.wandb) else '.pdf'
                    dtw_plot_path = os.path.join(self.plot_save_dir, f'dtw_warping_epoch_{epoch}_pair_{pair_idx}{plot_ext}')
                    self._plot_dtw_warping_path(ts1_np, ts2_np, warping_path, epoch, pair_idx, dtw_plot_path)
                    
                    # Log to wandb if enabled
                    if self.use_wandb and self.wandb:
                        self.wandb.log({
                            f'plots/dtw_warping_pair_{pair_idx}': wandb.Image(dtw_plot_path)
                        }, step=epoch)
                
                # Plot weight distribution (aggregate across all pairs)
                if all_weights:
                    aggregated_weights = torch.cat(all_weights, dim=0)
                    # Use PNG for wandb (PIL/wandb.Image cannot open PDFs)
                    plot_ext = '.png' if (self.use_wandb and self.wandb) else '.pdf'
                    weight_plot_path = os.path.join(self.plot_save_dir, f'weight_distribution_epoch_{epoch}{plot_ext}')
                    self._plot_weight_distribution(aggregated_weights, epoch, weight_plot_path)
                    
                    # Log to wandb if enabled
                    if self.use_wandb and self.wandb:
                        self.wandb.log({
                            'plots/weight_distribution': wandb.Image(weight_plot_path)
                        }, step=epoch)
                
                # Plot weight evolution over epochs
                if len(self.plot_weight_history) > 0 and any(len(h) > 1 for h in self.plot_weight_history.values()):
                    self._plot_weight_evolution(epoch)
        
        except Exception as e:
            import traceback
            error_msg = f"Failed to create plots at epoch {epoch}: {e}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            print(f"Error: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")

    def _pretrain_encoder(self, n_pretrain: int, pretrain_lr: float, use_identity: bool) -> None:
        """
        Encoder-only warmup before joint WETW training.

        use_identity=True (v2-style): MSE between Euclidean embedding distance and DTW.
        use_identity=False: persistent linear decoder reconstruction (MSE on raw series).
        """
        self.logger.info('=' * 60)
        self.logger.info('Encoder Pre-training Phase (%d epochs)' % n_pretrain)
        self.logger.info(
            'Mode: %s',
            'euclidean-vs-dtw (v2)' if use_identity else 'linear reconstruction',
        )
        self.logger.info('=' * 60)

        n_train = len(self.train_data_np)
        n_pairs_cap = min(5000, n_train * (n_train - 1) // 2)
        chunk = max(1, int(self.batch_size))

        decoder = None
        decoder_opt = None
        if not use_identity:
            dim_emb = int(self.cfg.encoder_model.dim_emb)
            dim_series = int(self.dim_series)
            decoder = nn.Linear(dim_emb, dim_series, device=self.device)
            nn.init.xavier_uniform_(decoder.weight)
            nn.init.zeros_(decoder.bias)
            decoder_opt = optim.Adam(decoder.parameters(), lr=pretrain_lr)

        orig_enc_lr = self.encoder_optimizer.param_groups[0]['lr']
        for pg in self.encoder_optimizer.param_groups:
            pg['lr'] = pretrain_lr

        if self.wetw_model is not None:
            self.wetw_model.eval()
            for p in self.wetw_model.parameters():
                p.requires_grad = False

        l2 = nn.PairwiseDistance(p=2).to(self.device)
        scale = training_scale_dists_enabled(self.cfg)
        dim_emb = int(self.cfg.encoder_model.dim_emb)

        for pe in range(1, n_pretrain + 1):
            self.encoder_model.train()
            self.encoder_optimizer.zero_grad()
            if decoder_opt is not None:
                decoder_opt.zero_grad()
            if use_identity:
                gi = torch.randint(0, n_train, (n_pairs_cap,), device=self.device)
                gj = torch.randint(0, n_train, (n_pairs_cap,), device=self.device)
                same = gi == gj
                if same.any():
                    gj = torch.where(same, (gj + 1) % n_train, gj)

                # MSE over all pairs, back-propagated chunk by chunk (gradient
                # accumulation): identical gradients to one big graph, but memory
                # no longer grows with n_pairs_cap * L (needed for L >= ~3000).
                true_dists = self.lossF_distances_train._targets_from_global_pairs(gi, gj)
                loss_sum = 0.0
                for start in range(0, n_pairs_cap, chunk):
                    end = min(start + chunk, n_pairs_cap)
                    i_np = gi[start:end].cpu().numpy()
                    j_np = gj[start:end].cpu().numpy()
                    ts_i = torch.from_numpy(self.train_data_np[i_np]).float().to(self.device).unsqueeze(1)
                    ts_j = torch.from_numpy(self.train_data_np[j_np]).float().to(self.device).unsqueeze(1)
                    enc_i = self.encoder_model.encode(ts_i)
                    enc_j = self.encoder_model.encode(ts_j)
                    emb_i = enc_i[0] if isinstance(enc_i, tuple) else enc_i
                    emb_j = enc_j[0] if isinstance(enc_j, tuple) else enc_j
                    pred = l2(emb_i, emb_j)
                    if scale:
                        pred = pred / math.sqrt(dim_emb)
                    part = F.mse_loss(pred, true_dists[start:end], reduction='sum') / n_pairs_cap
                    part.backward()
                    loss_sum += part.item()
                loss = torch.tensor(loss_sum)
            else:
                idx = torch.randperm(n_train, device=self.device)[:chunk]
                batch = torch.from_numpy(self.train_data_np[idx.cpu().numpy()]).float().to(self.device).unsqueeze(1)
                enc = self.encoder_model.encode(batch)
                emb = enc[0] if isinstance(enc, tuple) else enc
                dec = decoder(emb)
                loss = F.mse_loss(dec, batch.squeeze(1))
                loss.backward()

            if hasattr(self.cfg.training, 'grad_clip') and self.cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.encoder_model.parameters(), self.cfg.training.grad_clip)
            self.encoder_optimizer.step()
            if decoder_opt is not None:
                if hasattr(self.cfg.training, 'grad_clip') and self.cfg.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), self.cfg.training.grad_clip)
                decoder_opt.step()

            if pe % 10 == 0 or pe == 1 or pe == n_pretrain:
                self.logger.info('  [Pretrain] Epoch %3d/%d | Loss: %.6f' % (pe, n_pretrain, loss.item()))

        for pg in self.encoder_optimizer.param_groups:
            pg['lr'] = orig_enc_lr

        if self.wetw_model is not None:
            for p in self.wetw_model.parameters():
                p.requires_grad = True
            self.wetw_model.train()

        self.logger.info('Encoder pre-training complete.')

    def train(self, extra_epoch: int = None) -> None:
        self.__setup()

        # Always materialize config at experiment root so downstream scripts (e.g. batch_trains.sh
        # find_latest_experiment) and eval can resolve the run even if valid loss is never finite
        # or no "best" checkpoint is written.
        try:
            OmegaConf.save(self.cfg, self.cfg.paths.conf_fp)
        except Exception as e:
            self.logger.warning("Could not save config.yaml at train start: %s", e)

        local_max_epoch = self.max_epoch
        max_wall_hours = getattr(self.cfg.training, 'max_wall_hours', None)
        max_wall_seconds = None
        if max_wall_hours is not None:
            try:
                max_wall_seconds = float(max_wall_hours) * 3600.0
            except Exception:
                max_wall_seconds = None
        train_start_wall = timer()

        # Sample data before pretrain
        self.__sample()
        if self.plot_during_training and self.plot_fixed_pairs_data is None:
            self.logger.info("Initializing fixed pairs after sampling at epoch 0")
            self._initialize_fixed_pairs()

        resume_epoch = getattr(self.cfg.training, 'resume_epoch', None)
        if resume_epoch is not None:
            resume_epoch = int(resume_epoch)
            enc_fp = self.cfg.paths.best_encoder_sd_fp
            wetw_fp = self.cfg.paths.best_wetw_sd_fp
            if not enc_fp or not os.path.isfile(enc_fp):
                raise FileNotFoundError(f"Resume requires encoder checkpoint: {enc_fp}")
            self.encoder_model.load_state_dict(
                torch.load(enc_fp, map_location=self.device)
            )
            if self.wetw_model is not None:
                if not wetw_fp or not os.path.isfile(wetw_fp):
                    raise FileNotFoundError(f"Resume requires WETW checkpoint: {wetw_fp}")
                self.wetw_model.load_state_dict(
                    torch.load(wetw_fp, map_location=self.device)
                )
            self.epoch = resume_epoch
            if resume_epoch >= int(getattr(self.cfg.training, 'encoder_joint_train_epochs', 0) or 0):
                self._logged_encoder_joint_to_frozen = False
            resume_wetw_lr = getattr(self.cfg.training, 'resume_wetw_lr', None)
            if resume_wetw_lr is not None and self.wetw_optimizer is not None:
                for pg in self.wetw_optimizer.param_groups:
                    pg['lr'] = float(resume_wetw_lr)
            resume_enc_lr = getattr(self.cfg.training, 'resume_encoder_lr', None)
            if resume_enc_lr is not None and self.encoder_optimizer is not None:
                for pg in self.encoder_optimizer.param_groups:
                    pg['lr'] = float(resume_enc_lr)
            self.logger.info(
                "Resumed from %s at epoch %d (best_valid_loss=%s)",
                self.cfg.exps_dir,
                resume_epoch,
                getattr(self.cfg.training, 'resume_best_valid_loss', 'unknown'),
            )
            print(
                f"Resuming training at epoch {resume_epoch} from {self.cfg.exps_dir}",
                flush=True,
            )
        
        # === ENCODER PRETRAINING PHASE ===
        n_pretrain = getattr(self.cfg.training, 'encoder_pretrain_epochs', None)
        if getattr(self.cfg.training, 'skip_encoder_pretrain', False):
            n_pretrain = 0
        pretrain_lr = float(getattr(self.cfg.training, 'encoder_pretrain_lr', 0.001))
        use_identity = bool(getattr(self.cfg.training, 'encoder_pretrain_use_identity', True))
        if n_pretrain is not None and n_pretrain > 0:
            self._pretrain_encoder(n_pretrain, pretrain_lr, use_identity)
        
        resume_best = getattr(self.cfg.training, 'resume_best_valid_loss', None)
        self.best_valid_loss = float(resume_best) if resume_best is not None else float('inf')
        self.total_compute_time = 0.0
        epochs_since_best = 0
        early_stopping_patience = getattr(self.cfg.training, 'early_stopping_patience', None)
        early_stopping_max_wait_after_patience = getattr(self.cfg.training, 'early_stopping_max_wait_after_patience', None)
        epochs_since_patience_exhausted = 0  # Track epochs after patience is exhausted
        encoder_best_sd = None  # Initialize to avoid undefined variable errors
        wetw_best_sd = None
        
        # Create main progress bar for epochs
        epoch_pbar = tqdm(
            range(local_max_epoch),
            desc="Training",
            unit="epoch",
            initial=self.epoch,
            leave=True,
            ncols=100
        )
        
        while self.epoch < local_max_epoch:
            
            if self.epoch == 0:
                # __sample() already called during pretrain phase
                if self.plot_during_training and self.plot_fixed_pairs_data is None:
                    self.logger.info("Initializing fixed pairs at epoch 0")
                    self._initialize_fixed_pairs()

            if not self._encoder_frozen_phase():
                self.__adjust_wd(self.cfg.encoder_model.wd_max, self.cfg.encoder_model.wd_min)

            epoch_train_loss, train_compute = self.__train_epoch()
            epoch_valid_loss, valid_compute = self.__validate_epoch()
            self.total_compute_time += train_compute + valid_compute

            # Plateau: step only after start_sched_after_epoch. Warmup_cosine: step every epoch.
            if not self._encoder_frozen_phase():
                if self.encoder_scheduler_use_metric:
                    if self.epoch >= self.cfg.training.start_sched_after_epoch:
                        self.encoder_scheduler.step(epoch_valid_loss)
                else:
                    if self.encoder_scheduler_epoch_ref is not None:
                        self.encoder_scheduler_epoch_ref['epoch'] = self.epoch
                    self.encoder_scheduler.step()
            if self.wetw_scheduler is not None:
                if self.wetw_scheduler_use_metric:
                    if self.epoch >= self.cfg.training.start_sched_after_epoch:
                        self.wetw_scheduler.step(epoch_valid_loss)
                else:
                    if self.wetw_scheduler_epoch_ref is not None:
                        self.wetw_scheduler_epoch_ref['epoch'] = self.epoch
                    self.wetw_scheduler.step()

            # Update epoch progress bar
            lr_current_enc = self.encoder_optimizer.param_groups[0]['lr'] if self.encoder_optimizer.param_groups else 0.0
            lr_current_wetw = self.wetw_optimizer.param_groups[0]['lr'] if (self.wetw_optimizer is not None and self.wetw_optimizer.param_groups) else 0.0
            
            # Check for improvement in validation loss
            improved = False
            if epoch_valid_loss < self.best_valid_loss:
                # Check if improvement is significant (above threshold)
                improvement_threshold = getattr(self.cfg.training, 'early_stopping_min_delta', 0.0)
                if (self.best_valid_loss - epoch_valid_loss) >= improvement_threshold:
                    self.best_valid_loss = epoch_valid_loss
                    epochs_since_best = 0
                    improved = True
                    encoder_best_sd = self.encoder_model.state_dict()
                    wetw_best_sd = self.wetw_model.state_dict() if self.wetw_model is not None else None

                    torch.save(encoder_best_sd, self.cfg.paths.best_encoder_sd_fp)
                    if wetw_best_sd is not None:
                        torch.save(wetw_best_sd, self.cfg.paths.best_wetw_sd_fp)
                    if getattr(self, "query_pair_residual", None) is not None:
                        qr_fp = os.path.join(self.cfg.exps_dir, "query_pair_residual.pt")
                        torch.save(self.query_pair_residual.state_dict(), qr_fp)
                    OmegaConf.save(self.cfg, self.cfg.paths.conf_fp)
                else:
                    epochs_since_best += 1
            else:
                epochs_since_best += 1
            
            # Log combined epoch summary to wandb (after updating best loss tracking)
            if self.use_wandb and self.wandb:
                # Check if this was the best validation loss (improved flag indicates we just updated best_valid_loss)
                is_best = improved
                
                self.wandb.log({
                    'epoch/train_loss': epoch_train_loss,
                    'epoch/valid_loss': epoch_valid_loss,
                    'epoch/best_valid_loss': self.best_valid_loss,
                    'epoch/is_best': 1.0 if is_best else 0.0,
                    'epoch/lr_encoder': lr_current_enc,
                    'epoch/lr_wetw': lr_current_wetw,
                    'epoch/epochs_since_best': epochs_since_best,
                    'epoch/encoder_frozen': 1.0 if self._encoder_frozen_phase() else 0.0,
                }, step=self.epoch)
            
            self.epoch += 1
            
            # Create plots if enabled and at the right frequency (after epoch increment)
            # Only plot when epoch >= plot_start_epoch, then at plot_start_epoch and every plot_frequency epochs
            if self.plot_during_training:
                past_start = self.epoch >= self.plot_start_epoch
                at_frequency = (self.epoch == self.plot_start_epoch) or (
                    self.plot_frequency > 0 and (self.epoch - self.plot_start_epoch) % self.plot_frequency == 0
                )
                should_plot = past_start and at_frequency
                if should_plot:
                    self.logger.info(f"Creating plots at epoch {self.epoch} (start={self.plot_start_epoch}, frequency={self.plot_frequency})")
                    self._create_plots(self.epoch)
            
            # Track epochs since patience was exhausted (if applicable)
            if early_stopping_patience and epochs_since_best > early_stopping_patience:
                epochs_since_patience_exhausted = epochs_since_best - early_stopping_patience
            else:
                epochs_since_patience_exhausted = 0
            
            # Update progress bar with early stopping info
            postfix_dict = {
                'train': f'{epoch_train_loss:.4f}',
                'valid': f'{epoch_valid_loss:.4f}',
                'best': f'{self.best_valid_loss:.4f}',
                'lr_e': f'{lr_current_enc:.1e}',
                'lr_w': f'{lr_current_wetw:.1e}'
            }
            if getattr(self.cfg.training, "encoder_joint_train_epochs", None) is not None:
                postfix_dict['enc'] = 'F' if self._encoder_frozen_phase() else 'J'
            if early_stopping_patience:
                postfix_dict['pat'] = f'{epochs_since_best}/{early_stopping_patience}'
            epoch_pbar.set_postfix(postfix_dict)
            epoch_pbar.update(1)

            # Early stopping logic
            should_stop = False
            stop_reason = None
            elapsed_wall = timer() - train_start_wall
            
            if self.epoch >= local_max_epoch:
                should_stop = True
                stop_reason = "Maximum epochs reached"
            elif max_wall_seconds is not None and elapsed_wall >= max_wall_seconds:
                should_stop = True
                stop_reason = (
                    f"Max wall-clock limit reached ({elapsed_wall/3600:.2f}h >= {max_wall_seconds/3600:.2f}h). "
                    "Stopping and keeping best validation checkpoint so far."
                )
            elif self.cfg.training.early_stopping:
                # Check if learning rates have reached minimum
                lr_at_minimum = (lr_current_enc <= self.cfg.encoder_model.min_lr * self.cfg.encoder_model.lr_reduce_factor and 
                                (self.wetw_optimizer is None or lr_current_wetw <= self.cfg.weight_model.min_lr * self.cfg.weight_model.lr_reduce_factor))
                
                # Early stopping logic: stop when patience exhausted, optionally wait for LR minimum
                if early_stopping_patience:
                    if epochs_since_best >= early_stopping_patience:
                        # Patience exhausted - check if we should wait for LR minimum
                        if early_stopping_max_wait_after_patience is not None:
                            # Stop if LR at minimum OR max wait time exceeded
                            if lr_at_minimum:
                                should_stop = True
                                stop_reason = (f"Early stopping: validation loss hasn't improved for {epochs_since_best} epochs "
                                             f"(patience={early_stopping_patience}) and learning rates reached minimum "
                                             f"(enc: {lr_current_enc:.2e}, wetw: {lr_current_wetw:.2e})")
                            elif epochs_since_patience_exhausted >= early_stopping_max_wait_after_patience:
                                should_stop = True
                                stop_reason = (f"Early stopping: validation loss hasn't improved for {epochs_since_best} epochs "
                                             f"(patience={early_stopping_patience}). "
                                             f"Stopped after waiting {epochs_since_patience_exhausted} epochs for LR to reach minimum "
                                             f"(max_wait={early_stopping_max_wait_after_patience}). "
                                             f"Current LR: enc={lr_current_enc:.2e}, wetw={lr_current_wetw:.2e}")
                            else:
                                # Still waiting for LR to reach minimum
                                self.logger.info(f"Patience exhausted ({epochs_since_best}/{early_stopping_patience}), "
                                               f"waiting for LR to reach minimum ({epochs_since_patience_exhausted}/{early_stopping_max_wait_after_patience} epochs). "
                                               f"Current LR: enc={lr_current_enc:.2e}, wetw={lr_current_wetw:.2e}")
                        else:
                            # No max wait configured - stop immediately when patience exhausted
                            should_stop = True
                            stop_reason = f"Early stopping: validation loss hasn't improved for {epochs_since_best} epochs (patience={early_stopping_patience})"
                else:
                    # No patience set: stop when LR reaches minimum (fallback behavior)
                    if lr_at_minimum:
                        should_stop = True
                        stop_reason = "Early stopping: learning rates reached minimum"
            
            if should_stop:


                self.logger.info(f'Stopping after a total of {self.epoch} epochs. Reason: {stop_reason}')
                print(f'Stopping after a total of {self.epoch} epochs.')
                print(f'Reason: {stop_reason}')

                # loading best models so far (if they exist)
                if encoder_best_sd is not None:
                    self.encoder_model.load_state_dict(encoder_best_sd)
                    if wetw_best_sd is not None and self.wetw_model is not None:
                        self.wetw_model.load_state_dict(wetw_best_sd)
                else:
                    self.logger.warning("No best model found, using current model state")

                # train_emb: train only (not train+valid) so it matches y_train for NCC/Clust
                train_data_for_emb = self.db[:self.train_size]
                emb_batch_size = int(
                    getattr(getattr(self.cfg, 'produce_embeddings', None), 'batch_size', 64)
                )
                train_shape, test_shape = save_embeddings_from_model(
                    model=self.encoder_model,
                    train_data=train_data_for_emb,
                    test_data=self.test_db,
                    train_out=self.cfg.paths.train_embedding_filepath,
                    test_out=self.cfg.paths.test_embedding_filepath,
                    device=self.device,
                    batch_size=emb_batch_size,
                )
                self.logger.info(
                    f"Saved embeddings with batch_size={emb_batch_size}: "
                    f"train={train_shape}, test={test_shape}"
                )

                # self.__evaluate_test_relative_error()  # Disabled: use run_relative_error.py after training
                epoch_pbar.close()
                break
        
        # Close progress bar if loop ends normally
        if self.epoch >= local_max_epoch:
            epoch_pbar.close()
        
        try:
            self._maybe_save_full_weight_dim_pdfs()
        except Exception as e:
            self.logger.warning("full weight dim PDFs failed: %s", e, exc_info=True)

        # Finish wandb run if enabled
        if self.use_wandb and self.wandb:
            try:
                self.wandb.finish()
                self.logger.info("wandb run finished")
            except Exception as e:
                self.logger.warning(f"Error finishing wandb run: {e}")                

    def __evaluate_test_relative_error(self) -> None:
        if not getattr(self.cfg.training, 'eval_test_relative_error', False):
            return

        try:
            self.logger.info("Evaluating test relative errors...")

            test_emb_fp = self.cfg.paths.test_embedding_filepath
            if not os.path.exists(test_emb_fp):
                raise FileNotFoundError(f"Test embeddings not found: {test_emb_fp}")

            X_test_emb = (np.load(test_emb_fp, allow_pickle=True) if test_emb_fp.endswith('.npy') else np.fromfile(test_emb_fp, dtype=np.float32)).reshape(-1, self.dim_embedding)
            if X_test_emb.shape[0] != self.size_test_db:
                self.logger.warning(
                    "Test embeddings size mismatch (expected %d, got %d).",
                    self.size_test_db,
                    X_test_emb.shape[0],
                )

            # True DTW distances among test set (load from file if available)
            window = int(self.cfg.training.warping_window * self.dim_series)
            n_pairs = self.size_test_db * (self.size_test_db - 1) // 2
            dist_fp = getattr(self.cfg.dataset, 'test_dists_fp', None)
            true_dtw = None
            if dist_fp and os.path.exists(dist_fp):
                self.logger.info("Loading DTW distances from %s", dist_fp)
                true_dtw = (np.load(dist_fp, allow_pickle=True) if dist_fp.endswith('.npy') else np.fromfile(dist_fp, dtype=np.float32)).astype(np.float32)
                if true_dtw.shape[0] != n_pairs:
                    self.logger.warning(
                        "DTW distances size mismatch (expected %d, got %d). Recomputing.",
                        n_pairs,
                        true_dtw.shape[0],
                    )
                    true_dtw = None

            if true_dtw is None:
                true_dtw = dtw.distance_matrix_fast(
                    self.test_db.astype(np.double),
                    compact=True,
                    parallel=True,
                    window=window,
                    use_pruning=False,
                )
                true_dtw = np.asarray(true_dtw, dtype=np.float32)

            # Predicted distances among test set
            pred_dists = self.__compute_predicted_test_pair_distances(X_test_emb)

            scale_factor = getattr(self.cfg.training, 'dist_pred_scale_factor', 1.0)
            if scale_factor != 1.0:
                pred_dists = pred_dists * scale_factor

            if training_scale_dists_enabled(self.cfg):
                dim_series = self.cfg.dataset.dim_series
                dim_emb = self.cfg.encoder_model.dim_emb
                pred_dists = pred_dists / np.sqrt(dim_emb)
                true_dtw = true_dtw / np.sqrt(dim_series)

            if pred_dists.shape[0] != true_dtw.shape[0]:
                raise ValueError(
                    f"Predicted/true size mismatch: pred={pred_dists.shape[0]} true={true_dtw.shape[0]}"
                )

            rel_errors = np.full_like(true_dtw, np.nan, dtype=np.float32)
            non_zero_mask = true_dtw > 1e-10
            rel_errors[non_zero_mask] = (
                (pred_dists[non_zero_mask] - true_dtw[non_zero_mask]) / true_dtw[non_zero_mask]
            )

            mean_err = float(np.nanmean(rel_errors))
            std_err = float(np.nanstd(rel_errors))

            self.logger.info("Test relative error: mean=%.6f std=%.6f", mean_err, std_err)
            print(f"Test relative error: mean={mean_err:.6f} std={std_err:.6f}")

            output_fp = self.cfg.paths.test_rel_error_fp
            rel_errors.tofile(output_fp)
            self.logger.info("Saved test relative errors to %s", output_fp)
        except Exception as e:
            self.logger.error(f"Test relative error evaluation failed: {e}")

    def __compute_predicted_test_pair_distances(self, X_test_emb: np.ndarray) -> np.ndarray:
        emb_t = torch.from_numpy(X_test_emb).to(self.device)
        n = emb_t.shape[0]
        total_pairs = n * (n - 1) // 2
        pred = np.empty(total_pairs, dtype=np.float32)
        offset = 0
        chunk = 512

        with torch.no_grad():
            if self.wetw_model is None:
                # Euclidean: use cdist for efficiency
                for i in tqdm(range(n - 1), desc="Computing predicted test distances (euclidean)"):
                    base = emb_t[i:i + 1]
                    others = emb_t[i + 1:]
                    dists = torch.cdist(base, others, p=2).squeeze(0)
                    dists_np = dists.cpu().numpy().astype(np.float32, copy=False)
                    pred[offset:offset + dists_np.shape[0]] = dists_np
                    offset += dists_np.shape[0]
            else:
                self.wetw_model.eval()
                for i in tqdm(range(n - 1), desc="Computing predicted test distances"):
                    base = emb_t[i].unsqueeze(0)
                    others = emb_t[i + 1:]
                    for start in range(0, others.shape[0], chunk):
                        end = min(start + chunk, others.shape[0])
                        batch_j = others[start:end]
                        batch_i = base.expand(batch_j.shape[0], -1)
                        weights = self.wetw_model(batch_i, batch_j)
                        dists = weighted_pairwise_distance(
                            batch_i, batch_j, weights, **wetw_distance_kwargs_from_cfg(self.cfg)
                        )
                        dists_np = dists.detach().cpu().numpy().astype(np.float32, copy=False)
                        pred[offset:offset + dists_np.shape[0]] = dists_np
                        offset += dists_np.shape[0]

        return pred
    

    def __train_epoch(self) -> tuple:
        enc_frozen = self._encoder_frozen_phase()
        if enc_frozen and not self._logged_encoder_joint_to_frozen:
            self.logger.info(
                "Epoch %d: freezing encoder (training.encoder_joint_train_epochs=%s); WETW-only phase, same pair loss.",
                self.epoch,
                getattr(self.cfg.training, "encoder_joint_train_epochs", None),
            )
            self._logged_encoder_joint_to_frozen = True

        for p in self.encoder_model.parameters():
            p.requires_grad = not enc_frozen
        cal = self.encoder_model.pred_dist_calib
        if enc_frozen and cal is not None:
            if self.wetw_model is not None:
                for p in cal.parameters():
                    p.requires_grad = True
            elif self._pred_calib_train_when_encoder_frozen():
                for p in cal.parameters():
                    p.requires_grad = True
        if enc_frozen:
            self.encoder_model.eval()
        else:
            self.encoder_model.train()
        if self.wetw_model is not None:
            self.wetw_model.train()

        run_enc_opt = self._encoder_optimizer_should_run(enc_frozen)

        losses = []
        epoch_compute_time = 0.0  # Accumulate compute-only time per batch

        # Only show batch progress bar if there are multiple batches
        show_batch_pbar = len(self.train_dataloader) > 1

        if show_batch_pbar:
            train_pbar = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f"Epoch {self.epoch} [Train]",
                leave=False,
                ncols=100,
                unit="batch"
            )
            batch_iter = train_pbar
        else:
            batch_iter = enumerate(self.train_dataloader)

        for batch_idx, (batch_data, batch_indices) in batch_iter:

            if run_enc_opt:
                self.encoder_optimizer.zero_grad()
            if self.wetw_optimizer is not None:
                self.wetw_optimizer.zero_grad()

            # Time only the forward/backward/step (excludes DataLoader wait, allocation)
            t_start = compute_time_start(self.device)
            if enc_frozen:
                with torch.no_grad():
                    batch_embedded = self.encoder_model.encode(batch_data)
            else:
                batch_embedded = self.encoder_model.encode(batch_data)

            loss = self.lossF_distances_train(
                batch_embedded,
                batch_indices,
                wetw_model=self.wetw_model,
                pred_dist_calib=self.encoder_model.pred_dist_calib,
                query_pair_residual=self.query_pair_residual,
            )

            loss.backward()

            # Gradient clipping if configured (match optimizer param splits: backbone vs WETW+calib)
            if hasattr(self.cfg.training, 'grad_clip') and self.cfg.training.grad_clip > 0:
                clip = self.cfg.training.grad_clip
                if run_enc_opt:
                    torch.nn.utils.clip_grad_norm_(
                        self._parameters_for_encoder_optimizer(), clip
                    )
                if self.wetw_model is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._parameters_for_wetw_optimizer(), clip
                    )

            if run_enc_opt:
                self.encoder_optimizer.step()
            if self.wetw_optimizer is not None:
                self.wetw_optimizer.step()
            epoch_compute_time += compute_time_elapsed(t_start, self.device)

            loss_item = loss.detach().item()
            losses.append(loss_item)
            
            # Update progress bar if shown
            if show_batch_pbar:
                avg_loss = np.mean(losses)
                train_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'batch_loss': f'{loss_item:.4f}'
                })

        if show_batch_pbar:
                train_pbar.close()
        
        # Log to file only at the end of epoch (compute-only time)
        avg_loss = np.mean(losses)
        self.logger.info('t{:d}={:.3f}s: l/dd={:.3f}'.format(self.epoch, epoch_compute_time, avg_loss))

        # Log to wandb if enabled
        if self.use_wandb and self.wandb:
            log_payload = {
                'train/epoch_loss': avg_loss,  # Epoch training loss
                'train/loss': avg_loss,  # Alias for compatibility
                'train/epoch': self.epoch,
                'train/lr_encoder': self.encoder_optimizer.param_groups[0]['lr'] if self.encoder_optimizer.param_groups else 0.0,
                'train/lr_wetw': self.wetw_optimizer.param_groups[0]['lr'] if (self.wetw_optimizer is not None and self.wetw_optimizer.param_groups) else 0.0,
                'train/epoch_time': epoch_compute_time,  # Compute-only time (excludes DataLoader wait)
                'train/encoder_frozen': 1.0 if enc_frozen else 0.0,
            }
            cal = self.encoder_model.pred_dist_calib
            if cal is not None:
                log_payload["train/pred_dist_calib_bias"] = float(cal.bias.detach().cpu())
                if cal.mode == "affine":
                    log_payload["train/pred_dist_calib_log_scale"] = float(
                        cal.log_scale.detach().cpu()
                    )
            self.wandb.log(log_payload, step=self.epoch)
        
        return avg_loss, epoch_compute_time

    def __validate_epoch(self) -> tuple:
        self.encoder_model.eval()
        if self.wetw_model is not None:
            self.wetw_model.eval()

        losses = []
        epoch_compute_time = 0.0  # Accumulate compute-only time per batch

        # Only show batch progress bar if there are multiple batches
        show_batch_pbar = len(self.valid_dataloader) > 1

        if show_batch_pbar:
            valid_pbar = tqdm(
                enumerate(self.valid_dataloader),
                total=len(self.valid_dataloader),
                desc=f"Epoch {self.epoch} [Valid]",
                leave=False,
                ncols=100,
                unit="batch"
            )
            batch_iter = valid_pbar
        else:
            batch_iter = enumerate(self.valid_dataloader)

        with torch.no_grad():
            for batch_idx, (batch_data, batch_indices) in batch_iter:

                # Time only the forward pass (excludes DataLoader wait, allocation)
                t_start = compute_time_start(self.device)
                batch_embedded = self.encoder_model.encode(batch_data)

                loss = self.lossF_distances_valid(
                    batch_embedded,
                    batch_indices,
                    wetw_model=self.wetw_model,
                    pred_dist_calib=self.encoder_model.pred_dist_calib,
                    query_pair_residual=self.query_pair_residual,
                )
                epoch_compute_time += compute_time_elapsed(t_start, self.device)

                loss_item = loss.detach().item()
                losses.append(loss_item)
                
                # Update progress bar if shown
                if show_batch_pbar:
                    avg_loss = np.mean(losses)
                    valid_pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'batch_loss': f'{loss_item:.4f}'
                    })

            # getting the learning rate for the logger (get first param group's LR)
            lr = self.encoder_optimizer.param_groups[0]['lr'] if self.encoder_optimizer.param_groups else 0.0

            avg_loss = np.mean(losses)
            self.logger.info('v{:d}={:.3f}s: l/dd={:.3f}, lr={}'.format(
                self.epoch, epoch_compute_time, avg_loss, lr))

            # Log to wandb if enabled
            if self.use_wandb and self.wandb:
                self.wandb.log({
                    'valid/epoch_loss': avg_loss,  # Epoch validation loss
                    'valid/loss': avg_loss,  # Alias for compatibility
                    'valid/epoch': self.epoch,
                    'valid/lr_encoder': lr,
                    'valid/epoch_time': epoch_compute_time  # Compute-only time (excludes DataLoader wait)
                }, step=self.epoch)
        
        if show_batch_pbar:
            valid_pbar.close()
        return avg_loss, epoch_compute_time

    def __sample(self) -> None:
        # Pre-split data (processed_valid_fp): use all_data to match pre-computed dists
        use_all_data = (
            getattr(self.cfg.dataset, 'processed_valid_fp', None) is not None
            or self.train_size < 100_000
        )
        self.train_data_np, self.train_dist_np, self.valid_data_np, self.valid_dist_np = self.sampler.sample(all_data=use_all_data)

        train_data_tensor = torch.from_numpy(
            self.train_data_np.reshape([-1, 1, self.dim_series])).to(self.device)

        self.train_dataloader = DataLoader(
            DatasetwithIdx(train_data_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # tensors are already on GPU — forked workers can't share CUDA context
        )

        self.lossF_distances_train.update_targets(self.train_dist_np)

        valid_data_tensor = torch.from_numpy(
            self.valid_data_np.reshape([-1, 1, self.dim_series])).to(self.device)

        self.valid_dataloader = DataLoader(
            DatasetwithIdx(valid_data_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.lossF_distances_valid.update_targets(self.valid_dist_np)


    def __adjust_wd(self, wd_max, wd_min):
        
        for param_group in self.encoder_optimizer.param_groups:
            current_lr = param_group['lr']

            if self.cfg.encoder_model.weight_decay_mode == 'linear':
                progress = current_lr / self.cfg.encoder_model.lr_init
                new_wd = wd_min + progress * (wd_max - wd_min) 
                param_group['weight_decay'] = new_wd
