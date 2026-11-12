import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle as cPickle
import wandb

# Add lib/NeuTS to path
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.environ["WETW_NEUTS_ROOT"]
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from tools import config
from geo_rnns.neutraj_trainer import NeuTrajTrainer, pad_sequence
from geo_rnns.neutraj_model import NeuSeq_Network
from geo_rnns.wrloss import WeightedRankingLoss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_neuts')

def sanitize_time_series(X, dim_series):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X)
    if X.ndim == 1:
        X = X.reshape(-1, dim_series)
    return X

def normalize_data(X, min_v, max_v):
    # Normalize to [0, 1]
    return (X - min_v) / (max_v - min_v + 1e-10)

def compute_grid_indices(X_norm, delta=0.001):
    # For 1D: index = int(val / delta)
    idx = (X_norm / delta).astype(int)
    max_idx = int(1.0 / delta)
    return np.clip(idx, 0, max_idx)

def compact_to_square(compact, n):
    D = np.zeros((n, n), dtype=np.float32)
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            D[i, j] = compact[idx]
            D[j, i] = compact[idx]
            idx += 1
    return D

class UCRNeuTSTrainer(NeuTrajTrainer):
    """Custom trainer for UCR datasets using samples_npy and WETW scaling."""
    
    def __init__(self, cfg):
        # Update NeuTS global config from Hydra cfg
        config.learning_rate = cfg.training.lr
        config.batch_size = cfg.training.batch_size
        config.sampling_num = cfg.training.sampling_num
        config.epochs = cfg.training.n_epochs
        config.patience = cfg.training.patience
        config.d = cfg.model.dim_emb
        config.dimensional = 1
        config.recurrent_unit = cfg.model.recurrent_unit
        config.stard_unit = cfg.model.stard_unit
        config.incell = cfg.model.incell
        config.mail_pre_degree = cfg.training.mail_pre_degree
        config.GPU = cfg.system.device.replace("cuda:", "") if "cuda" in cfg.system.device else "0"
        
        # Ensure exp_path exists and set logger_filepath
        exp_path = os.path.join(cfg.paths.exps_dir, cfg.dataset.dataset_name)
        os.makedirs(exp_path, exist_ok=True)
        config.logger_filepath = os.path.join(exp_path, 'fit.log')
        
        super().__init__(tagset_size=config.d, 
                         batch_size=config.batch_size, 
                         sampling_num=config.sampling_num)
        self.cfg = cfg

    def prepare_ucr_data(self, X_train, X_valid, train_dists_compact, valid_dists_compact):
        """Prepare data using NeuTS grid+value format."""
        # 1. Global normalization (based on train)
        min_v = X_train.min()
        max_v = X_train.max()
        
        X_train_norm = normalize_data(X_train, min_v, max_v)
        X_valid_norm = normalize_data(X_valid, min_v, max_v) if X_valid is not None else None
        
        # 2. Grid indices
        grid_train = compute_grid_indices(X_train_norm, delta=self.cfg.model.grid_delta)
        grid_valid = compute_grid_indices(X_valid_norm, delta=self.cfg.model.grid_delta) if X_valid is not None else None
        
        # 3. Concatenate [value, grid_index] for each timestep
        # X is (N, T), grid is (N, T)
        # NeuTS expects (N, T, 2)
        def build_input(X_norm, grid):
            N, T = X_norm.shape
            out = np.zeros((N, T, 2), dtype=np.float32)
            out[:, :, 0] = X_norm
            out[:, :, 1] = grid.astype(np.float32)
            return out

        self.train_seqs = build_input(X_train_norm, grid_train)
        self.valid_seqs = build_input(X_valid_norm, grid_valid) if X_valid is not None else None
        
        self.trajs_length = [X_train.shape[1]] * X_train.shape[0]
        self.valid_trajs_length = [X_valid.shape[1]] * X_valid.shape[0] if X_valid is not None else None
        
        self.grid_size = [int(1.0/self.cfg.model.grid_delta) + 1, 1, 1]
        self.max_length = X_train.shape[1]
        
        # Padded version (actually already uniform length for UCR)
        self.padded_trajs = self.train_seqs
        self.train_trajs_length = self.trajs_length
        
        # 4. Target distances (Scaled by sqrt(T) as per WETW)
        T = X_train.shape[1]
        # Compact squared DTW -> square -> sqrt -> scale
        self.train_distance = np.sqrt(np.maximum(compact_to_square(train_dists_compact, len(X_train)), 0)) / np.sqrt(T)
        self.valid_distance = np.sqrt(np.maximum(compact_to_square(valid_dists_compact, len(X_valid)), 0)) / np.sqrt(T) if valid_dists_compact is not None else None
        self.distance = self.train_distance

    def batch_generator(self, train_seqs, train_distance):
        j = 0
        from tools import sampling_methods as sm
        while j < len(train_seqs):
            anchor_input, trajs_input, negative_input, distance, negative_distance = [], [], [], [], []
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []
            
            # Handle end of dataset
            current_batch_size = min(self.batch_size, len(train_seqs) - j)
            if current_batch_size <= 0: break

            for i in range(current_batch_size):
                sampling_index_list = sm.distance_sampling(self.distance, len(train_seqs), j + i)
                negative_sampling_index_list = sm.negative_distance_sampling(self.distance, len(train_seqs), j + i)

                trajs_input.append(train_seqs[j + i])
                anchor_input.append(train_seqs[j + i])
                negative_input.append(train_seqs[j + i])
                
                if j + i not in batch_trajs_keys:
                    batch_trajs_keys[j + i] = 0
                    batch_trajs_input.append(train_seqs[j + i])
                    batch_trajs_len.append(self.trajs_length[j + i])

                anchor_input_len.append(self.trajs_length[j + i])
                trajs_input_len.append(self.trajs_length[j + i])
                negative_input_len.append(self.trajs_length[j + i])
                
                distance.append(1)
                negative_distance.append(1)
                
                for traj_index in sampling_index_list:
                    anchor_input.append(train_seqs[j + i])
                    trajs_input.append(train_seqs[traj_index])
                    anchor_input_len.append(self.trajs_length[j + i])
                    trajs_input_len.append(self.trajs_length[traj_index])
                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[traj_index] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
                    distance.append(np.exp(-float(train_distance[j + i][traj_index]) * config.mail_pre_degree))

                for traj_index in negative_sampling_index_list:
                    negative_input.append(train_seqs[traj_index])
                    negative_input_len.append(self.trajs_length[traj_index])
                    negative_distance.append(np.exp(-float(train_distance[j + i][traj_index]) * config.mail_pre_degree))
                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[traj_index] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
            
            max_anchor_length = max(anchor_input_len)
            max_sample_length = max(trajs_input_len)
            max_neg_length = max(negative_input_len)
            
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_length)
            negative_input = pad_sequence(negative_input, maxlen=max_neg_length)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_length, max_neg_length))

            yield (
                [np.array(anchor_input), np.array(trajs_input), np.array(negative_input), np.array(batch_trajs_input)],
                [anchor_input_len, trajs_input_len, negative_input_len, batch_trajs_len],
                [np.array(distance), np.array(negative_distance)]
            )
            j = j + self.batch_size

    def run_training(self):
        """Train similar to neutraj_train but with ranking loss on scaled distances."""
        spatial_net = NeuSeq_Network(2, self.target_size, self.grid_size, self.batch_size,
                                     self.sampling_num,
                                     stard_LSTM=config.stard_unit, incell=config.incell)
        
        spatial_net.cuda()
        
        optimizer = torch.optim.Adam(spatial_net.parameters(), lr=config.learning_rate)
        
        mse_loss_m = WeightedRankingLoss(sampling_num=self.sampling_num)
        mse_loss_m.cuda()
        objective = getattr(self.cfg.training, "objective", "native")  # "native" or "wd" (WarpDistill)
        logger.info(f"objective: {objective}")
        if objective == "wd":
            import types

            def _distances(net, inputs_arrays, inputs_len_arrays):
                # NeuSeq_Network.forward up to the embeddings; returns Euclidean distances, not exp(-d)
                emb = []
                for k in range(3):
                    x = torch.Tensor(inputs_arrays[k])
                    h = net._get_initial_hidden(x.size(0))
                    emb.append(net.rnn([x.cuda(), inputs_len_arrays[k]], h))
                return (F.pairwise_distance(emb[0], emb[1], p=2), F.pairwise_distance(emb[0], emb[2], p=2))

            spatial_net.forward = types.MethodType(_distances, spatial_net)

            def _warpdistill(dp, tp, dn, tn):
                d_hat = torch.cat([dp.view(-1), dn.view(-1)])
                d = -torch.log(torch.cat([tp.view(-1), tn.view(-1)]).clamp_min(1e-30)) / config.mail_pre_degree
                m = d > 1e-8  # drop the anchor-with-itself entries (target exp(0) = 1)
                return F.huber_loss((d_hat[m] - d[m]) / d[m], torch.zeros_like(d[m]), delta=0.1)

            mse_loss_m = _warpdistill
        
        wandb_mode = getattr(self.cfg.system, "wandb_mode", "online")
        wandb.init(
            project=getattr(self.cfg.system, "wandb_project", "dtwrl_baselines"),
            name=self.cfg.dataset.dataset_name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            reinit=True,
            mode=wandb_mode,
        )
        
        best_loss = np.inf
        exp_path = os.path.join(self.cfg.paths.exps_dir, self.cfg.dataset.dataset_name)
        os.makedirs(exp_path, exist_ok=True)
        
        patience_counter = 0
        wall_start = time.time()
        time_limit_hours = getattr(self.cfg.training, "time_limit_hours", 0)
        time_limit_sec = time_limit_hours * 3600 if time_limit_hours > 0 else None
        stopped_by_time_limit = False
        epochs_completed = 0

        if time_limit_sec:
            logger.info(f"Training with time limit: {time_limit_hours}h ({time_limit_sec:.0f}s)")
        
        # resume: epoch-level checkpoint; the elapsed training time counts against the time budget
        ckpt_fp = os.path.join(exp_path, "resume.pt")
        start_epoch = 0
        if os.path.exists(ckpt_fp):
            ck = torch.load(ckpt_fp, map_location="cuda")
            spatial_net.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["rng"])
            start_epoch, best_loss, patience_counter = ck["epoch"], ck["best_loss"], ck["patience"]
            epochs_completed = start_epoch
            wall_start = time.time() - ck["elapsed"]
            logger.info(f"Resumed from epoch {start_epoch} ({ck['elapsed']:.0f}s of budget used)")

        for epoch in range(start_epoch, config.epochs):
            spatial_net.train()
            epoch_losses = []
            
            pbar = tqdm(self.batch_generator(self.train_seqs, self.train_distance), 
                       total=len(self.train_seqs)//self.batch_size, desc=f"Epoch {epoch+1}")
            
            for i, batch in enumerate(pbar):
                inputs_arrays, inputs_len_arrays, target_arrays = batch[0], batch[1], batch[2]

                trajs_loss, negative_loss = spatial_net(inputs_arrays, inputs_len_arrays)

                positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1)).cuda()
                negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1)).cuda()

                loss = mse_loss_m(trajs_loss, positive_distance_target,
                                  negative_loss, negative_distance_target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())

            epochs_completed = epoch + 1
            avg_loss = np.mean(epoch_losses)
            elapsed = time.time() - wall_start
            logger.info(f"Epoch {epochs_completed} Average Loss: {avg_loss:.6f}  (elapsed: {elapsed:.0f}s)")
            wandb.log({"train_loss": avg_loss, "epoch": epochs_completed})
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(spatial_net.state_dict(), os.path.join(exp_path, "best_model.pth"))
                logger.info(f"Saved best model with loss {best_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info("Early stopping")
                    break

            torch.save({"model": spatial_net.state_dict(), "optimizer": optimizer.state_dict(),
                        "np_rng": np.random.get_state(), "rng": torch.get_rng_state(), "epoch": epochs_completed,
                        "best_loss": best_loss, "patience": patience_counter, "elapsed": time.time() - wall_start},
                       ckpt_fp + ".tmp")
            os.replace(ckpt_fp + ".tmp", ckpt_fp)

            if time_limit_sec and elapsed >= time_limit_sec:
                logger.info(f"Time limit reached ({elapsed:.0f}s >= {time_limit_sec:.0f}s). Stopping training.")
                stopped_by_time_limit = True
                torch.save(spatial_net.state_dict(), os.path.join(exp_path, "last_model.pth"))
                break
        
        train_time = time.time() - wall_start
        training_completed = (not stopped_by_time_limit) and (patience_counter < config.patience or epochs_completed >= config.epochs)

        train_meta = {
            "dataset": self.cfg.dataset.dataset_name,
            "epochs_completed": epochs_completed,
            "max_epochs": config.epochs,
            "best_loss": float(best_loss),
            "training_completed": training_completed,
            "stopped_by_time_limit": stopped_by_time_limit,
            "time_limit_sec": time_limit_sec,
            "train_time_sec": round(train_time, 2),
        }
        with open(os.path.join(exp_path, "train_meta.json"), "w") as f:
            json.dump(train_meta, f, indent=2)
        logger.info(f"Training finished: {epochs_completed} epochs in {train_time:.1f}s  (time_limited={stopped_by_time_limit})")

        with open(os.path.join(exp_path, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
            
        wandb.finish()

@hydra.main(config_path=".", config_name="config_neuts", version_base=None)
def main(cfg: DictConfig):
    ds_dir = os.path.dirname(cfg.dataset.dataset_fp)
    snpy_dir = os.path.join(ds_dir, "samples_npy")
    
    if not os.path.isdir(snpy_dir):
        raise FileNotFoundError(f"samples_npy not found in {ds_dir}")
        
    X_train_raw = np.load(os.path.join(snpy_dir, "train_sample.npy"))
    X_valid_raw = np.load(os.path.join(snpy_dir, "valid_sample.npy"))

    dim_series = X_train_raw.shape[1]
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.dim_series = dim_series
    OmegaConf.set_struct(cfg, True)

    X_train = sanitize_time_series(X_train_raw, dim_series)
    X_valid = sanitize_time_series(X_valid_raw, dim_series)
    
    train_dists = np.load(os.path.join(snpy_dir, "train_dists.npy"))
    valid_dists = np.load(os.path.join(snpy_dir, "valid_dists.npy"))
    
    exp_path = os.path.join(cfg.paths.exps_dir, cfg.dataset.dataset_name)
    os.makedirs(exp_path, exist_ok=True)
    norm_params = {"min_v": float(X_train.min()), "max_v": float(X_train.max())}
    with open(os.path.join(exp_path, "norm_params.pkl"), "wb") as f:
        cPickle.dump(norm_params, f)

    trainer = UCRNeuTSTrainer(cfg)
    trainer.prepare_ucr_data(X_train, X_valid, train_dists, valid_dists)
    trainer.run_training()

if __name__ == "__main__":
    main()
