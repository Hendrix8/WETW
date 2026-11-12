# coding = utf-8
"""Produce embeddings from a trained encoder given an experiment folder and dataset path."""

import os
import numpy as np
import torch
from omegaconf import OmegaConf

from wetw._reference.model.builder import ModelBuilder
from wetw._reference.util.helper_functions import pad_with_zeros


def compute_embeddings_from_array(model, data, device, batch_size):
    """Embed an in-memory 2D array [N, T] with shape-safe batching.

    Returns (N, dim_emb) float32.
    """
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
            encoded = model.encode(batch_tensor)
            if isinstance(encoded, tuple):
                emb = encoded[0].cpu().numpy()
            else:
                emb = encoded.cpu().numpy()
            embeddings.append(emb)
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
    return np.vstack(embeddings).astype(np.float32)


def save_embeddings_from_model(model, train_data, test_data, train_out, test_out, device, batch_size):
    """Compute and save train/test embeddings from arrays using one shared implementation."""
    if batch_size <= 0:
        raise ValueError("produce_embeddings.batch_size must be > 0")
    X_train_emb = compute_embeddings_from_array(model, train_data, device, batch_size)
    X_test_emb = compute_embeddings_from_array(model, test_data, device, batch_size)
    np.save(train_out, X_train_emb)
    np.save(test_out, X_test_emb)
    return X_train_emb.shape, X_test_emb.shape


def load_data_from_config(cfg):
    """Load X_train, X_test from dataset config (processed_fp, or processed_train_fp/processed_test_fp)."""
    if hasattr(cfg.dataset, 'processed_fp') and cfg.dataset.processed_fp:
        fp = cfg.dataset.processed_fp
        if os.path.exists(str(fp)) and str(fp).endswith('.npz'):
            d = np.load(fp)
            return d['X_train'].astype(np.float32), d['X_test'].astype(np.float32)
    if hasattr(cfg.dataset, 'processed_train_fp') and hasattr(cfg.dataset, 'processed_test_fp'):
        train_fp = cfg.dataset.processed_train_fp
        test_fp = cfg.dataset.processed_test_fp
        if os.path.exists(train_fp) and os.path.exists(test_fp):
            if str(train_fp).endswith('.npy') and str(test_fp).endswith('.npy'):
                return np.load(train_fp).astype(np.float32), np.load(test_fp).astype(np.float32)
    raise FileNotFoundError(
        "Dataset files not found. Set dataset.processed_fp (.npz) or "
        "dataset.processed_train_fp and dataset.processed_test_fp (.npy)"
    )


def produce_embeddings(cfg):
    """
    Load encoder from experiment folder, load data from dataset config, produce and save embeddings.

    Requires:
        - cfg.produce_embeddings.produce_embeddings_only: True
        - cfg.produce_embeddings.exps_dir: path to experiment folder (encoder.pt, config.yaml)
        - Dataset paths in cfg.dataset (processed_fp or processed_train_fp/processed_test_fp)

    Saves train_emb.npy and test_emb.npy to output_dir (default: exps_dir).
    """
    pe = cfg.produce_embeddings
    exps_dir = pe.exps_dir or getattr(cfg, 'exps_dir', None)
    if not exps_dir or not os.path.isdir(exps_dir):
        raise ValueError(
            "produce_embeddings.exps_dir must be set to an existing experiment folder. "
            "Example: produce_embeddings.exps_dir=/path/to/CornellWhaleChallenge-2026-02-24-10-45-31"
        )

    # Resolve to actual experiment dir if parent was given
    if not os.path.exists(os.path.join(exps_dir, 'config.yaml')):
        subdirs = [
            d for d in os.listdir(exps_dir)
            if os.path.isdir(os.path.join(exps_dir, d))
            and os.path.exists(os.path.join(exps_dir, d, 'config.yaml'))
        ]
        if subdirs:
            subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(exps_dir, x)), reverse=True)
            exps_dir = os.path.join(exps_dir, subdirs[0])
            print(f"Using latest experiment: {exps_dir}")

    exp_config_path = os.path.join(exps_dir, 'config.yaml')
    if not os.path.exists(exp_config_path):
        raise FileNotFoundError(f"Config not found: {exp_config_path}")

    exp_cfg = OmegaConf.load(exp_config_path)
    OmegaConf.resolve(exp_cfg)
    # Merge: experiment config for model, current config for dataset paths and device
    merged = OmegaConf.merge(exp_cfg, cfg)
    merged.exps_dir = exps_dir
    OmegaConf.resolve(merged)

    output_dir = pe.output_dir or exps_dir
    os.makedirs(output_dir, exist_ok=True)

    device = str(merged.system.device)
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    # Load encoder
    encoder_fp = os.path.join(exps_dir, 'encoder.pt')
    if not os.path.exists(encoder_fp):
        raise FileNotFoundError(f"Encoder not found: {encoder_fp}")

    model = ModelBuilder(merged)
    state_dict = torch.load(encoder_fp, map_location=device)
    has_encoder_prefix = any(k.startswith('encoder.') for k in state_dict.keys())
    if has_encoder_prefix:
        model.load_state_dict(state_dict, strict=False)
    else:
        model_state_dict = {f'encoder.{k}': v for k, v in state_dict.items()}
        model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    model.to(device)

    target_dim = merged.dataset.dim_series
    X_train, X_test = load_data_from_config(merged)
    train_dim, test_dim = X_train.shape[1], X_test.shape[1]

    if train_dim != target_dim:
        X_train = pad_with_zeros(X_train, target_dim)
    if test_dim != target_dim:
        X_test = pad_with_zeros(X_test, target_dim)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    embed_batch_size = int(getattr(pe, 'batch_size', 64))
    train_out = os.path.join(output_dir, 'train_emb.npy')
    test_out = os.path.join(output_dir, 'test_emb.npy')
    train_shape, test_shape = save_embeddings_from_model(
        model=model,
        train_data=X_train,
        test_data=X_test,
        train_out=train_out,
        test_out=test_out,
        device=device,
        batch_size=embed_batch_size,
    )

    print(f"Saved embeddings: {train_out} ({train_shape}), {test_out} ({test_shape})")
    return train_out, test_out
