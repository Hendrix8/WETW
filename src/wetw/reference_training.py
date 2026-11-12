"""Portable entry point to the retained paper trainer (not a new training recipe)."""

from pathlib import Path
import copy
import logging
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from .data import distance, write_json


def make_condensed(series, path, window=0.1, squared=False):
    """Materialize a SciPy-order upper triangle; training archive uses squared DTW."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    n = len(series)
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(n * (n - 1) // 2,)
    )
    at = 0
    for i in range(n):
        for j in range(i + 1, n):
            value = distance(series[i], series[j], window)
            out[at] = value * value * series.shape[1] if squared else value
            at += 1
    out.flush()
    del out
    tmp.replace(path)


def train_reference(cfg, data, out):
    from ._reference.util.train import Train
    from ._reference import architecture, light_generator
    from ._reference.model import builder

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(c, False)
    c.system.device = str(cfg.device)
    c.encoder_model.arch_ablation = str(cfg.variant)
    opt = str(cfg.get("optimizer", "preset")).lower()
    if opt == "adamw":
        c.encoder_model.optimizer = "adamw"
        c.weight_model.optimizer = "adamw"
    elif opt != "preset":
        raise ValueError(f"optimizer must be adamw or preset, got {opt!r}")
    if cfg.reference_layout:
        from ._reference.model.ResidualAE import ResidualEncoder

        builder.ResidualEncoder = ResidualEncoder
    else:
        architecture.install()
    light_generator.install()
    c.exps_dir = str(out)
    c.logging.use_wandb = False
    c.logging.log_filepath = str(out / "training.log")
    c.logging.plot_during_training = False
    c.training.eval_test_relative_error = False
    filenames = {
        "best_encoder_sd_fp": "encoder.pt",
        "best_wetw_sd_fp": "wetw.pt",
        "conf_fp": "config.yaml",
        "train_embedding_filepath": "embeddings_train.npy",
        "test_embedding_filepath": "embeddings_test.npy",
        "db_sorted_indices_path": "db_sorted_indices.bin",
        "rel_error_result_fp": "rel_error.npy",
        "test_rel_error_fp": "test_rel_error.npy",
        "train_val_dist_prefix": "dists",
        "seasam_sample_filepath": "seasam_sample.bin",
        "seasam_sample_idx_filepath": "seasam_sample_idxs.bin",
        "seasam_libpath": "unused_for_explicit_splits.so",
    }
    c.paths = {k: str(out / v) for k, v in filenames.items()}
    for split in ["train", "valid", "test"]:
        c.dataset["processed_" + split + "_fp"] = str(Path(data) / f"{split}.npy")
        target = Path(data) / f"{split}_dists.npy"
        if split != "test":
            if not target.exists():
                # Preserve the archived trainer's cache interface, but allow a
                # normal run to start from series alone. This is quadratic in
                # the split size and writes the complete condensed cache.
                make_condensed(
                    np.load(c.dataset["processed_" + split + "_fp"]),
                    target,
                    c.training.warping_window,
                    squared=True,
                )
            c.dataset["dists_" + split + "_fp"] = str(target)
    start = time.perf_counter()
    trainer = Train(c)
    trainer.train()
    for name in ["encoder.pt", "wetw.pt"]:
        if not (out / name).exists():
            raise RuntimeError(f"Trainer did not save {name}; inspect training.log")
    write_json(
        out / "fit.json",
        {
            "complete": True,
            "seconds": time.perf_counter() - start,
            "epoch": int(trainer.epoch),
            "backend": "archived paper trainer",
            "smoke": bool(cfg.smoke),
        },
    )
    return out
