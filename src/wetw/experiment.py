"""Hydra experiment entry point: wetw-train dataset=lendb data.root=/data."""

from pathlib import Path
import json
import logging
import random
import time
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from .data import SPLITS, write_json, fingerprint
from .models import WESee, WETW
from .search import SearchIndex
from .reference_training import train_reference
from .refine import refine
from .encoder_ranking import refine_encoder
from .evaluation import evaluate_search, select_prefilter, pair_fidelity


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = OmegaConf.create(ck["config"])
    cfg.system.device = device
    encoder = WESee(cfg, ck["variant"], ck.get("reference_layout", False)).to(device)
    head = WETW(
        encoder.dimension, kind=ck["head_kind"], rank=ck.get("head_rank", 64)
    ).to(device)
    encoder.load_state_dict(ck["encoder"])
    head.load_state_dict(ck["head"])
    return encoder.eval(), head.eval(), ck


def run(cfg):
    OmegaConf.set_struct(cfg, False)
    torch.set_num_threads(int(cfg.threads))
    seed = int(cfg.release.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use device=cpu")
    cfg.system.device = str(cfg.device)
    data = Path(cfg.data.root).expanduser().resolve() / str(cfg.release.dataset)
    out = Path(cfg.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if cfg.smoke:
        from .demo import prepare_smoke

        prepare_smoke(data, seed)
        cfg.dataset.dim_series = 32
        cfg.encoder_model.num_en_resblock = 2
        cfg.encoder_model.multi_temporal_head_kernels = [3, 5, 7]
        cfg.encoder_model.time_fusion_pool_target = 16
    required = (
        ["train", "valid", "test"] if cfg.stage in ("fit", "refine") else list(SPLITS)
    )
    for name in required:
        p = data / f"{name}.npy"
        if not p.is_file():
            raise FileNotFoundError(
                f"{p}; import disjoint splits with wetw prepare (see docs/data.md)"
            )
        a = np.load(p, mmap_mode="r")
        if a.ndim != 2 or a.shape[1] != cfg.dataset.dim_series:
            raise ValueError(
                f"{p}: expected length {cfg.dataset.dim_series}, received {a.shape}"
            )
    if cfg.stage not in ("all", "fit", "refine", "evaluate"):
        raise ValueError("stage must be all, fit, refine or evaluate")
    if cfg.objective not in ("ranking", "regression"):
        raise ValueError("objective must be ranking or regression")
    scientific_keys = (
        "encoder_model",
        "training",
        "dataset",
        "variant",
        "head",
        "objective",
        "reference_layout",
        "refinement",
        "encoder_ranking",
        "smoke",
    )
    signature = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        if OmegaConf.is_config(cfg[key])
        else cfg[key]
        for key in scientific_keys
    }
    signature["seed"] = seed
    signature["data"] = {
        name: fingerprint(data / f"{name}.npy") for name in ("train", "valid", "test")
    }
    signature_path = out / "run_signature.json"
    if (
        cfg.resume
        and signature_path.exists()
        and json.loads(signature_path.read_text()) != signature
    ):
        raise ValueError(
            "Output belongs to different data or training settings; choose a new output directory or explicitly set resume=false."
        )
    write_json(signature_path, signature)
    OmegaConf.save(cfg, out / "resolved_config.yaml")
    write_json(
        out / "data_provenance.json",
        {
            name: {
                "path": str(data / f"{name}.npy"),
                "sha256": fingerprint(data / f"{name}.npy"),
            }
            for name in required
        },
    )
    fit = out / "fit"
    model_path = out / "model.pt"
    if cfg.stage in ("all", "fit"):
        if not (cfg.resume and (fit / "fit.json").exists()):
            train_reference(cfg, data, fit)
        if cfg.stage == "fit":
            return {"stage": "fit", "output": str(fit)}
    if cfg.checkpoint:
        encoder, head, ck = load_checkpoint(cfg.checkpoint, str(cfg.device))
    else:
        encoder = WESee(cfg, str(cfg.variant), bool(cfg.reference_layout)).to(
            cfg.device
        )
        raw = torch.load(fit / "encoder.pt", map_location=cfg.device, weights_only=True)
        incompat = encoder.load_state_dict(raw, strict=False)
        if incompat.missing_keys or any(
            not k.startswith("pred_dist_calib.") for k in incompat.unexpected_keys
        ):
            raise RuntimeError(str(incompat))
        head = WETW(
            encoder.dimension,
            kind=str(cfg.head),
            rank=int(cfg.head_rank or encoder.dimension),
        ).to(cfg.device)
        if cfg.head == "mlp_w" and cfg.refinement.warm_start:
            head.load_state_dict(
                torch.load(fit / "wetw.pt", map_location=cfg.device, weights_only=True)
            )
    encoder.eval()
    # Stage (iii): ranking refinement of the encoder, then of the generator on its frozen
    # embeddings. The regression-only ablations keep the stage-(ii) encoder.
    if (
        cfg.stage in ("all", "refine")
        and not cfg.checkpoint
        and cfg.objective == "ranking"
        and cfg.encoder_ranking.enabled
    ):
        encoder, _ = refine_encoder(
            encoder,
            data,
            out / "encoder_ranking",
            str(cfg.device),
            float(cfg.training.warping_window),
            steps=int(cfg.encoder_ranking.steps),
            lr=float(cfg.encoder_ranking.lr),
            seed=seed,
            resume=bool(cfg.resume),
        )
        for prm in encoder.parameters():
            prm.requires_grad_(False)
    embeddings = {}
    with torch.inference_mode():
        for split in ("train", "valid"):
            series = np.load(data / f"{split}.npy", mmap_mode="r")
            parts = []
            for st in range(0, len(series), cfg.training.embed_batch):
                parts.append(
                    encoder(
                        torch.as_tensor(
                            np.array(series[st : st + cfg.training.embed_batch]),
                            device=cfg.device,
                        )
                    )
                    .cpu()
                    .numpy()
                )
            embeddings[split] = np.concatenate(parts)
    if cfg.stage in ("all", "refine"):
        head = refine(cfg, encoder, head, data, embeddings, out / "refine")
        torch.save(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "head_kind": str(cfg.head),
                "head_rank": int(head.rank),
                "variant": str(cfg.variant),
                "reference_layout": bool(cfg.reference_layout),
                "smoke": bool(cfg.smoke),
            },
            model_path,
        )
        if cfg.stage == "refine":
            return {"stage": "refine", "checkpoint": str(model_path)}
    elif cfg.stage == "evaluate" and not cfg.checkpoint:
        encoder, head, ck = load_checkpoint(model_path, str(cfg.device))
    valid = np.load(data / "valid.npy", mmap_mode="r")
    selection = select_prefilter(
        embeddings["valid"],
        head,
        valid,
        cfg.training.warping_window,
        list(cfg.search.rho_grid),
        list(cfg.search.kappa_grid),
        cfg.search.validation_queries,
        str(cfg.device),
        float(cfg.search.kappa_tolerance),
    )
    write_json(out / "prefilter_selection.json", selection)
    # Test/query/database splits are opened only after the selection is frozen.
    db = np.load(data / "database.npy", mmap_mode="r")
    queries = np.load(data / "queries.npy", mmap_mode="r")
    if cfg.search.query_limit is not None:
        queries = queries[: cfg.search.query_limit]
    index = SearchIndex(
        encoder, head, db, cfg.training.warping_window, str(cfg.device), normalized=True
    )
    result = {
        "dataset": str(cfg.release.dataset),
        "smoke": bool(cfg.smoke),
        "protocol": "stages (i)-(ii) encoder training, stage (iii) encoder and generator ranking refinement, kappa per shortlist on validation",
        "kappa": selection["kappa"],
        "search": evaluate_search(
            index,
            queries,
            cfg.search.rho_grid,
            min(cfg.search.k, len(db)),
            selection["kappa"],
        ),
        "timing_device": str(cfg.device),
        "threads": int(cfg.threads),
    }
    test = np.load(data / "test.npy", mmap_mode="r")
    et = index.encode(test).cpu().numpy()
    # Validation-only scalar calibration; test targets never choose the scale.
    vpair = pair_fidelity(
        embeddings["valid"],
        head,
        valid,
        cfg.training.warping_window,
        pairs=64 if cfg.smoke else 10000,
        device=str(cfg.device),
    )
    calibration = max(1e-6, 1 + vpair["mean_signed_re"])
    result["calibration"] = calibration
    result["fidelity"] = pair_fidelity(
        et,
        head,
        test,
        cfg.training.warping_window,
        pairs=64 if cfg.smoke else 20000,
        calibration=calibration,
        device=str(cfg.device),
    )
    if cfg.search.baseline:
        static = WETW(encoder.dimension, kind="ed").to(cfg.device).eval()
        edindex = SearchIndex(
            encoder,
            static,
            db,
            cfg.training.warping_window,
            str(cfg.device),
            normalized=True,
        )
        result["euclidean_search"] = evaluate_search(
            edindex, queries, cfg.search.rho_grid, min(cfg.search.k, len(db)), 1
        )
    write_json(out / "metrics.json", result)
    from .report import plot_run

    plot_run(out / "metrics.json", out / "search.png")
    print(json.dumps(result, indent=2))
    return result


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
