"""Hydra launchers for retained baseline trainers in a source checkout."""

from pathlib import Path
import json, os, subprocess, sys
import hydra
import numpy as np
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="conf", config_name="baseline")
def main(c):
    root = Path(__file__).resolve().parents[2]
    # Editable/source installs keep experiments beside src. Wheels intentionally
    # contain the reusable WETW package; use the checkout for external trainers.
    scripts = root / "experiments/baselines"
    if not scripts.exists():
        raise RuntimeError(
            "Baseline trainers require the source checkout (pip install -e .)."
        )
    data = Path(c.data.root).resolve() / c.release.dataset
    out = Path(c.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(c, out / "resolved_config.yaml")
    staged = out / "input"
    samples = staged / c.release.dataset / "samples_npy"
    samples.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        dest = samples / f"{split}_sample.npy"
        if not dest.exists():
            dest.symlink_to(data / f"{split}.npy")
        cache = data / f"{split}_dists.npy"
        if cache.exists():
            # Shared public caches contain squared DTW. The retained baseline
            # scripts expect sqrt-DTW in their condensed arrays.
            np.save(
                samples / f"{split}_dists.npy", np.sqrt(np.load(cache, mmap_mode="r"))
            )
    env = {**os.environ, "WANDB_MODE": "disabled", "WETW_DEVICE": str(c.device)}
    method = str(c.method)
    objective = str(c.objective)
    if objective not in ["native", "warpdistill"]:
        raise ValueError("objective must be native or warpdistill")
    if method == "ldps":
        env.update(
            LDPS_SD=str(samples),
            LDPS_DB=str(data / "database.npy"),
            LDPS_Q=str(data / "queries.npy"),
            LDPS_HOURS=str(c.hours),
            LDPS_MAX_ITER=str(c.max_steps),
            LDPS_LOSS="native" if objective == "native" else "wd",
        )
        cmd = [
            sys.executable,
            str(scripts / "run_ldps.py"),
            str(c.release.dataset),
            str(out / "run"),
        ]
    elif method in ["rfdtan", "neuts"]:
        upstream = c.upstream[method]
        if upstream is None:
            raise ValueError(
                f"Set upstream.{method} to the patched upstream checkout; see docs/baselines.md"
            )
        env["WETW_" + ("RFDTAN" if method == "rfdtan" else "NEUTS") + "_ROOT"] = str(
            Path(upstream).resolve()
        )
        d = staged / c.release.dataset
        labels = (
            np.load(data / "train_labels.npy")
            if (data / "train_labels.npy").exists()
            else np.zeros(
                len(np.load(data / "train.npy", mmap_mode="r")), dtype=np.int64
            )
        )
        np.savez(
            d / "bundle.npz",
            X_train=np.load(data / "train.npy", mmap_mode="r"),
            y_train=labels,
            X_test=np.load(data / "valid.npy", mmap_mode="r"),
            y_test=np.zeros(
                len(np.load(data / "valid.npy", mmap_mode="r")), dtype=np.int64
            ),
        )
        cmd = [
            sys.executable,
            str(scripts / f"scripts_{method}/train_{method}.py"),
            f"dataset.dataset_name={c.release.dataset}",
            f"dataset.dataset_fp={d / 'bundle.npz'}",
            f"paths.exps_dir={out / 'runs'}",
            f"model.dim_emb={c.embedding_dim}",
            f"system.device={c.device}",
            "system.wandb_mode=disabled",
            f"training.time_limit_hours={c.hours}",
        ]
        if method == "rfdtan":
            cmd += [
                f"dataset.dim_series={c.dataset.dim_series}",
                "model.n_recurrences=4",
                "+training.objective=" + ("native" if objective == "native" else "dtw"),
            ]
        else:
            cmd += [
                "+training.objective=" + ("native" if objective == "native" else "wd")
            ]

    elif method == "cdps":
        cmd = [
            sys.executable,
            str(scripts / "train_cdps.py"),
            "--data",
            str(data),
            "--output",
            str(out / "run"),
            "--dimension",
            str(c.embedding_dim),
            "--fit-samples",
            str(c.fit_samples),
            "--device",
            str(c.device),
            "--objective",
            objective,
        ]
    else:
        raise ValueError("Available methods: ldps, cdps, rfdtan, neuts")
    if method == "rfdtan":
        if objective == "native" and (data / "train_labels.npy").exists():
            cmd.append(f"+training.labels_fp={data / 'train_labels.npy'}")
        elif objective == "warpdistill":
            cmd.append("+training.smooth_prior=true")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
