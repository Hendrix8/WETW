#!/usr/bin/env python3
"""Run the main WETW experiment and the paper's ablations with Hydra presets.

Main model (Figures 1 and 5, Table VI): ``full`` WESee encoder and ``mlp_w`` WETW
generator, trained in stages (i) and (ii), then stage (iii): ranking refinement of the
encoder, refinement of the generator on its frozen embeddings, and kappa chosen per
shortlist on validation.

``--ablations`` adds, for every dataset:

* Table IV, distance functions on the same frozen stage-(ii) embeddings, all with the
  regression objective only: Euclidean (``ed``), diagonal (``global_w``), full matrix
  (``lowrank`` at rank d) and regression-only WETW (``mlp_w``). These runs also give
  the L_WD column of Table VI for WESee.
* Table V, a direct MLP (``mlp_direct``) on the same symmetric pair features, trained
  on the ranking-refined embeddings with the same budget, objectives and kappa rule.
* Table VII, the temporal branch: the encoder without the TFE (``sfe``), widened to
  the full encoder's parameter count within 1%, through the same complete pipeline.
  The with-TFE arm is the main model.

The Euclidean row of Table VI (column L) fine-tunes the encoder alone and is run with
``scripts/euclidean_rank.py``.
"""

import argparse, subprocess, sys
from pathlib import Path

DATASETS = [
    "whalesounds",
    "cornellwhalechallenge",
    "audiomnistds",
    "dreamera",
    "lendb",
    "w01_seismic",
    "mosquitosound",
    "fruitflies",
]
# Residual width of the no-TFE encoder that matches the full encoder's parameter
# count within 1% (paper Table VII; 193 on the long series, 70/71 on the short ones).
MATCHED_WIDTH = {
    "whalesounds": 193,
    "cornellwhalechallenge": 193,
    "audiomnistds": 193,
    "mosquitosound": 193,
    "fruitflies": 193,
    "dreamera": 70,
    "w01_seismic": 70,
    "lendb": 71,
}

p = argparse.ArgumentParser()
p.add_argument("--data-root", required=True)
p.add_argument("--output", default="outputs/paper_runs")
p.add_argument("--device", default="cuda")
p.add_argument("--datasets", nargs="+", default=DATASETS)
p.add_argument("--ablations", action="store_true")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
root = Path(a.data_root).resolve()
output = Path(a.output).resolve()
for ds in a.datasets:
    # (name, variant, head, objective, extra overrides)
    cases = [("full_mlp_w_ranking", "full", "mlp_w", "ranking", [])]
    if a.ablations:
        cases += [
            (f"full_{h}_regression", "full", h, "regression", [])
            for h in ["ed", "global_w", "lowrank", "mlp_w"]
        ]
        cases += [
            ("full_mlp_direct_ranking", "full", "mlp_direct", "ranking", []),
            (
                "notfe_matched_mlp_w_ranking",
                "sfe",
                "mlp_w",
                "ranking",
                [f"encoder_model.num_en_channels={MATCHED_WIDTH[ds]}"],
            ),
        ]
    for name, variant, head, objective, extra in cases:
        cmd = [
            sys.executable,
            "-m",
            "wetw.experiment",
            f"dataset={ds}",
            f"data.root={root}",
            f"output={output / ds / name}",
            f"device={a.device}",
            f"variant={variant}",
            f"head={head}",
            f"objective={objective}",
            "optimizer=preset",
            *extra,
        ]
        if variant != "full":
            cmd.append("reference_layout=false")
        print(" ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)
