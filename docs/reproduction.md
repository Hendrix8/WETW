# Reproducing the paper

There are two distinct workflows: run new experiments from data, or redraw the exact stored numeric evidence. The repository supports both; a figure redrawn from a JSON snapshot is not a newly measured experiment.

## Study map

Paper numbering follows the manuscript: eight search collections (Table I) and the AudioMNIST scalability study.

| Paper result | Fresh experiment entry point | Saved evidence (`results/paper/`) |
|---|---|---|
| Figure 1 (mean over eight collections) and Figure 5 (per collection): complete search, recall@10 versus seconds per query | `scripts/run_paper.py`, then `scripts/benchmark_native_fv.py`; exact searches with `scripts/benchmark_native.py --mode usp/lbk/lbw` and `scripts/build_messi.py`; downsampled DTW with `wetw benchmark --methods ds2 ds4 ds8 ds16 ds32`; NeuralWarp with `experiments/neuralwarp/nw_search10.py` | `evidence/end_to_end/<collection>/<method>/result.json`, `evidence/end_to_end/kappa_selection.json`, `evidence/neuralwarp/`, `figures.json` (`fig_e2e_mean`, `fig_e2e_v2`) |
| Table II: candidates reaching DTW with LB_Keogh or LB_Webb alone | `scripts/benchmark_native.py --mode lbk/lbw` | `evidence/end_to_end/<collection>/{lbk,lbw}/`, `tables/table_lower_bound_survival.tex` |
| Figure 6: AudioMNIST scalability (database size, series length, warping window) | length/window Hydra presets, `scripts/reproduce_scalability.py`, then stage (iii) as below | `scalability_points.json`, `baseline_scalability.json`, `evidence/scalability_controls/`, `selections/`, `figures.json` (`fig_amnist`) |
| Figure 7 and Table III: training-set size, DTW-target (CPU) and training (GPU) cost, break-even | `scripts/training_size.py` | `evidence/training_size/`, `evidence/offline_cost/offline_cost.json`, `tables/table_offline_cost.tex`, `figures.json` (`fig_trainsize`) |
| Table IV: Euclidean, diagonal and full-matrix distances versus regression-only WETW on frozen WESee embeddings | `scripts/run_paper.py --ablations` (`full_{ed,global_w,lowrank,mlp_w}_regression`) | `evidence/heads/`, `evidence/static_controls/`, `tables/table_heads.tex` |
| Table V: WETW versus a direct MLP in WETW-FV | `scripts/run_paper.py --ablations` (`full_mlp_direct_ranking`) | `evidence/direct_mlp/`, `tables/table_direct_mlp.tex` |
| Table VI: native objective, $\mathcal{L}_{\mathrm{WD}}$ and $\mathcal{L}$ for every encoder, with Euclidean distance and WETW | WESee: `scripts/run_paper.py --ablations` and `scripts/euclidean_rank.py`; baselines: `wetw-baseline` and `experiments/baselines`; NeuralWarp: `experiments/neuralwarp/{neuralwarp_v2,nw_rank}.py` | `evidence/heads/`, `evidence/euclidean_rank/`, `evidence/components/`, `tables/table_components.tex` |
| Table VII: temporal branch, capacity-matched encoder without the TFE | `scripts/run_paper.py --ablations` (`notfe_matched_mlp_w_ranking`); the with-TFE arm is the main model | `evidence/tfe_capacity_matched/`, `tables/table_architecture_capacity_matched.tex` |
| Training seeds (Section V-A: recall@1 0.948 ± 0.004, recall@10 0.896 ± 0.008 at a 2% shortlist) | `wetw-train dataset=<name> release.seed=<seed>` | `evidence/seeds/` |

`evidence/main_models/<collection>.json` holds the validation and test recall of each main model for every prefilter ratio and shortlist size. The objective $\mathcal{L}_{\mathrm{WD}}$ of the paper is called `warpdistill` in the code, and $\mathcal{L}$ (with ranking regularization) is `objective=ranking`.

## Protocols

**Main model (stages i–iii).** Stage (i) warms up the encoder on the MSE between Euclidean embedding distances and DTW targets. Stage (ii) trains encoder and generator jointly with $\mathcal{L}_{\mathrm{WD}}$. Stage (iii) first refines the encoder for 3,000 updates with the Euclidean distance and the regression + ranking objective (Adam `1e-4`, 32 regression pairs and 16 ranking triplets per update), keeping the checkpoint among updates 0, 300, 1,000 and 3,000 with the best validation recall@1; it then freezes the encoder and refines the generator on its stored embeddings with a cosine schedule for the preset's 30K or 100K updates. `wetw-train` runs all three stages; the stage (iii) encoder refinement is `src/wetw/encoder_ranking.py` (`encoder_ranking.enabled=false` skips it).

**Prefilter ratio $\kappa$.** For every shortlist fraction $\rho$, `wetw-train` evaluates $\kappa\in\{1,2,3,5,10,20\}$ and whole-database rescoring (`0`) on validation queries and keeps the smallest $\kappa$ whose validation recall@1 and recall@10 are both within 0.005 of the best value of each measure; if none qualifies, it keeps the highest sum of the two. The choice is written to `prefilter_selection.json` and can be passed to `wetw search`, `wetw benchmark` and `scripts/benchmark_native_fv.py` with `--kappa <path>`. The paper's choices are in `evidence/end_to_end/kappa_selection.json`.

**Table IV, distance functions.** All four distances share the frozen stage (ii) WESee embeddings, the regression objective, the training-pair sampler, the refinement budget and the validation scale correction. The full-matrix measure is `head=lowrank` at full rank (`head_rank=null`, 128 × 128). These runs also give the $\mathcal{L}_{\mathrm{WD}}$ column of Table VI for WESee.

**Table V, direct MLP.** `head=mlp_direct` predicts the distance itself from the same symmetric pair features $[|\mathbf e_i-\mathbf e_j|,\mathbf e_i\odot\mathbf e_j,\mathbf e_i+\mathbf e_j]$, starting from the Euclidean distance, and goes through the same stage (iii), budget and $\kappa$ rule as WETW. `evidence/direct_mlp/properties_*` records the self-distance ratio $\hat d_{ii}/d_{\mathrm{NN}}$.

**Table VI, Euclidean ranking control.** Euclidean distance has no generator, so `scripts/euclidean_rank.py` applies the stage (iii) encoder refinement alone and evaluates the Euclidean shortlist. For the baseline encoders, the WETW cells train a generator on each encoder's embeddings with $\mathcal{L}_{\mathrm{WD}}$ and with $\mathcal{L}$ under the same protocol ([baseline guide](baselines.md)).

**Table VII, temporal branch.** The encoder without the TFE (`variant=sfe`) is widened to the full encoder's parameter count within 1% (`encoder_model.num_en_channels`: 193 on the long series, 70 on DREAMER-A and Seismic, 71 on LenDB) and goes through the complete pipeline, including stage (iii) and the $\kappa$ rule.

**NeuralWarp.** `experiments/neuralwarp/neuralwarp_v2.py` trains the authors' unchanged model natively or with $\mathcal{L}_{\mathrm{WD}}$; `nw_search10.py` scores the full database for each of the 100 queries on a GPU and records the ranks of all ten exact neighbours, and `nw_rank.py` adds the ranking term to its warping network (Table VI). Set `NW_BUNDLE` to the folder holding these scripts and their `specs.json`.

## Main WETW model

```bash
wetw-train dataset=lendb data.root=/path/to/data device=cuda optimizer=preset
python scripts/run_paper.py --data-root /path/to/data --device cuda --dry-run
python scripts/run_paper.py --data-root /path/to/data --device cuda --ablations
```

The default encoder backend is the retained paper trainer. `src/wetw/_reference/` was copied from the archived study code with package-relative imports and the portable `mpmath` dependency. Per-dataset training schedules are pinned in the presets; `optimizer=preset` keeps their archived optimizers (SGD for the encoder and Adam for the generator on the five long-series presets, AdamW on DREAMER-A, LenDB and Seismic), while the package default `optimizer=adamw` trains every dataset with AdamW, which gave higher validation recall in an optimizer-only swap.

The main comparison uses two full-encoder implementation layouts of the same architecture: one on CornellWhale, AudioMNIST-DS and DREAMER-A, and another on the other five. The dataset presets select `reference_layout=true` automatically for the former. New random initializations are not guaranteed to recover the identical selected run; the three-seed spread is in `evidence/seeds/seed_stats.json`.

## Complete search with native verification

```bash
python scripts/benchmark_native_fv.py \
  --checkpoint outputs/lendb/model.pt --data data/lendb \
  --output outputs/lendb/native_fv --device cpu \
  --kappa outputs/lendb/prefilter_selection.json
```

This includes query encoding, Euclidean prefiltering, WETW rescoring, shortlist selection and UCR-Suite verification with EAPrunedDTW. Database encoding is offline. The lower-level `wetw benchmark` uses individual `dtaidistance` calls for verification: it is convenient for correctness and method comparisons but its latency is not the paper's cascade latency.

Each invocation uses one CPU thread per query. GPU ranking (NeuralWarp) is timed with synchronization. Compare fresh timings on the same machine.

## Training-size study

```bash
python scripts/training_size.py --data-root data --dataset lendb \
  --sizes 250 500 1000 2000 5000 10000 --device cuda
```

The driver draws nested training subsets from one seeded permutation and keeps the validation, test, database and query splits fixed. It restricts an existing condensed training cache to exactly the same row identities, and its `subset.json` records every row. See [`training_data_budget.md`](training_data_budget.md) for the pair-sampling details and the saved Figure 7 results.

## AudioMNIST scalability

Length presets: `audiomnist_L1000`, `L2000`, `L4000`, `L8000`, `L16000`, `L47998` (all prefixed `audiomnist_`). Window presets add `_w0.01`, `_w0.025`, `_w0.05`, `_w0.2` to `audiomnist_L4000`. Each needs its own prepared split directory.

First fit the initial model at each setting, keeping the reference encoder layout:

```bash
wetw-train dataset=audiomnist_L4000 reference_layout=true stage=fit optimizer=preset \
  data.root=data output=outputs/initial/audiomnist_L4000 device=cuda
```

Then run the retained follow-up, which fine-tunes the encoder and the generator with the validation-selected settings in `configs/paper/scalability*.json`:

```bash
python scripts/reproduce_scalability.py --data-root data \
  --initial-root outputs/initial --output outputs/scalability \
  --device cuda --task 3 --dry-run
```

Omit `--dry-run` to execute; omit `--task` to run every setting. Inspect `configs/paper/scalability.json` for task indices. Stage (iii) then refines that encoder with ranking supervision (`scripts/euclidean_rank.py`), refines the generator on the new embeddings for 30K or 100K updates chosen on validation, and picks $\kappa$ with the rule above. The paper's choices for each setting are in `selections/<setting>/selection.json` (generator budget) and `prefilter_selection.json` ($\kappa$).

Database-size results use nested prefixes of 3,250/6,500/13,000/26,000 series at L=47,998; use the same row order. Measure the resulting models with the native benchmark.

## Saved evidence and limitations

```bash
wetw paper
```

`figures.json` holds every plotted point of Figures 1, 5, 6 and 7, exported from the paper's plotting scripts; `wetw paper` redraws them with a simpler, portable layout and copies the LaTeX tables. `manifest.json` records timing provenance: NeuralWarp ranks on an A100 GPU, every other method on one CPU thread of the same CPU model. Individual result JSONs keep their measurement fields. Missing results remain missing.

Raw data, intermediate checkpoints and original split IDs are not part of this Git repository. The final trained models of Figures 1, 5 and 6 are in `pretrained/`; the models of the training-size study, the seed repeats, the ablations and the learned baselines are not. Full-scale performance cannot be validated by a small CPU smoke test and depends on the exact data and version, seeds, kernels and hardware.

## Resuming and changing configuration

Use separate output directories for separate datasets, architectures and objectives. A completed encoder fit is reused when `resume=true`. The archived trainer has explicit resume options in `training`, but does not restore the entire optimizer state automatically, so an interrupted fit may restart. The stage (iii) encoder refinement resumes from its last validation checkpoint, and generator refinement restores model, optimizer, cosine scheduler, RNG and best validation state.

The release's inference distance returns exactly zero for identical embeddings; optimization keeps the original epsilon-stabilized square root. This avoids a positive numerical self-distance in the public API without changing the trained parameters.
