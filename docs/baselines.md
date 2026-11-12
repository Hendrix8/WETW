# Baselines and third-party implementations

External methods keep their own architectures and objectives. They are optional: the main WETW demo and tests do not require TensorFlow, or RF-DTAN's CUDA extensions.

## Hydra entry point

```bash
pip install -e '.[baselines]'
wetw-baseline --cfg job --resolve
wetw-baseline dataset=lendb method=ldps data.root=data device=cuda
wetw-baseline dataset=lendb method=ldps objective=warpdistill data.root=data device=cuda
wetw-baseline dataset=lendb method=cdps data.root=data device=cuda
```

`method` selects LDPS, CDPS, RF-DTAN or NeuTS. `objective=native` is the default; `warpdistill` requests the paper's $\mathcal{L}_{\mathrm{WD}}$ objective, the relative Huber error on DTW distances (marked by the subscript WD in the paper). Output paths are separate by method and objective. The staging directory converts the shared squared-DTW caches to the distance format expected by each retained script.

LDPS uses 128 shapelets across lengths 0.15L/0.30L/0.45L, Adagrad at 0.01, 64 sampled pairs and a six-hour default budget. CDPS uses 128 shapelets of length 0.2L and 300 iterations; `fit_samples` defaults to 2,000 for long series. CDPS's retained implementation computes its own training targets, so inspect its source when comparing preprocessing costs.

The release includes short CPU checks of both shapelet implementations. These do not establish full-dataset convergence or reproduce the six-hour budgets.

## RF-DTAN

Obtain the [authors' repository](https://github.com/BGU-CS-VIL/RF-DTAN), originally pinned in this study to commit `3e631ad`. The retained trainer expects the upstream InceptionTime/difw implementation. The historical adjustments are recorded in `third_party/patches/RF-DTAN_UPSTREAM_VERSION.txt` and the companion patch scripts: configurable 128-dimensional output, training-objective selection, real class labels for native ICAE, and gradient-backend fixes.

```bash
wetw-baseline dataset=lendb method=rfdtan data.root=data \
  upstream.rfdtan=/path/to/patched/RF-DTAN device=cuda
```

The native and $\mathcal{L}_{\mathrm{WD}}$ variants use four recurrences. Native training uses `train_labels.npy` when supplied; without labels the original unsupervised single-class convention applies. RF-DTAN requires a working difw backend. Do not silently replace a failed warp-training run with a no-warp run.

## NeuTS

Use the [authors' NeuTS code](https://github.com/yaodi833/NeuTS), not an unrelated RNN replacement. The retained driver imports `tools.config` and `geo_rnns`. Its older upstream dependencies can require a separate environment.

```bash
wetw-baseline dataset=lendb method=neuts data.root=data \
  upstream.neuts=/path/to/NeuTS device=cuda
```

The matching native/WD changes are retained under `third_party/patches/`. Dataset-specific sampling and preprocessing remain part of NeuTS. Do not compare its model directly on raw, unprocessed arrays.

## NeuralWarp

`experiments/neuralwarp/neuralwarp_v2.py` trains the unchanged model natively or with $\mathcal{L}_{\mathrm{WD}}$ and selects it on validation data, `nw_search10.py` runs the full-database search of Figures 1 and 5, and `nw_rank.py` adds the ranking term (Table VI). They require the authors' **time-series** NeuralWarp repository exposing `cnn_models.CNNWarpedSim` and `hyper_parameters.HyperParams`; the computer-vision project with the same name is a different method.

The adapters use the TensorFlow 1 graph API. Use the compatible upstream environment or a verified `tensorflow.compat.v1` adaptation; the default WETW PyTorch environment does not install TensorFlow. The native version additionally requires class labels. Optional PAA caps the model input on long series; exact DTW targets and verification must stay at full length.

```bash
# train and validation-select one run of the bundle (native or L_WD, see the bundle's specs)
python experiments/neuralwarp/neuralwarp_v2.py train --bundle <bundle> --task <i> --hours 6
# score the database for the 100 queries in shards on a GPU, then aggregate
python experiments/neuralwarp/neuralwarp_v2.py search --bundle <bundle> --task <i> --shard 0
python experiments/neuralwarp/neuralwarp_v2.py aggregate --bundle <bundle> --task <i>
```

The bundle holds the specs (data paths, PAA factor, objective) of each run. Ranking is pairwise and runs on a GPU in the paper; whole-query ranking and CPU verification of the shortlist are both counted in Figures 1 and 5.

## MESSI-DTW

The paper uses MESSI-DTW through the [DaiSy library](https://github.com/MChatzakis/DaiSy), under its own MIT license. Supply a DaiSy checkout:

```bash
python scripts/build_messi.py --daisy /path/to/DaiSy
```

The retained runner takes flat float32 arrays:

```text
outputs/bin/messi_run db.f32 queries.f32 N L NQ radius k output_prefix threads
```

Set `threads=1` for the paper protocol. Convert C-contiguous normalized arrays with `np.asarray(x, dtype=np.float32).tofile(path)`. Use radius `floor(window*L)-1`, matching the native UCR and dtaidistance conventions. Validate returned distances and recall against the same ground truth; never hardcode recall to one because the algorithm is intended to be exact.

## Reusing WETW with another encoder

The public `WETW(dimension=...)` module takes two embedding tensors. Train it using `warpdistill` and `log_ranking`, then pass the encoder and learned head to `SearchIndex`. Match the encoder's original normalization and input preprocessing. Baseline embeddings can have very different scales; the original frozen-head experiments used feature normalization for some encoders, so a new raw-embedding head should not be labelled the identical paper control without that normalization.

## Scope and attribution

LDPS/CDPS implementations here are the GPU research adaptations used by this project, not new claims of authorship of the algorithms. The source comments and historical patches retain their attribution. The retained encoder utilities also include AdaNorm support credited to its original authors; the default paper path does not replace upstream baseline models with it.

External baseline adapters are included for reproduction and configuration transparency. Their full GPU runs, TensorFlow compatibility and external CUDA builds are not covered by the core CPU test suite. The saved paper tables/evidence remain available independently of those dependencies.
