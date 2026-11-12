# Pretrained models

Every model reported in the paper's search experiments, one file per model: the eight search collections of Figures 1 and 5 and the ten AudioMNIST settings of Figure 6. Each file holds the encoder after the stage (iii) ranking refinement, the validation-selected generator refined on its embeddings, the training configuration and the prefilter ratio $\kappa$ chosen on validation with the paper's rule. None is a synthetic smoke checkpoint.

Shared properties:

- Embedding dimension: 128; generator: `mlp_w`, hidden width 128.
- Input: independently z-normalized univariate series of the model's length.
- The release drops an unused training-calibration key; tensor weights are otherwise unchanged. Each file reproduces the paper's stored query embeddings on CPU (relative difference at most 2e-4).
- The public distance returns zero for identical embeddings instead of the optimization epsilon.

## Search collections (Figures 1 and 5)

`kappa` is a map from shortlist fraction $\rho$ to $\kappa$ (`0` rescores the whole database); `results/paper/evidence/end_to_end/kappa_selection.json` records the same selection.

| File | Collection | Length | Encoder parameters | $\kappa$ at $\rho$ = 2% |
|---|---|---:|---:|---:|
| `whalesounds.pt` | WhaleSounds | 2,500 | 2.95M | 20 |
| `cornellwhalechallenge.pt` | CornellWhale | 4,000 | 2.95M | 20 |
| `audiomnistds.pt` | AudioMNIST-DS | 4,000 | 2.95M | 5 |
| `dreamera.pt` | DREAMER-A | 256 | 0.56M | 10 |
| `lendb.pt` | LenDB | 540 | 0.58M | 5 |
| `w01_seismic.pt` | Seismic | 256 | 0.56M | 5 |
| `mosquitosound.pt` | MosquitoSound | 3,750 | 2.95M | 5 |
| `fruitflies.pt` | FruitFlies | 5,000 | 2.95M | 5 |

## AudioMNIST scalability (Figure 6)

One $\kappa$ per setting, selected at the 2% shortlist; `results/paper/selections/am2k_*/` records the generator and $\kappa$ selections. All use the reference full WESee layout with TFE (3.0M encoder parameters).

| File | Length | DTW window | Generator updates | $\kappa$ |
|---|---:|---:|---:|---:|
| `audiomnist_L1000.pt` | 1,000 | 10% | 30K | 2 |
| `audiomnist_L2000.pt` | 2,000 | 10% | 30K | 2 |
| `audiomnist_L4000.pt` | 4,000 | 10% | 30K | 3 |
| `audiomnist_L8000.pt` | 8,000 | 10% | 30K | 3 |
| `audiomnist_L16000.pt` | 16,000 | 10% | 30K | 2 |
| `audiomnist_L47998.pt` | 47,998 | 10% | 30K | 2 |
| `audiomnist_L4000_w0.01.pt` | 4,000 | 1% | 30K | 3 |
| `audiomnist_L4000_w0.025.pt` | 4,000 | 2.5% | 100K | 3 |
| `audiomnist_L4000_w0.05.pt` | 4,000 | 5% | 30K | 10 |
| `audiomnist_L4000_w0.2.pt` | 4,000 | 20% | 100K | 3 |

## Loading

```python
from wetw.evaluation import kappa_for
from wetw.experiment import load_checkpoint

encoder, head, metadata = load_checkpoint("pretrained/lendb.pt")
kappa = kappa_for(metadata["kappa"], 0.02)
```

`metadata["config"]` gives the length (`dataset.dim_series`) and DTW window (`training.warping_window`). Checkpoint portability was checked on CPU. Data-domain performance is specific to each collection; input length alone does not establish applicability to another dataset.
