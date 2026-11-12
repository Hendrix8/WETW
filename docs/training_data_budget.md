# Training-data budget and pair sampling

## Training protocol

The retained paper trainer uses two sampling stages. Encoder warm-up samples
at most 5,000 unordered series pairs per epoch (or all possible pairs when
there are fewer than 5,000). During joint training, `PairwiseDistDiff` uses
the BlogB batch sampler: for a batch of size `b`, it samples up to
`b * ceil(log2(b))` unordered pairs from the batch. Both stages reuse the
corresponding targets from the condensed DTW cache. The joint term therefore
does not evaluate every pair among the training series on every epoch.

The training-size driver changes the number of training series while keeping
validation, test, database, and query splits fixed. Its default subsets are
nested prefixes of one permutation (seed 42). If a full `train_dists.npy`
exists, the driver extracts the exact matching condensed distances for each
subset. If caches are absent, ordinary training creates them from the series
using the configured DTW window (default 0.1); this computes all within-split
pairs and can take substantial time and disk space. To amortize that cost over
a size sweep, generate the full cache once with `wetw targets` first.

## What the saved training-size sweep shows

Figure 7 retrains WESee and WETW on nested subsets from 250 series up to the full
training set, with the validation set, database and query set fixed. For each
dataset the shortlist is the smallest fraction at which the full model reaches
recall@10 of 0.95, and the highlighted point is the smallest tested subset that
reaches 0.95 at that fraction. Table III reports the DTW-target (CPU) and
training (GPU) cost at these sizes.

| Dataset | Shortlist | Training series | Share of the database |
|---|---:|---:|---:|
| WhaleSounds | 20% | 1K | 2.5% |
| CornellWhale | 30% | 5K | 20.4% |
| AudioMNIST-DS | 10% | 2K | 9.6% |
| DREAMER-A | 20% | 20K | 33.3% |
| LenDB | 2% | 10K | 16.7% |
| Seismic | 2% | 100K | 11.9% |
| MosquitoSound | 10% | 5K | 8.3% |
| FruitFlies | 5% | 5K | 18.7% |

Recall is measured on the 100 search queries with one seed, and the tested
sizes are discrete, so a size between two tested points may already suffice.
The plotted values are in `results/paper/figures.json` (`fig_trainsize`), and
the underlying runs, epoch counts and cost measurements are in
`results/paper/evidence/training_size/`. They describe the saved sweep and
should not be read as a guarantee that a newly initialized training run will
reproduce the same recall.
