# Data and distance conventions

## Portable split format

Place five arrays in `data/<preset>/`: `train.npy`, `valid.npy`, `test.npy`, `database.npy`, and `queries.npy`. Each array is a finite, C-contiguous, univariate float32 matrix `(N,L)`. Series length must match the preset. Training, validation, pairwise test, database and query identities must be disjoint.

`wetw prepare --input splits.npz --output data/lendb` writes this layout, normalizes each series independently using its population standard deviation, and records SHA-256 fingerprints. Provide `train_ids`, `valid_ids`, `test_ids`, `database_ids`, and `queries_ids` to verify disjoint identities. Without IDs the importer explicitly records that identity disjointness could not be checked. Equal recordings may legitimately have equal values; a content hash alone cannot establish distinct provenance.

An NPZ can be created from existing splits:

```python
import numpy as np
np.savez("splits.npz",
    train=np.load("train_sample.npy"), valid=np.load("valid_sample.npy"),
    test=np.load("test_sample.npy"), database=np.load("db.npy"),
    queries=np.load("q100.npy"))
```

Use `--already-normalized` to retain verified paper arrays unchanged. Do not normalize concatenated train/validation data jointly. Optional `train_labels.npy` is used only by label-supervised baseline objectives.

## Dataset sources and paper sizes

Seven labelled collections come from the MONSTER archive; Seismic is unlabelled. AudioMNIST scalability uses a separate full-length AudioMNIST setup. Consult the MONSTER distribution and the underlying dataset terms before redistributing recordings. The release contains model weights and numeric results, not raw recordings.

| Collection | Train | Validation | Pairwise test | Database | Search queries | L |
|---|---:|---:|---:|---:|---:|---:|
| WhaleSounds | 30K | 10K | 10K | 40K | 100 | 2,500 |
| CornellWhale | 20K | 4.5K | 4.5K | 24.5K | 100 | 4,000 |
| MosquitoSound | 50K | 10K | 10K | 60K | 100 | 3,750 |
| FruitFlies | 20K | 6.8K | 6.8K | 26.8K | 100 | 5,000 |
| DREAMER-A | 20K | 5K | 5K | 60K | 100 | 256 |
| LenDB | 20K | 5K | 5K | 60K | 100 | 540 |
| AudioMNIST-DS | 5K | 2K | 2K | 20.9K | 100 | 4,000 |
| Seismic | 100K | 50K | 10K | 840K | 100 | 256 |
| AudioMNIST (scalability) | 2K | 300 | 300 | 26K | 100 | up to 47,998 |

Counts are those reported in the manuscript. Downloading another version and randomly splitting it does not recreate the exact paper identities. Exact numerical replication requires the original split arrays/IDs and initial checkpoints. No alternative random split is silently called the paper split.

The [MONSTER paper](https://arxiv.org/abs/2502.15122) identifies the archive; [AudioMNIST](https://github.com/soerenab/AudioMNIST) describes the original spoken-digit corpus. The manuscript bibliography is the authority for the specific collection versions.

## Three distance representations

For squared alignment cost `C` and length `L`:

- Public `distance(a,b)` and fidelity/search targets: `sqrt(C/L)`.
- Archived encoder-training caches `train_dists.npy`, `valid_dists.npy`: **C**, in SciPy condensed upper-triangle order. The retained trainer takes a square root and divides by `sqrt(L)` itself.
- The retained external-baseline and Figure 6 follow-up scripts consume **sqrt(C)**. Their launchers explicitly convert the public squared caches while staging inputs.

Do not rename a square-root cache and pass it to the encoder trainer. Values can remain finite while the resulting supervision becomes incorrect.

Build squared training caches with:

```bash
wetw targets --data data/lendb --squared
```

This is an explicit all-pairs preparation option. A float32 condensed cache occupies `2*N*(N-1)` bytes: about 0.8 GB for 20K series and 20 GB for 100K series. The archived trainer makes additional in-memory copies; provide enough RAM. Target generation can be much more expensive than fitting. The release's sparse `Targets` API can compute selected pairs on demand for custom training, but does not change the archived trainer's cache protocol.

The package retains `dtaidistance`'s `window=floor(w*L)` convention, whose maximum displacement is `window-1`. The native C engine uses the matching radius. At zero window the portable API explicitly evaluates the diagonal path. All timings should report this convention.

## Import existing research folders

The following importer copies original arrays unchanged and **requires** the distance units to be declared:

```bash
python scripts/import_research_layout.py --samples /path/to/samples_npy \
  --database /path/to/db.npy --queries /path/to/q100.npy \
  --cache-format sqrt --output data/lendb
```

Use `sqrt` for ordinary `dtaidistance.distance_fast` cache values, `squared` for alignment costs, and `normalized` for `sqrt(cost/L)`. Conversion is chunked and writes the squared caches required by this release's encoder trainer. Preserve source split IDs separately when they exist.
