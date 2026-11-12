"""Explicit disjoint splits, reproducible data fingerprints, and DTW targets."""

from pathlib import Path
import hashlib
import json
import numpy as np
from dtaidistance import dtw

SPLITS = ("train", "valid", "test", "database", "queries")


def znorm(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Series must be a finite (n,length) array with length >= 2")
    return np.ascontiguousarray(
        (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8),
        dtype=np.float32,
    )


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def prepare_npz(source, destination, normalize=True):
    """Import existing splits. Source keys must be train/valid/test/database/queries.

    Optional <split>_ids permit exact auditing of split disjointness. Without
    IDs this function does not claim that row-index disjointness is verified.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        missing = set(SPLITS) - set(data.files)
        if missing:
            raise ValueError(f"Missing splits: {sorted(missing)}")
        ids = {
            s: set(map(str, data[s + "_ids"]))
            for s in SPLITS
            if s + "_ids" in data.files
        }
        for i, a in enumerate(ids):
            for b in list(ids)[i + 1 :]:
                if ids[a] & ids[b]:
                    raise ValueError(f"Overlapping IDs: {a} and {b}")
        manifest = {
            "normalization": "per-series population z-score"
            if normalize
            else "supplied unchanged",
            "source_sha256": fingerprint(source),
            "split_ids_verified": len(ids) == 5,
            "splits": {},
        }
        length = None
        for s in SPLITS:
            x = (
                znorm(data[s])
                if normalize
                else np.ascontiguousarray(data[s], dtype=np.float32)
            )
            if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
                raise ValueError(f"Invalid {s} array")
            if length is None:
                length = x.shape[1]
            if x.shape[1] != length:
                raise ValueError("All splits must have the same series length")
            np.save(destination / f"{s}.npy", x)
            if s + "_ids" in data.files:
                np.save(destination / f"{s}_ids.npy", data[s + "_ids"])
            manifest["splits"][s] = {
                "shape": list(x.shape),
                "sha256": fingerprint(destination / f"{s}.npy"),
            }
    write_json(destination / "manifest.json", manifest)
    return manifest


def distance(a, b, window=0.1):
    """sqrt(sum squared alignment errors / L), matching the paper targets.

    dtaidistance's window convention is retained: window=floor(window*L).
    A zero radius is evaluated explicitly as the diagonal path.
    """
    a, b = (
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
    )
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError("DTW expects equal-length vectors")
    if not 0 <= window <= 1:
        raise ValueError("window must be a fraction in [0,1]")
    radius = int(window * len(a))
    if radius == 0:
        return float(np.linalg.norm(a - b) / np.sqrt(len(a)))
    value = dtw.distance_fast(a, b, window=radius, use_pruning=True)
    # The C pruning upper bound can round below the optimum on near ties.
    if not np.isfinite(value):
        value = dtw.distance_fast(a, b, window=radius, use_pruning=False)
    return float(value / np.sqrt(len(a)))


def cross_distances(left, right, window=0.1):
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i, j] = distance(a, b, window)
    return result


def condensed_index(i, j, n):
    a, b = np.minimum(i, j), np.maximum(i, j)
    return n * a - a * (a + 1) // 2 + b - a - 1


class Targets:
    """Sparse on-demand symmetric target cache, or an existing condensed array."""

    def __init__(self, series, window=0.1, condensed=None, squared_cache=False):
        self.series, self.window = series, window
        self.cache = {}
        self.condensed = condensed
        self.squared_cache = squared_cache
        if condensed is not None and condensed.shape != (
            len(series) * (len(series) - 1) // 2,
        ):
            raise ValueError("Condensed cache has the wrong length")

    def pairs(self, i, j):
        out = []
        series_length = self.series.shape[1]
        for a, b in zip(i, j):
            a, b = sorted((int(a), int(b)))
            if a == b:
                out.append(0.0)
                continue
            if self.condensed is not None:
                value = self.condensed[condensed_index(a, b, len(self.series))]
                out.append(
                    np.sqrt(value / series_length) if self.squared_cache else value
                )
                continue
            key = (a, b)
            if key not in self.cache:
                self.cache[key] = distance(self.series[a], self.series[b], self.window)
            out.append(self.cache[key])
        return np.asarray(out, dtype=np.float32)

    def row(self, i):
        return self.pairs(np.full(len(self.series), i), np.arange(len(self.series)))
