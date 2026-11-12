#!/usr/bin/env python3
"""Import existing paper arrays/caches without guessing their distance units."""

import argparse, shutil
from pathlib import Path
import numpy as np
from wetw.data import fingerprint, write_json

p = argparse.ArgumentParser()
p.add_argument("--samples", required=True)
p.add_argument("--database", required=True)
p.add_argument("--queries", required=True)
p.add_argument("--output", required=True)
p.add_argument(
    "--cache-format", required=True, choices=["squared", "sqrt", "normalized"]
)
a = p.parse_args()
src = Path(a.samples)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
files = {s: src / f"{s}_sample.npy" for s in ["train", "valid", "test"]}
files.update(database=Path(a.database), queries=Path(a.queries))
manifest = {
    "split_ids_verified": False,
    "normalization": "supplied unchanged",
    "cache_input_format": a.cache_format,
    "files": {},
}
length = None
for name, f in files.items():
    x = np.load(f, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"Expected matrix: {f}")
    if length is None:
        length = x.shape[1]
    if x.shape[1] != length:
        raise ValueError("Split lengths differ")
    target = out / f"{name}.npy"
    if target.resolve() != f.resolve():
        shutil.copy2(f, target)
    manifest["files"][name] = {"shape": list(x.shape), "sha256": fingerprint(target)}
    cache = src / f"{name}_dists.npy"
    if name in ["train", "valid", "test"] and cache.exists():
        d = np.load(cache, mmap_mode="r")
        n = len(x)
        if d.shape != (n * (n - 1) // 2,):
            raise ValueError(f"Wrong condensed-cache length: {cache}")
        temp = out / f"{name}_dists.tmp.npy"
        dest = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.float32, shape=d.shape
        )
        for start in range(0, len(d), 1000000):
            values = np.asarray(d[start : start + 1000000], dtype=np.float64)
            if a.cache_format == "sqrt":
                values = values**2
            elif a.cache_format == "normalized":
                values = values**2 * length
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid targets: {cache}")
            dest[start : start + len(values)] = values
        dest.flush()
        del dest
        temp.replace(out / f"{name}_dists.npy")
write_json(out / "manifest.json", manifest)
print(out)
