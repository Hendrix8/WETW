"""Deterministic toy data for executable checks, never a paper measurement."""

from pathlib import Path
import numpy as np
from .data import prepare_npz


def prepare_smoke(directory, seed=42):
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, 32)
    arrays = {}
    offset = 0
    for name, n in [
        ("train", 24),
        ("valid", 16),
        ("test", 16),
        ("database", 32),
        ("queries", 4),
    ]:
        freq = rng.randint(1, 4, size=(n, 1))
        phase = rng.uniform(-0.4, 0.4, size=(n, 1))
        arrays[name] = (np.sin(freq * t + phase) + 0.05 * rng.randn(n, len(t))).astype(
            np.float32
        )
        arrays[name + "_ids"] = np.arange(offset, offset + n)
        offset += n
    np.savez(directory / "source.npz", **arrays)
    prepare_npz(directory / "source.npz", directory)
