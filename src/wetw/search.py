"""Euclidean prefilter -> learned reranking -> exact DTW verification."""

from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from .data import distance, znorm


@dataclass
class SearchResult:
    indices: np.ndarray
    distances: np.ndarray
    encoding_seconds: float
    ranking_seconds: float
    verification_seconds: float
    shortlist_size: int
    prefilter_size: int

    @property
    def seconds(self):
        return self.encoding_seconds + self.ranking_seconds + self.verification_seconds


class SearchIndex:
    """Fixed-length, univariate filter-and-verify search.

    Returned distances are exact normalized band-DTW distances, but neighbours
    may be missed when rho < 1. rho=1 recovers exact search. ``database`` and
    queries are z-normalized independently unless ``normalized=True``.
    """

    def __init__(
        self,
        encoder,
        head,
        database,
        window=0.1,
        device="cpu",
        batch_size=256,
        normalized=False,
    ):
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.head = head.to(self.device).eval()
        self.database = (
            np.ascontiguousarray(database, dtype=np.float32)
            if normalized
            else znorm(database)
        )
        if (
            self.database.ndim != 2
            or not len(self.database)
            or not np.isfinite(self.database).all()
        ):
            raise ValueError("Invalid database")
        self.window, self.batch_size = window, batch_size
        self.embeddings = self.encode(self.database)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def encode(self, series):
        chunks = []
        for start in range(0, len(series), self.batch_size):
            x = torch.as_tensor(
                np.array(series[start : start + self.batch_size], dtype=np.float32),
                device=self.device,
            )
            chunks.append(self.encoder(x))
        result = torch.cat(chunks)
        if not torch.isfinite(result).all():
            raise ValueError("Encoder produced non-finite embeddings")
        return result

    @torch.inference_mode()
    def ranking(self, query, k=10, rho=0.02, kappa=5):
        """Euclidean prefilter of kappa * rho * N candidates, WETW rescoring, top rho * N.

        kappa = 1 keeps the Euclidean shortlist; kappa = 0 rescores the whole database.
        """
        n = len(self.database)
        if not 1 <= k <= n:
            raise ValueError("k must be between 1 and database size")
        if not 0 < rho <= 1 or kappa < 0:
            raise ValueError("Require 0 < rho <= 1 and kappa >= 0 (0 = whole database)")
        m = min(n, max(k, math.ceil(rho * n)))
        p = n if kappa == 0 else min(n, max(m, math.ceil(kappa * rho * n)))
        self._sync()
        start = time.perf_counter()
        z = self.encode(query[None])
        self._sync()
        encoded = time.perf_counter() - start
        start = time.perf_counter()
        base = torch.argsort((self.embeddings - z).norm(dim=-1), stable=True)
        ids = base[:p]
        if p == m:  # kappa = 1: the Euclidean shortlist itself
            shortlist = ids.cpu().numpy()
            self._sync()
            return shortlist, encoded, time.perf_counter() - start, p
        scores = []
        for off in range(0, p, self.batch_size):
            emb = self.embeddings[ids[off : off + self.batch_size]]
            scores.append(self.head(z.expand(len(emb), -1), emb))
        scores = torch.cat(scores)
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite learned distances")
        shortlist = ids[torch.argsort(scores, stable=True)[:m]].cpu().numpy()
        self._sync()
        ranked = time.perf_counter() - start
        return shortlist, encoded, ranked, p

    def query(self, query, k=10, rho=0.02, kappa=5, normalized=False):
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.database.shape[1],):
            raise ValueError("Query length differs from database")
        if not normalized:
            query = znorm(query[None])[0]
        shortlist, encoded, ranked, p = self.ranking(query, k, rho, kappa)
        start = time.perf_counter()
        distances = np.asarray(
            [distance(query, self.database[j], self.window) for j in shortlist]
        )
        order = np.lexsort((shortlist, distances))[:k]
        verified = time.perf_counter() - start
        return SearchResult(
            shortlist[order],
            distances[order],
            encoded,
            ranked,
            verified,
            len(shortlist),
            p,
        )
