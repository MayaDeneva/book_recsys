"""Max-similarity recommender: rank by closeness to the *nearest* liked book, not the mean.

Mean-pooling a diverse history into one profile washes out individual tastes and drifts toward
the popular centroid (the "same recs every time" failure). Max-sim instead scores each candidate
by its highest cosine to *any* history item, so every liked book pulls in its own neighbours.
Optional `weights` (aligned to history) scale each item's similarity for recency weighting.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from book_recsys.models.aggregate import aligned_weights


class MaxSimRecommender:
    """Rank books by max cosine similarity to any history item (optionally recency-weighted)."""

    def __init__(self, book_ids, matrix) -> None:
        self._ids = list(book_ids)
        self._pos = {b: i for i, b in enumerate(self._ids)}
        self._matrix = normalize(matrix, axis=1)

    def fit(self, train_data=None) -> "MaxSimRecommender":
        return self

    def recommend(self, query, k: int, weights=None) -> list:
        idx, w = aligned_weights(query, weights, self._pos)
        if not idx:
            return []
        sims = self._matrix @ self._matrix[idx].T
        sims = np.asarray(sims.todense()) if sp.issparse(sims) else np.asarray(sims)
        if w is not None:
            sims = sims * np.asarray(w, dtype=float)  # scale each history column (recency)
        scores = sims.max(axis=1)
        seen = set(query)
        out = []
        for i in np.argsort(-scores):
            book = self._ids[i]
            if book not in seen:
                out.append(book)
                if len(out) == k:
                    break
        return out

    def score_items(self, query, item_ids) -> list:
        """Score each candidate by its max cosine to any history item (for sampled-neg eval)."""
        idx = [self._pos[b] for b in query if b in self._pos]
        if not idx:
            return [float("-inf")] * len(item_ids)
        hist = self._matrix[idx]
        out = []
        for b in item_ids:
            if b not in self._pos:
                out.append(float("-inf"))
                continue
            sims = self._matrix[self._pos[b]] @ hist.T
            sims = np.asarray(sims.todense()) if sp.issparse(sims) else np.asarray(sims)
            out.append(float(sims.max()))
        return out
