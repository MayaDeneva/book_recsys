"""Max-similarity recommender: rank by closeness to the *nearest* liked book, not the mean.

Mean-pooling a diverse history into one profile washes out individual tastes and drifts toward
the popular centroid (the "same recs every time" failure). Max-sim instead scores each candidate
by its highest cosine to *any* history item, so every liked book pulls in its own neighbours.
Optional `weights` (aligned to history) scale each item's similarity for recency weighting.

`agg` chooses how the per-history similarities collapse to one score:
  - "max" (default, the study baseline): the candidate's single highest cosine to any liked book.
    Every seed pulls its own neighbours — great for diverse tastes, but a single off-distribution
    seed (e.g. a lone foreign-language book, which an English encoder maps to a degenerate cluster
    at ~0.95+ cosine) defines a whole high-scoring neighbourhood that out-ranks every genuine-topic
    match. `max` counts the best single match, not how many seeds agree, so one outlier seed can
    hijack the feed regardless of how many "normal" seeds you have.
  - "zmean" (robustness variant): z-score each seed's similarity column — putting every seed on a
    comparable scale, so "top neighbour of the outlier seed" no longer dwarfs "top neighbour of a
    normal seed" — then average across seeds. No single seed can contribute more than 1/|history|
    of the signal, so the majority wins. Trades a little of max's multi-interest sharpness for
    robustness to outlier seeds (mild centroid drift). Used by the live feed; the offline benchmark
    keeps "max" as the baseline.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from book_recsys.models.aggregate import aligned_weights


class MaxSimRecommender:
    """Rank books by similarity to history items (optionally recency-weighted). `agg` is "max"
    (nearest liked book) or "zmean" (outlier-robust per-seed-standardized average)."""
    weight_aware = True  # accepts per-history `weights` (recency or event importance)

    def __init__(self, book_ids, matrix, agg: str = "max") -> None:
        self._ids = list(book_ids)
        self._pos = {b: i for i, b in enumerate(self._ids)}
        self._matrix = normalize(matrix, axis=1)
        self._agg = agg

    def fit(self, train_data=None) -> "MaxSimRecommender":
        return self

    def _aggregate(self, sims: np.ndarray, w) -> np.ndarray:
        """Collapse a (n_items, n_history) cosine matrix to one score per item."""
        if self._agg == "max":
            if w is not None:
                sims = sims * np.asarray(w, dtype=float)  # scale each history column (recency)
            return sims.max(axis=1)
        # "zmean": per-seed standardization removes the scale gap between a degenerate outlier
        # cluster and genuine-topic neighbours, then the average lets the majority outvote it.
        z = (sims - sims.mean(axis=0, keepdims=True)) / (sims.std(axis=0, keepdims=True) + 1e-9)
        if w is not None:
            wv = np.asarray(w, dtype=float)
            return (z * wv).sum(axis=1) / wv.sum()
        return z.mean(axis=1)

    def recommend(self, query, k: int, weights=None) -> list:
        idx, w = aligned_weights(query, weights, self._pos)
        if not idx:
            return []
        sims = self._matrix @ self._matrix[idx].T
        sims = np.asarray(sims.todense()) if sp.issparse(sims) else np.asarray(sims)
        scores = self._aggregate(sims, w)
        seen = set(query)
        out = []
        for i in np.argsort(-scores):
            book = self._ids[i]
            if book not in seen:
                out.append(book)
                if len(out) == k:
                    break
        return out

    def score_items(self, query, item_ids, weights=None) -> list:
        """Score each candidate against the history (for sampled-neg eval / feed re-ranking).
        `weights` (aligned to history) scale each item's similarity, matching recommend(). With
        agg="zmean" the per-seed standardization is over the supplied `item_ids` batch."""
        idx, w = aligned_weights(query, weights, self._pos)
        rows = [self._pos.get(b, -1) for b in item_ids]
        if not idx or not any(r >= 0 for r in rows):
            return [float("-inf")] * len(item_ids)
        sims = self._matrix[[r for r in rows if r >= 0]] @ self._matrix[idx].T
        sims = np.asarray(sims.todense()) if sp.issparse(sims) else np.asarray(sims)
        agg = self._aggregate(sims, w)
        out, j = [], 0
        for r in rows:
            if r >= 0:
                out.append(float(agg[j]))
                j += 1
            else:
                out.append(float("-inf"))
        return out
