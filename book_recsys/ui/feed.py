"""Swipe feed: candidate generation, exclusion, and negative penalization.

Holds one or more recommenders (each exposing recommend(history, k) and
score_items(history, candidates) -> scores) so the UI can toggle which one drives the feed.
The book embeddings (one shared normalized copy) power the disliked-similarity penalty.
"""
import numpy as np

from book_recsys.vecmath import l2_normalize, minmax


class FeedService:
    """Rank the next swipe cards from a chosen recommender, minus a penalty for similarity to
    disliked books. `recommenders` is a {name: recommender} dict (or a single recommender)."""

    def __init__(self,
                 recommenders,
                 embeddings,
                 book_ids,
                 pool: int = 200,
                 default: str = "") -> None:
        if not isinstance(recommenders, dict):
            recommenders = {"default": recommenders}
        self._recs = dict(recommenders)
        self._default = default or next(iter(self._recs))
        self._emb = l2_normalize(np.asarray(embeddings, dtype="float32"))
        self._row = {b: i for i, b in enumerate(book_ids)}
        self._pool = pool

    def methods(self) -> list:
        """Recommender names available for the UI toggle (first is the default)."""
        return list(self._recs)

    def next(self, liked, disliked, seen, k: int = 10, lam: float = 1.0, method=None) -> list:
        liked = list(liked)
        if not liked:
            return []
        rec = self._recs.get(method) or self._recs[self._default]
        candidates = rec.recommend(liked, self._pool)
        exclude = set(seen) | set(liked) | set(disliked)
        candidates = [c for c in candidates if c not in exclude]
        if not candidates:
            return []
        base = minmax(np.asarray(rec.score_items(liked, candidates), dtype="float64"))
        if disliked and lam:
            base = base - lam * self._max_sim_to_disliked(candidates, disliked)
        order = np.argsort(-base, kind="stable")[:k]
        return [candidates[i] for i in order]

    def _max_sim_to_disliked(self, candidates, disliked) -> np.ndarray:
        """For each candidate, cosine similarity to its NEAREST disliked book (0 if none)."""
        d_rows = [self._row[d] for d in disliked if d in self._row]
        if not d_rows:
            return np.zeros(len(candidates))
        c_rows = [self._row[c] for c in candidates]
        sims = self._emb[c_rows] @ self._emb[d_rows].T  # normalized -> cosine
        return sims.max(axis=1)
