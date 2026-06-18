"""Swipe feed: hybrid candidate generation, exclusion, and negative penalization.

Pure logic — takes any recommender exposing recommend(history, k) and
score_items(history, candidates) -> scores, plus the book embeddings for the
penalty term.
"""
import numpy as np

from book_recsys.vecmath import l2_normalize, minmax


class FeedService:
    """Rank the next swipe cards: hybrid score, minus a penalty for similarity to disliked."""

    def __init__(self, recommender, embeddings, book_ids, pool: int = 200) -> None:
        self._rec = recommender
        self._emb = l2_normalize(np.asarray(embeddings, dtype="float32"))
        self._row = {b: i for i, b in enumerate(book_ids)}
        self._pool = pool

    def next(self, liked, disliked, seen, k: int = 10, lam: float = 1.0) -> list:
        liked = list(liked)
        if not liked:
            return []
        candidates = self._rec.recommend(liked, self._pool)
        exclude = set(seen) | set(liked) | set(disliked)
        candidates = [c for c in candidates if c not in exclude]
        if not candidates:
            return []
        base = minmax(np.asarray(self._rec.score_items(liked, candidates), dtype="float64"))
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
