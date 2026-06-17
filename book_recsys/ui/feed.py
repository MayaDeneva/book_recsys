"""Swipe feed: hybrid candidate generation, exclusion, and negative penalization.

Pure logic — takes any recommender exposing recommend(history, k) and
score_items(history, candidates) -> scores, plus the book embeddings for the
penalty term.
"""
import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class FeedService:
    """Rank the next swipe cards: hybrid score, minus a penalty for similarity to disliked."""

    def __init__(self, recommender, embeddings, book_ids, pool: int = 200) -> None:
        self._rec = recommender
        self._emb = _l2_normalize(np.asarray(embeddings, dtype="float32"))
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
        base = _minmax(np.asarray(self._rec.score_items(liked, candidates), dtype="float64"))
        order = np.argsort(-base, kind="stable")[:k]
        return [candidates[i] for i in order]
