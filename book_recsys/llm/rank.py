"""Pure ranking for LLM-steered recsys: fuse weighted signals, penalize 'avoid',
optionally filter by genre. No LLM here — the LLM only supplies the SteeringState.
"""
import numpy as np

from book_recsys.llm.fusion import weighted_reciprocal_rank_fusion
from book_recsys.vecmath import l2_normalize, minmax


class SteeredRanker:
    """Rank candidates from a SteeringState using the existing recsys primitives."""

    def __init__(self,
                 cf_model,
                 retriever,
                 similar,
                 embeddings,
                 book_ids,
                 encoder,
                 catalog_genre=None,
                 pool: int = 200,
                 lam: float = 1.0) -> None:
        self._cf = cf_model
        self._retriever = retriever
        self._similar = similar
        self._emb = l2_normalize(np.asarray(embeddings, dtype="float32"))
        self._row = {b: i for i, b in enumerate(book_ids)}
        self._encoder = encoder
        self._genre = catalog_genre or {}
        self._pool = pool
        self._lam = lam

    def rank(self, state, history_ids, seen, k: int = 10, anchor_id=None) -> list:
        history_ids = list(history_ids)
        w = state.history_weight
        weighted_lists = []
        # A signal at weight 0 is "off" — skip it entirely (the fusion util keeps
        # zero-weight items, so leaving them in would pollute candidates with score-0
        # neighbours). The anchor is an explicit, user-named book, so it always carries
        # a fixed weight, independent of the history<->topic blend (so a gift query with
        # history_weight~0 still honours a named recipient-favourite).
        if history_ids and w > 0:
            weighted_lists.append((self._cf.recommend(history_ids, self._pool), w / 2))
            weighted_lists.append((self._retriever.by_history(history_ids, self._pool), w / 2))
        if anchor_id is not None:
            weighted_lists.append((self._similar.recommend(anchor_id, self._pool), 0.5))
        if state.topic and (1 - w) > 0:
            weighted_lists.append((self._retriever.by_text(state.topic, self._pool), 1 - w))
        if not weighted_lists:
            return []

        fused = weighted_reciprocal_rank_fusion(weighted_lists)
        exclude = set(history_ids) | set(seen)
        candidates = [b for b in fused if b not in exclude]
        if state.genre:
            g = state.genre.lower()
            candidates = [b for b in candidates if g in str(self._genre.get(b, "")).lower()]
        if not candidates:
            return []

        # base score = inverse fused rank (earlier = higher), min-max normalized.
        base = minmax(np.array([-i for i in range(len(candidates))], dtype="float64"))
        if state.avoid:
            base = base - self._lam * self._avoid_penalty(candidates, state.avoid)
        order = np.argsort(-base, kind="stable")[:k]
        return [candidates[i] for i in order]

    def _avoid_penalty(self, candidates, avoid) -> np.ndarray:
        avoid_vecs = l2_normalize(np.asarray(self._encoder.encode(list(avoid)), dtype="float32"))
        penalty = np.zeros(len(candidates), dtype="float64")
        for i, book_id in enumerate(candidates):
            if book_id in self._row:  # unknown ids contribute no penalty (stay 0)
                penalty[i] = float((self._emb[self._row[book_id]] @ avoid_vecs.T).max())
        return penalty
