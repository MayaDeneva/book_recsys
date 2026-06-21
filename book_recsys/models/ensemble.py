"""Reciprocal Rank Fusion ensemble: combine recommenders with different coverage/score scales."""
from book_recsys.llm.fusion import weighted_reciprocal_rank_fusion


class RRFEnsembleRecommender:
    """Fuse several pre-fitted recommenders' ranked lists via weighted Reciprocal Rank Fusion.

    Rank-based, so components with incomparable score scales (e.g. SASRec's softmax vs a hybrid's
    blended score) and *different catalog coverage* combine cleanly — an item absent from one
    component simply contributes 0 there, so a strong-but-narrow model (SASRec) re-ranks the head
    while a full-coverage model (hybrid / max-sim) supplies the tail. Each component must expose
    `recommend(query, k)`; components are passed already fitted.

    Per-history `weights` (recency or event importance) are forwarded only to components that
    declare `weight_aware = True`; the rest are called unweighted, so mixing weight-aware and
    plain recommenders in one ensemble is safe.
    """
    weight_aware = True

    @staticmethod
    def _kw(rec, weights):  # forward per-history weights only to components that accept them
        return {
            "weights": weights
        } if weights is not None and getattr(rec, "weight_aware", False) else {}

    def __init__(self,
                 components: dict,
                 weights: dict | None = None,
                 k: int = 60,
                 pool: int = 200) -> None:
        self._components = dict(components)
        self._fuse_w = {name: (weights or {}).get(name, 1.0) for name in self._components}
        self._k = k
        self._pool = pool

    def fit(self, train_data=None) -> "RRFEnsembleRecommender":
        return self

    def recommend(self, query, k: int, weights=None) -> list:
        weighted = [(rec.recommend(query, self._pool, **self._kw(rec,
                                                                 weights)), self._fuse_w[name])
                    for name, rec in self._components.items()]
        return weighted_reciprocal_rank_fusion(weighted, k=self._k)[:k]

    def score_items(self, query, item_ids, weights=None) -> list:
        """Rank-fuse each component's scores over the SAME candidate set (so the ensemble can
        drive score-based rankers like FeedService). Each component ranks `item_ids` by its own
        score_items; the weighted RRF of those per-candidate ranks is the fused score."""
        fused = [0.0] * len(item_ids)
        for name, rec in self._components.items():
            scores = rec.score_items(query, item_ids, **self._kw(rec, weights))
            order = sorted(range(len(item_ids)), key=lambda i: scores[i], reverse=True)
            weight = self._fuse_w[name]
            for rank, i in enumerate(order):
                fused[i] += weight / (self._k + rank + 1)
        return fused
