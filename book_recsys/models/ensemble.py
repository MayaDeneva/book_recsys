"""Reciprocal Rank Fusion ensemble: combine recommenders with different coverage/score scales."""
from book_recsys.llm.fusion import weighted_reciprocal_rank_fusion


class RRFEnsembleRecommender:
    """Fuse several pre-fitted recommenders' ranked lists via weighted Reciprocal Rank Fusion.

    Rank-based, so components with incomparable score scales (e.g. SASRec's softmax vs a hybrid's
    blended score) and *different catalog coverage* combine cleanly — an item absent from one
    component simply contributes 0 there, so a strong-but-narrow model (SASRec) re-ranks the head
    while a full-coverage model (hybrid / max-sim) supplies the tail. Each component must expose
    `recommend(query, k)`; components are passed already fitted.
    """

    def __init__(self,
                 components: dict,
                 weights: dict | None = None,
                 k: int = 60,
                 pool: int = 200) -> None:
        self._components = dict(components)
        self._weights = {name: (weights or {}).get(name, 1.0) for name in self._components}
        self._k = k
        self._pool = pool

    def fit(self, train_data=None) -> "RRFEnsembleRecommender":
        return self

    def recommend(self, query, k: int) -> list:
        weighted = [(self._components[name].recommend(query, self._pool), self._weights[name])
                    for name in self._components]
        return weighted_reciprocal_rank_fusion(weighted, k=self._k)[:k]
