"""Recency-weighted wrapper: make any weight-aware recommender favour recent history.

Wraps a base recommender (svd / content / max-sim) so that, instead of pooling the history
uniformly, it exponentially down-weights older interactions. History is assumed chronological
(the eval harness and the UI both supply it in order), so the last item weighs most. This is
the UC2 "multi-scale recency" lever as a drop-in `Recommender` — it scores through the same
harness as every other method, so flat-vs-recency is a clean ablation.
"""
from book_recsys.models.aggregate import recency_weights


class RecencyWeightedRecommender:
    """Decorate a weight-aware recommender with exponential recency weighting (scale `tau`)."""

    def __init__(self, base, tau: float = 5.0) -> None:
        self._base = base
        self._tau = tau

    def fit(self, train_data=None) -> "RecencyWeightedRecommender":
        return self

    def recommend(self, query, k: int, weights=None) -> list:
        w = recency_weights(list(range(len(query))), self._tau)
        return self._base.recommend(query, k, weights=w)
