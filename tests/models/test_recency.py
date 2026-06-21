from book_recsys.models.recency import RecencyWeightedRecommender


class _Base:

    def __init__(self):
        self.calls = []

    def recommend(self, history, k, weights=None):
        self.calls.append((list(history), None if weights is None else list(weights)))
        return list(history)[:k]


def test_recency_wrapper_passes_order_weights_to_base():
    base = _Base()
    RecencyWeightedRecommender(base, tau=2.0).recommend(["x", "y", "z"], 2)
    hist, w = base.calls[0]
    assert hist == ["x", "y", "z"]
    assert w[-1] == 1.0 and w[0] < w[1] < w[2]  # later (more recent) picks weighted higher


def test_recency_wrapper_returns_base_output():
    assert RecencyWeightedRecommender(_Base(), tau=2.0).recommend(["x", "y"], 1) == ["x"]


def test_recency_wrapper_empty_history():
    assert RecencyWeightedRecommender(_Base(), tau=2.0).recommend([], 3) == []


def test_recency_wrapper_fit_returns_self():
    rec = RecencyWeightedRecommender(_Base(), tau=2.0)
    assert rec.fit() is rec
