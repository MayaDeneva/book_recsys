from book_recsys.models.ensemble import RRFEnsembleRecommender


class _Fixed:
    def __init__(self, recs):
        self._recs = recs

    def recommend(self, query, k):
        return self._recs[:k]


def test_rrf_ranks_items_in_both_components_first():
    a = _Fixed(["x", "y", "z"])
    b = _Fixed(["y", "w", "x"])
    out = RRFEnsembleRecommender({"a": a, "b": b}).fit().recommend(["h"], 4)
    assert set(out[:2]) == {"x", "y"}   # in both lists -> outrank the singletons
    assert set(out[2:]) == {"w", "z"}


def test_rrf_covers_items_present_in_only_one_component():
    out = RRFEnsembleRecommender({"a": _Fixed(["x"]), "b": _Fixed(["q"])}).fit().recommend(["h"], 5)
    assert set(out) == {"x", "q"}        # disjoint coverage: both still surfaced


def test_rrf_weights_favour_the_heavier_component():
    a = _Fixed(["x", "y"])
    b = _Fixed(["y", "x"])
    out = RRFEnsembleRecommender({"a": a, "b": b}, weights={"a": 1.0, "b": 10.0}).fit()
    assert out.recommend(["h"], 2)[0] == "y"   # b (heavy) puts y first


def test_rrf_respects_k():
    out = RRFEnsembleRecommender({"a": _Fixed(["x", "y", "z"])}).fit().recommend(["h"], 2)
    assert len(out) == 2


def test_rrf_empty_when_all_components_empty():
    assert RRFEnsembleRecommender({"a": _Fixed([])}).fit().recommend(["h"], 5) == []


def test_rrf_fit_returns_self():
    ens = RRFEnsembleRecommender({"a": _Fixed(["x"])})
    assert ens.fit() is ens
