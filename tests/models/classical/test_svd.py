import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.models.classical.svd import SvdRecommender


def _df(rows):
    return pd.DataFrame([{USER: u, BOOK: b, RATING: r, TS: 0} for u, b, r in rows])


# Two clusters: {u0,u1} read {b0,b1}; {u2,u3} read {b2,b3}.
TRAIN = _df([
    ("u0", "b0", 5), ("u0", "b1", 4),
    ("u1", "b0", 4), ("u1", "b1", 5),
    ("u2", "b2", 5), ("u2", "b3", 4),
    ("u3", "b2", 4), ("u3", "b3", 5),
])


def test_recommends_co_cluster_book_first():
    rec = SvdRecommender(n_factors=2).fit(TRAIN)
    out = rec.recommend(["b0"], k=3)
    assert out[0] == "b1"  # same cluster as b0


def test_excludes_seen_books():
    rec = SvdRecommender(n_factors=2).fit(TRAIN)
    assert "b0" not in rec.recommend(["b0"], k=3)


def test_empty_history_returns_empty():
    rec = SvdRecommender(n_factors=2).fit(TRAIN)
    assert rec.recommend([], k=3) == []


def test_recommend_weights_bias_toward_emphasized_history():
    rec = SvdRecommender(n_factors=2).fit(TRAIN)
    # history spans both clusters; emphasising b0 should rank its cluster-mate b1 above b3
    emph_b0 = rec.recommend(["b0", "b2"], k=4, weights=[9.0, 1.0])
    emph_b2 = rec.recommend(["b0", "b2"], k=4, weights=[1.0, 9.0])
    assert emph_b0.index("b1") < emph_b0.index("b3")
    assert emph_b2.index("b3") < emph_b2.index("b1")


def test_fit_returns_self():
    rec = SvdRecommender(n_factors=2)
    assert rec.fit(TRAIN) is rec


def test_score_items_co_cluster_scores_higher():
    rec = SvdRecommender(n_factors=2).fit(TRAIN)
    s = rec.score_items(["b0"], ["b1", "b2"])   # b1 co-cluster with b0; b2 other cluster
    assert s[0] > s[1]


def test_score_items_unknown_or_empty_is_neg_inf():
    rec = SvdRecommender(n_factors=2).fit(TRAIN)
    assert rec.score_items(["b0"], ["zzz"]) == [float("-inf")]
    assert rec.score_items([], ["b1"]) == [float("-inf")]
