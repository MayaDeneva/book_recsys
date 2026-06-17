import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.models.classical.popularity import PopularityRecommender


def _df():
    rows = [
        {USER: "u0", BOOK: "b0", RATING: 5, TS: 0},
        {USER: "u1", BOOK: "b0", RATING: 5, TS: 0},
        {USER: "u2", BOOK: "b0", RATING: 5, TS: 0},
        {USER: "u0", BOOK: "b1", RATING: 5, TS: 0},
        {USER: "u1", BOOK: "b1", RATING: 5, TS: 0},
        {USER: "u0", BOOK: "b2", RATING: 5, TS: 0},
    ]
    return pd.DataFrame(rows)


def test_recommends_most_popular_first():
    rec = PopularityRecommender().fit(_df())
    assert rec.recommend(query=[], k=3) == ["b0", "b1", "b2"]


def test_excludes_already_seen_books():
    rec = PopularityRecommender().fit(_df())
    assert rec.recommend(query=["b0"], k=2) == ["b1", "b2"]


def test_respects_k():
    rec = PopularityRecommender().fit(_df())
    assert rec.recommend(query=[], k=1) == ["b0"]


def test_fit_returns_self():
    rec = PopularityRecommender()
    assert rec.fit(_df()) is rec


def test_score_items_by_popularity():
    rec = PopularityRecommender().fit(_df())   # b0 most popular, b2 least
    s = rec.score_items([], ["b0", "b2"])
    assert s[0] > s[1]


def test_score_items_unknown_is_neg_inf():
    rec = PopularityRecommender().fit(_df())
    assert rec.score_items([], ["zzz"]) == [float("-inf")]
