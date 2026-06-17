import pandas as pd

from book_recsys.data.filters import filter_min_rating
from book_recsys.data.schema import BOOK, RATING, TS, USER


def _df(ratings):
    return pd.DataFrame([{USER: "u", BOOK: f"b{i}", RATING: r, TS: 0}
                         for i, r in enumerate(ratings)])


def test_keeps_only_ratings_at_or_above_threshold():
    out = filter_min_rating(_df([0, 3, 4, 5]), min_rating=4)
    assert sorted(out[RATING]) == [4, 5]


def test_drops_implicit_zero_ratings():
    out = filter_min_rating(_df([0, 0, 5]), min_rating=1)
    assert list(out[RATING]) == [5]


def test_index_is_reset():
    out = filter_min_rating(_df([1, 5]), min_rating=5)
    assert list(out.index) == [0]
