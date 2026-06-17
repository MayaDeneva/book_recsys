import pandas as pd

from book_recsys.data.kcore import k_core_filter
from book_recsys.data.schema import BOOK, RATING, TS, USER


def _df(pairs):
    rows = [{USER: u, BOOK: b, RATING: 5, TS: 0} for u, b in pairs]
    return pd.DataFrame(rows)


def test_drops_users_below_threshold():
    df = _df([("u0", "b0"), ("u0", "b1"), ("u1", "b0")])
    out = k_core_filter(df, min_user=2, min_book=1)
    assert set(out[USER]) == {"u0"}


def test_iterates_until_stable():
    df = _df([("u0", "b0"), ("u0", "b1"), ("u1", "b1")])
    out = k_core_filter(df, min_user=2, min_book=2)
    assert len(out) == 0


def test_keeps_stable_core():
    df = _df([("u0", "b0"), ("u0", "b1"), ("u1", "b0"), ("u1", "b1")])
    out = k_core_filter(df, min_user=2, min_book=2)
    assert len(out) == 4
    assert list(out.index) == [0, 1, 2, 3]  # index reset
