import pandas as pd

from book_recsys.data.sample import sample_users
from book_recsys.data.schema import BOOK, RATING, TS, USER


def _df(n_users):
    rows = []
    for i in range(n_users):
        rows.append({USER: f"u{i}", BOOK: "b0", RATING: 5, TS: 0})
        rows.append({USER: f"u{i}", BOOK: "b1", RATING: 4, TS: 1})
    return pd.DataFrame(rows)


def test_samples_requested_number_of_users():
    out = sample_users(_df(100), n_users=10, seed=42)
    assert out[USER].nunique() == 10


def test_keeps_all_rows_for_sampled_users():
    out = sample_users(_df(100), n_users=10, seed=42)
    counts = out.groupby(USER).size()
    assert (counts == 2).all()


def test_is_deterministic_with_seed():
    a = sample_users(_df(100), n_users=10, seed=42)
    b = sample_users(_df(100), n_users=10, seed=42)
    assert set(a[USER]) == set(b[USER])


def test_returns_all_when_fewer_users_than_requested():
    out = sample_users(_df(5), n_users=10, seed=42)
    assert out[USER].nunique() == 5
