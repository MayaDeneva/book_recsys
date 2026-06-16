import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.data.splits import global_time_split, leave_last_n_out


def _df(triples):
    rows = [{USER: u, BOOK: b, RATING: 5, TS: t} for u, b, t in triples]
    return pd.DataFrame(rows)


def test_global_time_split_partitions_by_timestamp():
    df = _df([("u", f"b{t}", t) for t in range(10)])  # ts 0..9
    train, val, test = global_time_split(df, train_frac=0.6, val_frac=0.2)
    assert train[TS].max() <= val[TS].min()
    assert val[TS].max() <= test[TS].min()
    assert len(train) + len(val) + len(test) == 10


def test_global_time_split_train_is_majority():
    df = _df([("u", f"b{t}", t) for t in range(10)])
    train, val, test = global_time_split(df, train_frac=0.6, val_frac=0.2)
    assert len(train) >= len(val)
    assert len(train) >= len(test)


def test_leave_last_n_out_holds_out_latest_per_user():
    df = _df([
        ("u0", "b0", 1), ("u0", "b1", 2), ("u0", "b2", 3),
        ("u1", "b0", 5), ("u1", "b3", 6),
    ])
    train, holdout = leave_last_n_out(df, n=1)
    assert set(zip(holdout[USER], holdout[BOOK])) == {("u0", "b2"), ("u1", "b3")}
    assert len(train) == 3
