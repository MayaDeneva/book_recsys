import gzip
import json

import pandas as pd

from book_recsys.data.ingest import stream_interactions_json
from book_recsys.data.kcore import k_core_filter
from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.data.streaming import load_interactions, streaming_kcore_sample


def _rec(u, b, rating=5):
    return {
        "user_id": u,
        "book_id": b,
        "is_read": True,
        "rating": rating,
        "date_added": "Tue Nov 17 11:37:35 -0800 2017"
    }


def _write_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _frame(pairs):
    return pd.DataFrame([{USER: u, BOOK: b, RATING: 5, TS: 0} for u, b in pairs])


def test_load_matches_slow_parser(tmp_path):
    rows = [_rec("u0", "b0", 5), _rec("u1", "b1", 3), _rec("u0", "b1", 4)]
    p = tmp_path / "g.json.gz"
    _write_gz(p, rows)
    fast = load_interactions([p]).astype({USER: "object", BOOK: "object"})
    slow = pd.concat(list(stream_interactions_json(p)), ignore_index=True)
    fast_set = set(map(tuple, fast[[USER, BOOK, RATING, TS]].to_numpy().tolist()))
    slow_set = set(map(tuple, slow[[USER, BOOK, RATING, TS]].to_numpy().tolist()))
    assert fast_set == slow_set


def test_load_columns_and_timestamp(tmp_path):
    p = tmp_path / "g.json.gz"
    _write_gz(p, [_rec("u0", "b0")])
    df = load_interactions([p])
    assert list(df.columns) == [USER, BOOK, RATING, TS]
    assert int(df.iloc[0][TS]) > 0


def test_load_missing_date_is_zero(tmp_path):
    p = tmp_path / "g.json.gz"
    _write_gz(p, [{"user_id": "u0", "book_id": "b0", "rating": 5}])  # no date_added
    df = load_interactions([p])
    assert int(df.iloc[0][TS]) == 0


def test_sample_matches_kcore_survivors(tmp_path):
    rows = [
        _rec("u0", "b0"),
        _rec("u0", "b1"),
        _rec("u1", "b0"),
        _rec("u1", "b1"),
        _rec("u2", "x9")
    ]  # x9 dangling, u2 too few
    p = tmp_path / "g.json.gz"
    _write_gz(p, rows)
    out = tmp_path / "s.parquet"
    summary = streaming_kcore_sample([p],
                                     min_user=2,
                                     min_book=2,
                                     n_users=10,
                                     seed=1,
                                     out_path=out,
                                     verbose=False)
    written = pd.read_parquet(out)
    expected = k_core_filter(_frame([(r["user_id"], r["book_id"]) for r in rows]),
                             min_user=2,
                             min_book=2)
    assert set(written[USER]) <= set(expected[USER])
    assert set(written[BOOK]) <= set(expected[BOOK])
    assert "x9" not in set(written[BOOK])
    assert summary["n_users"] == written[USER].nunique()
    assert list(written.columns) == [USER, BOOK, RATING, TS]


def test_sample_deterministic(tmp_path):
    rows = [_rec(f"u{i}", "b0") for i in range(10)] + [_rec(f"u{i}", "b1") for i in range(10)]
    p = tmp_path / "g.json.gz"
    _write_gz(p, rows)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    streaming_kcore_sample([p],
                           min_user=2,
                           min_book=2,
                           n_users=4,
                           seed=7,
                           out_path=a,
                           verbose=False)
    streaming_kcore_sample([p],
                           min_user=2,
                           min_book=2,
                           n_users=4,
                           seed=7,
                           out_path=b,
                           verbose=False)
    assert set(pd.read_parquet(a)[USER]) == set(pd.read_parquet(b)[USER])


def test_sample_empty_when_nothing_survives(tmp_path):
    p = tmp_path / "g.json.gz"
    _write_gz(p, [_rec("u0", "b0")])
    out = tmp_path / "s.parquet"
    summary = streaming_kcore_sample([p],
                                     min_user=2,
                                     min_book=2,
                                     n_users=5,
                                     seed=1,
                                     out_path=out,
                                     verbose=False)
    assert summary["n_users"] == 0
    assert pd.read_parquet(out).empty
