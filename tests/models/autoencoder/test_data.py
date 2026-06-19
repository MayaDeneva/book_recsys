import numpy as np
import pandas as pd
import pytest

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.models.autoencoder.data import build_matrix, reproduce_sasrec_sample


def _df(rows):
    return pd.DataFrame([{USER: u, BOOK: b, RATING: 0, TS: t} for u, b, t in rows])


def test_reproduce_caps_users_and_history():
    rows = []
    for u in range(5):
        for t in range(10):
            rows.append((f"u{u}", f"b{t}", t))
    out = reproduce_sasrec_sample(_df(rows), n_users=3, max_hist=4, seed=42)
    assert out[USER].nunique() == 3
    assert (out.groupby(USER).size() == 4).all()  # capped to most-recent 4


def test_reproduce_keeps_most_recent_history():
    rows = [("u0", f"b{t}", t) for t in range(6)]
    out = reproduce_sasrec_sample(_df(rows), n_users=1, max_hist=2, seed=42)
    assert set(out[BOOK]) == {"b4", "b5"}  # newest two by timestamp


def test_reproduce_expect_rows_mismatch_raises():
    rows = [("u0", "b0", 0), ("u0", "b1", 1)]
    with pytest.raises(ValueError, match="expected 999"):
        reproduce_sasrec_sample(_df(rows), n_users=1, max_hist=10, expect_rows=999)


def test_reproduce_expect_rows_match_ok():
    rows = [("u0", "b0", 0), ("u0", "b1", 1)]
    out = reproduce_sasrec_sample(_df(rows), n_users=1, max_hist=10, expect_rows=2)
    assert len(out) == 2


def test_build_matrix_shape_binary_and_vocab():
    train = _df([("u0", "b0", 0), ("u0", "b1", 1), ("u1", "b0", 0)])
    matrix, ids, pos, counts = build_matrix(train, min_item_count=1)
    assert matrix.shape == (2, 2)
    assert set(matrix.data) == {1.0}  # binary
    assert set(ids) == {"b0", "b1"} and pos[ids[0]] == 0
    assert counts[pos["b0"]] == 2 and counts[pos["b1"]] == 1


def test_build_matrix_min_item_count_filters():
    train = _df([("u0", "b0", 0), ("u1", "b0", 0), ("u0", "b1", 1)])
    matrix, ids, pos, counts = build_matrix(train, min_item_count=2)
    assert ids == ["b0"]  # b1 (count 1) dropped
    assert matrix.shape[1] == 1


def test_build_matrix_dedupes_repeat_interactions():
    train = _df([("u0", "b0", 0), ("u0", "b0", 1)])  # same user-item twice
    matrix, ids, pos, counts = build_matrix(train, min_item_count=1)
    assert matrix[0, pos["b0"]] == 1.0  # not 2.0
