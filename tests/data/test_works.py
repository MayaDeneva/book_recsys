import numpy as np
import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.data.works import collapse_editions


def _setup():
    catalog = pd.DataFrame({BOOK: ["b1", "b2", "b3", "b4"], "title": ["T", "T", "U", "V"]})
    emb = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    work_of = {"b1": "W1", "b2": "W1", "b3": "W2"}  # b4 absent -> its own work
    inter = pd.DataFrame({                                # b2 (2) > b1 (1) -> canonical of W1
        USER: ["u1", "u2", "u3", "u1", "u2"],
        BOOK: ["b1", "b2", "b2", "b4", "b3"],
        RATING: [5, 5, 4, 3, 5],
        TS: [1, 2, 3, 4, 5],
    })
    return inter, catalog, emb, work_of


def test_picks_highest_interaction_canonical_and_aligns_embeddings():
    inter, catalog, emb, work_of = _setup()
    i2, c2, e2 = collapse_editions(inter, catalog, emb, work_of)
    assert set(c2[BOOK]) == {"b2", "b3", "b4"}  # b1 collapsed into b2
    assert "work_id" in c2.columns
    rows = {b: list(r) for b, r in zip(c2[BOOK], e2)}  # embeddings realigned to catalog
    assert rows == {"b2": [2.0, 2.0], "b3": [3.0, 3.0], "b4": [4.0, 4.0]}


def test_remaps_interactions_and_dedupes_by_work():
    inter, catalog, emb, work_of = _setup()
    i2, _, _ = collapse_editions(inter, catalog, emb, work_of)
    assert set(i2[BOOK]) <= {"b2", "b3", "b4"}  # b1 -> b2 in interactions too
    assert not i2.duplicated([USER, BOOK]).any()  # one row per (user, work)
