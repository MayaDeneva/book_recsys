import numpy as np

from book_recsys.features.index import build_index, search


def test_query_returns_itself_as_nearest():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    index = build_index(vecs)
    scores, idx = search(index, vecs[[0]], k=1)
    assert idx[0][0] == 0
    assert scores.shape == (1, 1)


def test_ranks_by_cosine():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    index = build_index(vecs)
    _, idx = search(index, np.array([[1.0, 0.0]]), k=3)
    # nearest to [1,0]: itself (0), then [0.9,0.1] (2), then [0,1] (1)
    assert list(idx[0]) == [0, 2, 1]


def test_index_size_matches_input():
    vecs = np.random.default_rng(0).random((5, 4))
    index = build_index(vecs)
    assert index.ntotal == 5
