import numpy as np

from book_recsys.models.content.similar import SimilarItemsRecommender

BOOK_IDS = ["b0", "b1", "b2", "b3"]
MATRIX = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])


def test_returns_nearest_neighbors_excluding_anchor():
    rec = SimilarItemsRecommender(BOOK_IDS, MATRIX).fit()
    assert rec.recommend("b0", k=2) == ["b1", "b3"]


def test_anchor_itself_never_returned():
    rec = SimilarItemsRecommender(BOOK_IDS, MATRIX).fit()
    assert "b0" not in rec.recommend("b0", k=4)


def test_unknown_anchor_returns_empty():
    rec = SimilarItemsRecommender(BOOK_IDS, MATRIX).fit()
    assert rec.recommend("nope", k=3) == []


import scipy.sparse as sp


def test_works_on_sparse_matrix():
    rec = SimilarItemsRecommender(BOOK_IDS, sp.csr_matrix(MATRIX)).fit()
    assert rec.recommend("b0", k=2) == ["b1", "b3"]
