import numpy as np

from book_recsys.models.content.content import ContentRecommender

BOOK_IDS = ["b0", "b1", "b2", "b3"]
MATRIX = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])


def test_ranks_by_cosine_to_profile_excluding_seen():
    rec = ContentRecommender(BOOK_IDS, MATRIX).fit()
    assert rec.recommend(["b0"], k=3) == ["b1", "b3", "b2"]


def test_recommend_weights_shift_profile_toward_emphasized_history():
    rec = ContentRecommender(BOOK_IDS, MATRIX).fit()
    # b0=[1,0], b2=[0,1]; emphasising b2 pulls the profile toward [0,1] -> b3 ([0.6,0.8]) first
    assert rec.recommend(["b0", "b2"], k=2, weights=[1.0, 9.0])[0] == "b3"
    assert rec.recommend(["b0", "b2"], k=2, weights=[9.0, 1.0])[0] == "b1"


def test_excludes_all_history():
    rec = ContentRecommender(BOOK_IDS, MATRIX).fit()
    out = rec.recommend(["b0", "b1"], k=3)
    assert "b0" not in out and "b1" not in out


def test_empty_history_returns_empty():
    rec = ContentRecommender(BOOK_IDS, MATRIX).fit()
    assert rec.recommend([], k=3) == []


def test_unknown_history_ids_ignored():
    rec = ContentRecommender(BOOK_IDS, MATRIX).fit()
    assert rec.recommend(["nope"], k=3) == []


def test_fit_returns_self():
    rec = ContentRecommender(BOOK_IDS, MATRIX)
    assert rec.fit() is rec


import scipy.sparse as sp


def test_works_on_sparse_matrix():
    rec = ContentRecommender(BOOK_IDS, sp.csr_matrix(MATRIX)).fit()
    assert rec.recommend(["b0"], k=3) == ["b1", "b3", "b2"]


def test_score_items_ranks_similar_higher():
    rec = ContentRecommender(BOOK_IDS, MATRIX).fit()
    s = rec.score_items(["b0"], ["b1", "b2"])   # b1 same direction as b0; b2 orthogonal
    assert s[0] > s[1]


def test_score_items_unknown_and_empty_history_are_neg_inf():
    rec = ContentRecommender(BOOK_IDS, MATRIX).fit()
    assert rec.score_items(["b0"], ["nope"]) == [float("-inf")]
    assert rec.score_items([], ["b1"]) == [float("-inf")]
