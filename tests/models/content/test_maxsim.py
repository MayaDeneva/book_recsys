import numpy as np
import scipy.sparse as sp

from book_recsys.models.content.maxsim import MaxSimRecommender

BOOK_IDS = ["b0", "b1", "b2", "b3", "b4"]
# b3 is b0's neighbour, b4 is b1's neighbour, b2 is the bland centroid match.
MATRIX = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.9, 0.1], [0.1, 0.9]])


def test_maxsim_surfaces_neighbours_of_each_history_item():
    rec = MaxSimRecommender(BOOK_IDS, MATRIX).fit()
    out = rec.recommend(["b0", "b1"], k=3)
    # each liked book pulls in its own neighbour; b2 (the mean-pool favourite) does not win
    assert set(out[:2]) == {"b3", "b4"}


def test_maxsim_weights_bias_toward_emphasized_history():
    rec = MaxSimRecommender(BOOK_IDS, MATRIX).fit()
    out = rec.recommend(["b0", "b1"], k=3, weights=[1.0, 5.0])  # emphasise b1
    assert out.index("b4") < out.index("b3")


def test_maxsim_excludes_seen():
    out = MaxSimRecommender(BOOK_IDS, MATRIX).fit().recommend(["b0"], k=4)
    assert "b0" not in out


def test_maxsim_empty_history_returns_empty():
    assert MaxSimRecommender(BOOK_IDS, MATRIX).fit().recommend([], k=3) == []


def test_maxsim_fit_returns_self():
    rec = MaxSimRecommender(BOOK_IDS, MATRIX)
    assert rec.fit() is rec


def test_maxsim_works_on_sparse_matrix():
    rec = MaxSimRecommender(BOOK_IDS, sp.csr_matrix(MATRIX)).fit()
    assert set(rec.recommend(["b0", "b1"], k=2)) == {"b3", "b4"}


def test_maxsim_score_items_higher_for_closer_book():
    rec = MaxSimRecommender(BOOK_IDS, MATRIX).fit()
    s = rec.score_items(["b0"], ["b3", "b1"])  # b3 near b0; b1 orthogonal to b0
    assert s[0] > s[1]


def test_maxsim_score_items_takes_max_over_history():
    rec = MaxSimRecommender(BOOK_IDS, MATRIX).fit()
    s = rec.score_items(["b0", "b1"], ["b4"])  # b4 near b1 -> scored via b1, not b0
    assert s[0] > 0.9


def test_maxsim_score_items_unknown_or_empty_is_neg_inf():
    rec = MaxSimRecommender(BOOK_IDS, MATRIX).fit()
    assert rec.score_items(["b0"], ["zzz"]) == [float("-inf")]
    assert rec.score_items([], ["b1"]) == [float("-inf")]


def test_maxsim_score_items_works_on_sparse_matrix():
    rec = MaxSimRecommender(BOOK_IDS, sp.csr_matrix(MATRIX)).fit()
    assert rec.score_items(["b0"], ["b3", "b1"])[0] > rec.score_items(["b0"], ["b3", "b1"])[1]


def test_score_items_weights_scale_similarity():
    emb = np.array([[1, 0], [0, 1], [1, 0]], dtype="float32")  # c shares a's direction
    rec = MaxSimRecommender(["a", "b", "c"], emb)
    full = rec.score_items(["a", "b"], ["c"], weights=[1.0, 1.0])
    half = rec.score_items(["a", "b"], ["c"], weights=[0.5, 1.0])  # down-weight a (c's match)
    assert full[0] > half[0]  # scaling a's column down lowers c's max-sim
    assert MaxSimRecommender.weight_aware is True
