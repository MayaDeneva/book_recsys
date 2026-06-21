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


def test_maxsim_score_items_mixed_known_unknown():
    rec = MaxSimRecommender(BOOK_IDS, MATRIX).fit()
    s = rec.score_items(["b0"], ["b3", "zzz"])  # b3 known, zzz absent -> [score, -inf]
    assert s[0] > 0 and s[1] == float("-inf")


# ---------------------------------------------------------------------------
# zmean aggregation: a single off-distribution seed (e.g. a lone foreign-language book among
# many English ones) must NOT hijack the feed. e0/e1/e2 are the "English" majority near the
# x-axis; x0 is the lone "Bulgarian" seed on the y-axis whose candidate xc sits at ~1.0 cosine
# (a tight degenerate cluster). Under max-sim xc's 1.0 beats English candidate ec's ~0.86;
# under zmean the 3 English seeds outvote the 1 outlier.
# ---------------------------------------------------------------------------
IDS2 = ["e0", "e1", "e2", "ec", "x0", "xc", "bland"]
M2 = np.array([
    [1.0, 0.0, 0.0],     # e0  English seed
    [0.96, 0.0, 0.10],   # e1  English seed
    [0.96, 0.0, -0.10],  # e2  English seed
    [0.80, 0.0, 0.60],   # ec  English candidate (off-axis: max-sim ~0.86)
    [0.0, 1.0, 0.0],     # x0  lone "Bulgarian" seed
    [0.0, 1.0, 0.02],    # xc  "Bulgarian" candidate (~1.0 cosine to x0)
    [0.6, 0.6, 0.0],     # bland centroid
])
SEEDS2 = ["e0", "e1", "e2", "x0"]


def test_maxsim_max_lets_one_outlier_seed_hijack():
    rec = MaxSimRecommender(IDS2, M2, agg="max").fit()
    out = rec.recommend(SEEDS2, k=3)
    assert out.index("xc") < out.index("ec")  # outlier candidate wins under raw max-sim


def test_zmean_keeps_majority_above_outlier():
    rec = MaxSimRecommender(IDS2, M2, agg="zmean").fit()
    out = rec.recommend(SEEDS2, k=3)
    assert out.index("ec") < out.index("xc")  # majority-language candidate now wins


def test_zmean_weights_can_still_emphasize_a_seed():
    rec = MaxSimRecommender(IDS2, M2, agg="zmean").fit()
    out = rec.recommend(SEEDS2, k=3, weights=[1.0, 1.0, 1.0, 50.0])  # lean hard on x0
    assert out.index("xc") < out.index("ec")  # weighting overrides the majority


def test_zmean_score_items_demotes_the_outlier():
    z = MaxSimRecommender(IDS2, M2, agg="zmean").fit().score_items(SEEDS2, ["ec", "xc"])
    assert z[0] > z[1]  # English candidate scored above the outlier
    raw = MaxSimRecommender(IDS2, M2, agg="max").fit().score_items(SEEDS2, ["ec", "xc"])
    assert raw[1] > raw[0]  # ...whereas raw max-sim prefers the outlier
