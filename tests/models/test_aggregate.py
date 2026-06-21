import math

import numpy as np
import scipy.sparse as sp

from book_recsys.models.aggregate import aligned_weights, recency_weights, weighted_profile


def test_aligned_weights_none_drops_unknowns():
    idx, w = aligned_weights(["a", "x", "b"], None, {"a": 0, "b": 1})
    assert idx == [0, 1] and w is None


def test_aligned_weights_keeps_weights_parallel_to_kept_ids():
    idx, w = aligned_weights(["a", "x", "b"], [5.0, 9.0, 7.0], {"a": 0, "b": 1})
    assert idx == [0, 1] and w == [5.0, 7.0]


def test_recency_weights_most_recent_is_one_others_decay():
    w = recency_weights([0, 10], tau=10)
    assert w[1] == 1.0  # most recent timestamp -> no decay
    assert math.isclose(w[0], math.exp(-1.0))  # 10 units older, tau=10 -> exp(-1)


def test_recency_weights_single_timestamp():
    assert list(recency_weights([5], tau=10)) == [1.0]


def test_recency_weights_larger_tau_decays_slower():
    assert recency_weights([0, 10], tau=100)[0] > recency_weights([0, 10], tau=1)[0]


def test_recency_weights_empty():
    assert len(recency_weights([], tau=10)) == 0


_M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_weighted_profile_plain_mean_when_no_weights():
    assert list(weighted_profile(_M, [0, 1])) == [0.5, 0.5]


def test_weighted_profile_weights_emphasize_rows():
    # (3*[1,0] + 1*[0,1]) / 4 = [0.75, 0.25]
    assert list(weighted_profile(_M, [0, 1], [3.0, 1.0])) == [0.75, 0.25]


def test_weighted_profile_works_on_sparse_matrix():
    out = weighted_profile(sp.csr_matrix(_M), [0, 1], [3.0, 1.0])
    assert list(out) == [0.75, 0.25]
