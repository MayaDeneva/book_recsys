import math

from book_recsys.eval.metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_counts_hits_over_relevant():
    assert recall_at_k(["b0", "b9", "b8"], {"b0", "b5"}, k=3) == 0.5


def test_recall_at_k_zero_when_no_relevant():
    assert recall_at_k(["b0"], set(), k=3) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k(["b0", "b1"], {"b0", "b1"}, k=2) == 1.0


def test_ndcg_at_k_discounts_by_position():
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(["b9", "b0", "b8"], {"b0"}, k=3) == expected


def test_ndcg_at_k_zero_when_no_relevant():
    assert ndcg_at_k(["b0"], set(), k=3) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr(["b9", "b0", "b8"], {"b0"}) == 0.5


def test_mrr_zero_when_no_hit():
    assert mrr(["b9", "b8"], {"b0"}) == 0.0
