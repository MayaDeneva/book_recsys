import math

from book_recsys.eval.metrics import (intra_list_diversity, mrr, ndcg_at_k, recall_at_k,
                                      serendipity_at_k)


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


def test_intra_list_diversity_identical_items_is_zero():
    vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    assert intra_list_diversity(["a", "b"], vectors, k=2) == 0.0


def test_intra_list_diversity_orthogonal_items_is_one():
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    assert intra_list_diversity(["a", "b"], vectors, k=2) == 1.0


def test_intra_list_diversity_averages_over_pairs():
    # pairs (a,b)=1, (a,c)=0, (b,c)=1 -> mean 2/3
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 0.0]}
    assert intra_list_diversity(["a", "b", "c"], vectors, k=3) == 2.0 / 3.0


def test_intra_list_diversity_respects_k():
    vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
    # k=2 -> only a,b (identical) -> 0.0, ignoring the diverse c
    assert intra_list_diversity(["a", "b", "c"], vectors, k=2) == 0.0


def test_intra_list_diversity_skips_items_without_vectors():
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    # 'x' has no vector -> only a,b form a pair -> 1.0
    assert intra_list_diversity(["a", "x", "b"], vectors, k=3) == 1.0


def test_intra_list_diversity_zero_when_fewer_than_two_vectors():
    assert intra_list_diversity(["a"], {"a": [1.0, 0.0]}, k=3) == 0.0


def test_serendipity_relevant_niche_item_scores_self_information():
    # popularity 0.25 -> unexpectedness -log2(0.25) = 2.0
    assert serendipity_at_k(["a"], {"a"}, {"a": 0.25}, k=1) == 2.0


def test_serendipity_non_relevant_items_contribute_zero():
    # a relevant (-log2(0.5)=1.0), b not relevant (0) -> mean over k=2 = 0.5
    assert serendipity_at_k(["a", "b"], {"a"}, {"a": 0.5, "b": 0.5}, k=2) == 0.5


def test_serendipity_relevant_blockbuster_scores_zero():
    # most-popular item (popularity 1.0) -> -log2(1)=0, no surprise even though relevant
    assert serendipity_at_k(["a"], {"a"}, {"a": 1.0}, k=1) == 0.0


def test_serendipity_missing_popularity_treated_as_popular():
    # unknown popularity -> treated as 1.0 -> unexpectedness 0 -> never inflates the score
    assert serendipity_at_k(["a"], {"a"}, {}, k=1) == 0.0


def test_serendipity_empty_recommendation_is_zero():
    assert serendipity_at_k([], {"a"}, {"a": 0.5}, k=5) == 0.0


def test_intra_list_diversity_zero_vector_counts_as_dissimilar():
    # a zero vector has undefined direction -> cosine 0 -> dissimilarity 1.0
    vectors = {"a": [0.0, 0.0], "b": [1.0, 0.0]}
    assert intra_list_diversity(["a", "b"], vectors, k=2) == 1.0
