from book_recsys.llm.fusion import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion


def test_item_high_in_both_lists_ranks_first():
    a = ["x", "a", "b"]
    b = ["x", "c", "d"]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0] == "x"


def test_merges_disjoint_lists():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    assert set(fused) == {"a", "b"}


def test_single_list_preserves_order():
    assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []


def test_weighted_rrf_weight_one_list_dominates():
    # 'a' leads list A (weight 10), 'b' leads list B (weight 1) -> 'a' first.
    out = weighted_reciprocal_rank_fusion([(["a", "b"], 10.0), (["b", "a"], 1.0)])
    assert out[0] == "a"


def test_weighted_rrf_zero_weight_items_rank_last_not_dropped():
    # A zero-weight list contributes 0; its items remain in the output but rank
    # below any positively-weighted item. (Dropping an "off" signal is the ranker's
    # job, not this general util's — it never silently discards items.)
    out = weighted_reciprocal_rank_fusion([(["h1", "h2"], 1.0), (["t1", "t2"], 0.0)])
    assert out[:2] == ["h1", "h2"]  # positive-weight items rank first
    assert set(out) == {"h1", "h2", "t1", "t2"}  # zero-contribution items NOT dropped


def test_weighted_rrf_missing_items_contribute_zero():
    out = weighted_reciprocal_rank_fusion([(["a"], 1.0), (["b"], 1.0)])
    assert set(out) == {"a", "b"}
