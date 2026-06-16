from book_recsys.llm.fusion import reciprocal_rank_fusion


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
