import json

import torch

from book_recsys.models.sequential.model import SASRec
from book_recsys.models.sequential.recommender import SasRecRecommender, load_state_dict_clean

TINY = dict(n_items=6, hidden_size=8, n_layers=2, n_heads=2, inner_size=16, max_seq_length=4)
TOKENS = ["[PAD]", "b1", "b2", "b3", "b4", "b5"]  # index -> book_id, 0 = PAD


def _rec():
    torch.manual_seed(0)
    return SasRecRecommender(SASRec(**TINY).eval(), TOKENS, device="cpu")


def test_recommend_returns_k_known_unseen_ids():
    recs = _rec().recommend(["b1", "b2"], k=2)
    assert len(recs) == 2
    assert set(recs) <= {"b3", "b4", "b5"}  # never recommends seen b1/b2 or PAD


def test_recommend_excludes_seen():
    recs = _rec().recommend(["b1"], k=5)
    assert "b1" not in recs


def test_recommend_ignores_unknown_query_ids():
    # An id not in the vocab is dropped from the input sequence, not crashed on.
    recs = _rec().recommend(["b1", "UNKNOWN"], k=2)
    assert len(recs) == 2


def test_recommend_empty_query_returns_empty():
    assert _rec().recommend([], k=3) == []


def test_recommend_truncates_to_max_seq_length():
    # Query longer than max_seq_length must still work (keeps the most recent items).
    recs = _rec().recommend(["b1", "b2", "b3", "b4", "b5"], k=1)
    assert len(recs) == 1


def test_score_items_unknown_is_neg_inf():
    scores = _rec().score_items(["b1"], ["b3", "UNKNOWN"])
    assert scores[1] == float("-inf")
    assert scores[0] > float("-inf")


def test_fit_returns_self():
    rec = _rec()
    assert rec.fit(None) is rec


def test_from_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = SASRec(**TINY)
    state_path = tmp_path / "state.pt"
    torch.save(net.state_dict(), state_path)
    map_path = tmp_path / "item_map.json"
    map_path.write_text(json.dumps({"iid_field": "item_id", "item_id_token": TOKENS}))
    rec = SasRecRecommender.from_checkpoint(state_path, map_path, device="cpu")
    assert rec._tokens == TOKENS
    assert rec._model.hidden_size == TINY["hidden_size"]
    assert rec._model.max_seq_length == TINY["max_seq_length"]
    assert len(rec.recommend(["b1"], k=2)) == 2


def test_score_items_empty_query_all_neg_inf():
    # When query maps to no known ids, all scores must be -inf.
    scores = _rec().score_items([], ["b1", "b2"])
    assert scores == [float("-inf"), float("-inf")]


def test_load_state_dict_clean_reads_tensors(tmp_path):
    sd = {"a": torch.zeros(2)}
    p = tmp_path / "s.pt"
    torch.save(sd, p)
    assert "a" in load_state_dict_clean(p)
