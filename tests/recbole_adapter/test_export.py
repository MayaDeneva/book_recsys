from book_recsys.recbole_adapter.export import recbole_predictions


def test_maps_internal_ids_to_tokens():
    user_internal = [1, 2]
    item_internal_matrix = [[10, 11], [12, 10]]
    uid2token = {1: "userA", 2: "userB"}
    iid2token = {10: "bookX", 11: "bookY", 12: "bookZ"}
    preds = recbole_predictions(user_internal, item_internal_matrix, uid2token, iid2token)
    assert preds == {"userA": ["bookX", "bookY"], "userB": ["bookZ", "bookX"]}


def test_empty_batch_returns_empty_dict():
    assert recbole_predictions([], [], {}, {}) == {}
