import numpy as np

from book_recsys.ui.feed import FeedService


class FakeRec:
    """Stand-in for LearnedHybridRecommender: recommend() + score_items()."""

    def __init__(self, rec_order, scores):
        self._order = rec_order
        self._scores = scores

    def recommend(self, history, k):
        return self._order[:k]

    def score_items(self, history, candidates):
        return [self._scores[c] for c in candidates]


def test_next_excludes_seen_liked_disliked_and_ranks_by_score():
    book_ids = ["a", "b", "c", "d", "e"]
    emb = np.eye(5, dtype="float32")
    rec = FakeRec(rec_order=["b", "c", "d", "e"],
                  scores={"b": 0.2, "c": 0.9, "d": 0.5, "e": 0.1})
    fs = FeedService(rec, emb, book_ids, pool=10)
    # a liked, e disliked, d seen -> all excluded; remaining b,c ranked by score desc
    out = fs.next(liked=["a"], disliked=["e"], seen=["d"], k=10, lam=0.0)
    assert out == ["c", "b"]


def test_next_empty_liked_returns_empty():
    fs = FeedService(FakeRec([], {}), np.eye(2, dtype="float32"), ["a", "b"])
    assert fs.next(liked=[], disliked=[], seen=[], k=10) == []


def test_next_respects_k():
    book_ids = ["a", "b", "c", "d"]
    rec = FakeRec(["b", "c", "d"], {"b": 0.3, "c": 0.9, "d": 0.6})
    fs = FeedService(rec, np.eye(4, dtype="float32"), book_ids, pool=10)
    assert fs.next(liked=["a"], disliked=[], seen=[], k=1, lam=0.0) == ["c"]


def test_next_returns_empty_when_all_candidates_excluded():
    book_ids = ["a", "b", "c"]
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.5, "c": 0.5})
    fs = FeedService(rec, np.eye(3, dtype="float32"), book_ids, pool=10)
    # both candidates already seen -> nothing left to recommend
    assert fs.next(liked=["a"], disliked=[], seen=["b", "c"], k=10, lam=0.0) == []


def test_next_equal_scores_keep_recommend_order():
    book_ids = ["a", "b", "c"]
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.5, "c": 0.5})
    fs = FeedService(rec, np.eye(3, dtype="float32"), book_ids, pool=10)
    # equal scores -> _minmax returns zeros -> stable sort preserves recommend order
    assert fs.next(liked=["a"], disliked=[], seen=[], k=10, lam=0.0) == ["b", "c"]


def test_penalizes_candidate_similar_to_disliked():
    # b points the same direction as disliked x; c is orthogonal. Equal base scores,
    # so with lam>0 the penalty pushes b below c.
    book_ids = ["a", "b", "c", "x"]
    emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype="float32")
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.5, "c": 0.5})
    fs = FeedService(rec, emb, book_ids, pool=10)
    # lam=0 -> no penalty, stable order keeps recommend order
    assert fs.next(liked=["a"], disliked=["x"], seen=[], k=10, lam=0.0) == ["b", "c"]
    # lam=1 -> b (cos=1 to x) penalized below c (cos=0)
    assert fs.next(liked=["a"], disliked=["x"], seen=[], k=10, lam=1.0) == ["c", "b"]


def test_no_disliked_means_no_penalty():
    book_ids = ["a", "b", "c"]
    emb = np.eye(3, dtype="float32")
    rec = FakeRec(["b", "c"], {"b": 0.9, "c": 0.1})
    fs = FeedService(rec, emb, book_ids, pool=10)
    assert fs.next(liked=["a"], disliked=[], seen=[], k=10, lam=1.0) == ["b", "c"]


def test_unknown_disliked_id_is_ignored():
    book_ids = ["a", "b", "c"]
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.9, "c": 0.1})
    fs = FeedService(rec, np.eye(3, dtype="float32"), book_ids, pool=10)
    # disliked id not in the catalog -> no embedding -> no penalty, ranking unchanged
    assert fs.next(liked=["a"], disliked=["zzz"], seen=[], k=10, lam=1.0) == ["b", "c"]


def _two_method_fs():
    book_ids = ["a", "b", "c"]
    recs = {
        "m1": FakeRec(["b", "c"], {"b": 0.9, "c": 0.1}),   # m1 prefers b
        "m2": FakeRec(["b", "c"], {"b": 0.1, "c": 0.9}),   # m2 prefers c
    }
    return FeedService(recs, np.eye(3, dtype="float32"), book_ids, pool=10)


def test_methods_lists_recommender_names_default_first():
    assert _two_method_fs().methods() == ["m1", "m2"]


def test_next_uses_selected_method():
    fs = _two_method_fs()
    assert fs.next(["a"], [], [], k=10, lam=0.0, method="m2")[0] == "c"   # m2 prefers c
    assert fs.next(["a"], [], [], k=10, lam=0.0, method="m1")[0] == "b"   # m1 prefers b


def test_next_unknown_or_no_method_falls_back_to_default():
    fs = _two_method_fs()
    assert fs.next(["a"], [], [], k=10, lam=0.0)[0] == "b"               # default = m1
    assert fs.next(["a"], [], [], k=10, lam=0.0, method="nope")[0] == "b"
