import numpy as np
import pytest

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
