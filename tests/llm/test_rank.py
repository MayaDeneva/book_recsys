import numpy as np

from book_recsys.llm.rank import SteeredRanker
from book_recsys.llm.steer import SteeringState

BOOK_IDS = ["a", "b", "c", "d", "e"]
# Simple 2-d embeddings so cosine is predictable.
EMB = np.array([[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1]], dtype="float32")


class _CF:

    def recommend(self, history, n):
        return ["a", "b", "c"][:n]


class _Retriever:

    def by_history(self, history, n):
        return ["b", "a"][:n]

    def by_text(self, text, n):
        return ["d", "e", "c"][:n]


class _Similar:

    def recommend(self, anchor_id, n):
        return ["e", "c"][:n]


class _Encoder:

    def __init__(self, vec):
        self._vec = vec

    def encode(self, phrases):
        return np.array([self._vec for _ in phrases], dtype="float32")


def _ranker(**kw):
    return SteeredRanker(_CF(), _Retriever(), _Similar(), EMB, BOOK_IDS, _Encoder([1, 0]), **kw)


def test_rank_excludes_history_and_seen():
    out = _ranker().rank(SteeringState(history_weight=1.0), history_ids=["a"], seen={"b"})
    assert "a" not in out and "b" not in out


def test_rank_topic_only_uses_text_list():
    # history_weight 0 -> only by_text ("d","e","c") drives ranking.
    out = _ranker().rank(SteeringState(history_weight=0.0, topic="x"), history_ids=[], seen=set())
    assert set(out) <= {"c", "d", "e"}
    assert out[0] == "d"


def test_rank_avoid_penalty_demotes_similar_book():
    # Avoid vector [1,0] == book 'a'. Without 'a' (history), test 'c'(=[1,1]) vs 'e'(=[0,-1]).
    # 'c' is more similar to the avoid vector, so it should rank below 'e'.
    state = SteeringState(history_weight=0.0, topic="x", avoid=["spiky"])
    out = _ranker().rank(state, history_ids=[], seen=set())
    assert out.index("e") < out.index("c")


def test_rank_genre_filter_includes_only_matching():
    genre = {"c": "fantasy", "d": "history", "e": "fantasy"}
    out = SteeredRanker(_CF(),
                        _Retriever(),
                        _Similar(),
                        EMB,
                        BOOK_IDS,
                        _Encoder([1, 0]),
                        catalog_genre=genre).rank(SteeringState(history_weight=0.0,
                                                                topic="x",
                                                                genre="fantasy"),
                                                  history_ids=[],
                                                  seen=set())
    assert set(out) <= {"c", "e"}
    assert "d" not in out


def test_rank_anchor_adds_similar_results():
    out = _ranker().rank(SteeringState(history_weight=1.0),
                         history_ids=[],
                         seen=set(),
                         anchor_id="z")
    assert "e" in out  # 'e' comes only from similar.recommend


def test_rank_returns_at_most_k():
    out = _ranker().rank(SteeringState(history_weight=0.5, topic="x"),
                         history_ids=[],
                         seen=set(),
                         k=2)
    assert len(out) == 2


def test_rank_no_signals_returns_empty():
    # No history and no topic -> nothing to fuse.
    out = _ranker().rank(SteeringState(history_weight=1.0), history_ids=[], seen=set())
    assert out == []


def test_rank_avoid_penalty_zero_for_unknown_candidate():

    class _RetrieverWithUnknown:

        def by_history(self, history, n):
            return []

        def by_text(self, text, n):
            return ["d", "zzz"]  # 'zzz' is not in BOOK_IDS / has no embedding row

    ranker = SteeredRanker(_CF(), _RetrieverWithUnknown(), _Similar(), EMB, BOOK_IDS,
                           _Encoder([1, 0]))
    out = ranker.rank(SteeringState(history_weight=0.0, topic="x", avoid=["spiky"]),
                      history_ids=[],
                      seen=set())
    assert "zzz" in out  # unknown candidate survives (penalty 0), no crash


def test_rank_genre_filter_removing_all_returns_empty():
    genre = {"c": "fantasy", "d": "history", "e": "fantasy"}
    out = SteeredRanker(_CF(),
                        _Retriever(),
                        _Similar(),
                        EMB,
                        BOOK_IDS,
                        _Encoder([1, 0]),
                        catalog_genre=genre).rank(SteeringState(history_weight=0.0,
                                                                topic="x",
                                                                genre="nonexistent"),
                                                  history_ids=[],
                                                  seen=set())
    assert out == []  # genre filter removes every candidate


def test_rank_with_reasons_topic_only():
    pairs = _ranker().rank_with_reasons(SteeringState(history_weight=0.0, topic="WWII subs"),
                                        history_ids=[],
                                        seen=set())
    reasons = dict(pairs)
    assert pairs[0][0] == "d"
    assert reasons["d"] == "Matches your topic: WWII subs"


def test_rank_with_reasons_history_only_single_clause():
    # history surfaces from BOTH cf and by_history -> collapses to ONE clause.
    pairs = _ranker().rank_with_reasons(SteeringState(history_weight=1.0),
                                        history_ids=["x"],
                                        seen=set())
    assert pairs, "expected some history-based picks"
    assert all(r == "Similar to your reading history" for _, r in pairs)


def test_rank_with_reasons_anchor_clause():
    pairs = _ranker().rank_with_reasons(SteeringState(history_weight=1.0, anchor_book="Dune"),
                                        history_ids=[],
                                        seen=set(),
                                        anchor_id="z")
    assert dict(pairs)["e"] == "Like Dune"


def test_rank_with_reasons_combines_signals_in_order():
    # 'c' appears in by_text ("d","e","c") AND similar ("e","c") -> topic + anchor clauses.
    pairs = _ranker().rank_with_reasons(SteeringState(history_weight=0.0,
                                                      topic="cozy",
                                                      anchor_book="Dune"),
                                        history_ids=[],
                                        seen=set(),
                                        anchor_id="z")
    reasons = dict(pairs)
    assert reasons["c"] == "Matches your topic: cozy · like Dune"


def test_rank_delegates_to_rank_with_reasons():
    state = SteeringState(history_weight=0.5, topic="x")
    r = _ranker()
    ids = r.rank(state, history_ids=[], seen=set())
    pairs = r.rank_with_reasons(state, history_ids=[], seen=set())
    assert ids == [b for b, _ in pairs]


def test_reason_empty_when_no_signal():
    assert SteeredRanker._reason(set(), SteeringState(history_weight=0.5)) == ""
