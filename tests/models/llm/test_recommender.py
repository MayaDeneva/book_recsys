import numpy as np

from book_recsys.llm.retrieve import Retriever
from book_recsys.models.llm.recommender import LLMRecommender

BOOK_IDS = ["b0", "b1", "b2", "b3"]
EMB = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
DOCS = {b: f"doc {b}" for b in BOOK_IDS}


class _Encoder:
    def encode(self, texts):
        return np.array([[0.0, 1.0] for _ in texts])  # points at axis 1 (b2/b3)


class _RankClient:
    """Scores b1 highest, then b3, then b0; everything else 0."""

    def complete(self, prompt):
        return '[{"id":"b1","score":9},{"id":"b3","score":7},{"id":"b0","score":3}]'


def _recommender():
    retriever = Retriever(BOOK_IDS, EMB, encoder=_Encoder())
    return LLMRecommender(retriever, DOCS, _RankClient(), retrieve_n=4).fit()


def test_history_query_excludes_seen_and_reranks():
    out = _recommender().recommend(["b0"], k=2)
    assert "b0" not in out          # seen excluded
    assert out[0] == "b1"           # highest reranked, unseen


def test_text_query_returns_ranked_books():
    out = _recommender().recommend("a calm gift", k=2)
    assert out[0] == "b1"           # top reranked


def test_dict_query_fuses_history_and_intent():
    out = _recommender().recommend({"history": ["b0"], "query": "a calm gift"}, k=2)
    assert "b0" not in out
    assert out[0] == "b1"


def test_empty_query_returns_empty():
    assert _recommender().recommend([], k=2) == []


def test_fit_returns_self():
    retriever = Retriever(BOOK_IDS, EMB, encoder=_Encoder())
    rec = LLMRecommender(retriever, DOCS, _RankClient())
    assert rec.fit() is rec
