import numpy as np

from book_recsys.llm.retrieve import Retriever

BOOK_IDS = ["b0", "b1", "b2", "b3"]
EMB = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])


class _Encoder:
    def encode(self, texts):
        # always returns a vector pointing along axis 0
        return np.array([[1.0, 0.0] for _ in texts])


def test_by_history_returns_similar_books():
    r = Retriever(BOOK_IDS, EMB)
    out = r.by_history(["b0"], n=2)
    assert out[0] == "b0"  # nearest to itself
    assert "b1" in out     # b1 is the closest other


def test_by_history_unknown_ids_return_empty():
    r = Retriever(BOOK_IDS, EMB)
    assert r.by_history(["nope"], n=2) == []


def test_by_text_uses_encoder():
    r = Retriever(BOOK_IDS, EMB, encoder=_Encoder())
    out = r.by_text("axis-0 query", n=2)
    assert out[0] == "b0"  # encoder points at axis 0, b0 is the match


def test_n_larger_than_catalog_is_safe():
    r = Retriever(BOOK_IDS, EMB)
    out = r.by_history(["b0"], n=99)
    assert len(out) == 4  # no -1 padding leaks through
