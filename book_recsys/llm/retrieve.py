"""Retrieve candidate books from the embedding catalog via FAISS."""
import numpy as np

from book_recsys.features.index import build_index, search


class Retriever:
    """ANN retrieval over book embeddings, by user history or by encoded text."""

    def __init__(self, book_ids, embeddings, encoder=None, index=None) -> None:
        self._ids = list(book_ids)
        self._pos = {b: i for i, b in enumerate(self._ids)}
        self._emb = np.asarray(embeddings, dtype="float32")
        self._encoder = encoder
        self._index = index if index is not None else build_index(self._emb)

    def by_vector(self, vector, n: int) -> list:
        _, idx = search(self._index, np.asarray([vector], dtype="float32"), n)
        return [self._ids[i] for i in idx[0] if i != -1]

    def by_history(self, history_ids, n: int) -> list:
        rows = [self._pos[b] for b in history_ids if b in self._pos]
        if not rows:
            return []
        return self.by_vector(self._emb[rows].mean(axis=0), n)

    def by_text(self, text: str, n: int) -> list:
        vector = np.asarray(self._encoder.encode([text]))[0]
        return self.by_vector(vector, n)
