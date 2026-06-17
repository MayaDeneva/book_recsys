"""Item-item similarity recommender: top-k nearest neighbors of an anchor book."""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class SimilarItemsRecommender:
    """Rank catalog books by cosine similarity to a single anchor book.

    `query` passed to recommend() is the anchor book id (not a history list).
    """

    def __init__(self, book_ids, matrix) -> None:
        self._ids = list(book_ids)
        self._pos = {b: i for i, b in enumerate(self._ids)}
        self._matrix = normalize(matrix, axis=1)

    def fit(self, train_data=None) -> "SimilarItemsRecommender":
        return self

    def recommend(self, query, k: int) -> list:
        anchor = query
        if anchor not in self._pos:
            return []
        row = self._matrix[self._pos[anchor]]
        if sp.issparse(row):
            row = row.toarray()
        profile = np.asarray(row).ravel()
        scores = np.asarray(self._matrix @ profile).ravel()
        out = []
        for i in np.argsort(-scores):
            book = self._ids[i]
            if book != anchor:
                out.append(book)
                if len(out) == k:
                    break
        return out
