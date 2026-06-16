"""Content-based recommender: rank books by cosine to the user's profile."""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class ContentRecommender:
    """Build a user profile from their history's item vectors and rank by cosine.

    `matrix` is any (n_books x d) item representation — sparse TF-IDF/BoW or dense
    embeddings. Rows are L2-normalized at construction so a dot product is cosine.
    """

    def __init__(self, book_ids, matrix) -> None:
        self._ids = list(book_ids)
        self._pos = {b: i for i, b in enumerate(self._ids)}
        self._matrix = normalize(matrix, axis=1)

    def fit(self, train_data=None) -> "ContentRecommender":
        return self

    def recommend(self, query, k: int) -> list:
        idx = [self._pos[b] for b in query if b in self._pos]
        if not idx:
            return []
        profile = np.asarray(self._matrix[idx].mean(axis=0)).ravel()
        scores = np.asarray(self._matrix @ profile).ravel()
        seen = set(query)
        out = []
        for i in np.argsort(-scores):
            book = self._ids[i]
            if book not in seen:
                out.append(book)
                if len(out) == k:
                    break
        return out

    def score_items(self, query, item_ids) -> list:
        idx = [self._pos[b] for b in query if b in self._pos]
        if not idx:
            return [float("-inf")] * len(item_ids)
        profile = np.asarray(self._matrix[idx].mean(axis=0)).ravel()
        out = []
        for b in item_ids:
            if b in self._pos:
                row = self._matrix[self._pos[b]]
                row = row.toarray().ravel() if sp.issparse(row) else np.asarray(row).ravel()
                out.append(float(row @ profile))
            else:
                out.append(float("-inf"))
        return out
