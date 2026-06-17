"""SVD collaborative-filtering recommender with user fold-in."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from book_recsys.data.schema import BOOK, RATING, USER


class SvdRecommender:
    """Factorize the user-item rating matrix; rank by folded-in user preference."""

    def __init__(self, n_factors: int = 50) -> None:
        self.n_factors = n_factors
        self._ids: list = []
        self._pos: dict = {}
        self._item_factors = None

    def fit(self, train_data) -> "SvdRecommender":
        users = train_data[USER].astype("category")
        books = train_data[BOOK].astype("category")
        self._ids = list(books.cat.categories)
        self._pos = {b: i for i, b in enumerate(self._ids)}
        values = train_data[RATING].to_numpy(dtype=float)
        values[values == 0] = 1.0  # treat unrated interactions as implicit positives
        matrix = sp.csr_matrix(
            (values, (users.cat.codes.to_numpy(), books.cat.codes.to_numpy())),
            shape=(users.cat.categories.size, len(self._ids)),
        )
        k = min(self.n_factors, min(matrix.shape) - 1)
        _, _, vt = svds(matrix, k=k)
        self._item_factors = vt.T  # (n_items x k)
        return self

    def recommend(self, query, k: int) -> list:
        idx = [self._pos[b] for b in query if b in self._pos]
        if not idx or self._item_factors is None:
            return []
        user_vector = self._item_factors[idx].mean(axis=0)
        scores = self._item_factors @ user_vector
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
        if not idx or self._item_factors is None:
            return [float("-inf")] * len(item_ids)
        user_vector = self._item_factors[idx].mean(axis=0)
        return [float(self._item_factors[self._pos[b]] @ user_vector) if b in self._pos
                else float("-inf") for b in item_ids]
