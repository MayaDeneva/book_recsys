"""Most-popular-book baseline recommender."""
import pandas as pd

from book_recsys.data.schema import BOOK


class PopularityRecommender:
    """Ranks books by global interaction count, excluding the user's seen books."""

    def __init__(self) -> None:
        self._ranked: list = []

    def fit(self, train_data: pd.DataFrame) -> "PopularityRecommender":
        self._ranked = train_data[BOOK].value_counts().index.tolist()
        return self

    def recommend(self, query, k: int) -> list:
        seen = set(query)
        out = []
        for book in self._ranked:
            if book not in seen:
                out.append(book)
                if len(out) == k:
                    break
        return out

    def score_items(self, query, item_ids) -> list:
        rank = {book: i for i, book in enumerate(self._ranked)}
        return [float(-rank[b]) if b in rank else float("-inf") for b in item_ids]
