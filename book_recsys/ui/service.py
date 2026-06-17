"""UI service: search the catalog by title, run recommenders, format rich labels.

Works in book_ids internally (titles are not unique), and renders a display label of
`Title by Author — description snippet` so the user can tell similar/duplicate titles
apart. Author and description columns are optional (label degrades gracefully).
"""
import pandas as pd


class RecommenderService:
    """Wraps the catalog + recommenders so a UI can search titles and show context.

    history_recommenders: {name: Recommender} for UC1 (recommend(history_ids, k)).
    similar_recommender: a Recommender for UC4 (recommend(anchor_id, k)).
    """

    def __init__(self, catalog: pd.DataFrame, history_recommenders: dict,
                 similar_recommender) -> None:
        self._ids = list(catalog["book_id"])
        self._titles = list(catalog["title"])
        self._title = dict(zip(self._ids, self._titles))
        self._author = self._optional_col(catalog, "author")
        self._desc = self._optional_col(catalog, "description")
        self._image = self._optional_col(catalog, "image_url")
        self._hist = history_recommenders
        self._similar = similar_recommender

    @staticmethod
    def _optional_col(catalog: pd.DataFrame, name: str) -> dict:
        values = catalog[name] if name in catalog.columns else [""] * len(catalog)
        return dict(zip(catalog["book_id"], values))

    def label(self, book_id) -> str:
        """`Title by Author — description snippet`; parts omitted when missing."""
        title = self._title.get(book_id, str(book_id))
        author = str(self._author.get(book_id) or "").strip()
        base = f"{title} by {author}" if author else str(title)
        desc = str(self._desc.get(book_id) or "").strip().replace("\n", " ")
        return f"{base} — {desc[:70]}…" if desc else base

    def card(self, book_id) -> dict:
        """Full card data for the swipe UI: title, author, synopsis, and cover image.

        Goodreads' `nophoto` placeholder (cover-less books) is mapped to "" so the UI shows
        its text fallback instead of the grey placeholder image.
        """
        cover = str(self._image.get(book_id) or "").strip()
        return {
            "book_id": book_id,
            "title": str(self._title.get(book_id, book_id)),
            "author": str(self._author.get(book_id) or "").strip(),
            "description": str(self._desc.get(book_id) or "").strip().replace("\n", " "),
            "image_url": "" if "nophoto" in cover else cover,
        }

    def search(self, query: str, limit: int = 10) -> list:
        """book_ids whose title OR author contains `query` (case-insensitive), up to `limit`."""
        q = query.lower()
        out = []
        for book_id, title in zip(self._ids, self._titles):
            author = str(self._author.get(book_id) or "").lower()
            if q in str(title).lower() or q in author:
                out.append(book_id)
                if len(out) >= limit:
                    break
        return out

    def methods(self) -> list:
        """Available UC1 history-recommender method names (keys to recommend_by_history)."""
        return list(self._hist)

    def recommend_by_history(self, book_ids, method: str, k: int = 10) -> list:
        """UC1: recommend from liked book_ids; returns display labels."""
        recs = self._hist[method].recommend(list(book_ids), k)
        return [self.label(b) for b in recs]

    def similar_to(self, book_id, k: int = 10) -> list:
        """UC4: books similar to an anchor book_id; returns display labels."""
        return [self.label(b) for b in self._similar.recommend(book_id, k)]
