"""Build the single text document used to represent each book."""
import pandas as pd

DEFAULT_FIELDS = ("title", "plot", "shelves")
# "genre" is opt-in (needs a `genre` catalog column from scripts/enrich_catalog_genres.py).
# Add it to `fields` to run the +genre ablation, e.g. ("title", "genre", "plot", "shelves").

# Behavioural / status / format shelves that aren't genres (EDA: ~38% of shelf occurrences,
# `to-read` on 99% of books). Stripped from the embedded "Themes/shelves" line so it carries
# genre signal instead of near-ubiquitous noise.
NOISE_SHELVES = frozenset({
    "to-read",
    "currently-reading",
    "favorites",
    "favourites",
    "owned",
    "default",
    "series",
    "read",
    "to-buy",
    "wish-list",
    "wishlist",
    "library",
    "dnf",
    "did-not-finish",
    "re-read",
    "reread",
    "my-books",
    "books-i-own",
    "owned-books",
    "my-library",
    "to-read-fiction",
    "kindle",
    "ebook",
    "e-book",
    "ebooks",
    "audiobook",
    "audiobooks",
    "audio",
    "paperback",
    "hardcover",
    "nook",
})


def build_book_document(title: str,
                        description: str,
                        shelves: list[str],
                        author: str | None = None,
                        genre: str | None = None,
                        fields=DEFAULT_FIELDS) -> str:
    """Book document from the selected fields. `author` (when given) is appended to the
    title as "by {author}"; `genre` is the curated genre string (used only if "genre" is in
    `fields`). `fields` is any subset/order of DEFAULT_FIELDS + ("genre",)."""
    parts = []
    if "title" in fields:
        line = f"Title: {title}"
        if author:
            line += f" by {author}"
        parts.append(line)
    if "genre" in fields and genre:
        parts.append(f"Genre: {genre}")
    if "plot" in fields:
        parts.append(f"Plot: {description}")
    if "shelves" in fields:
        items = [] if shelves is None or isinstance(shelves, float) else list(shelves)
        items = [s for s in items if str(s).lower() not in NOISE_SHELVES]  # drop behavioural noise
        if items:
            parts.append("Themes/shelves: " + ", ".join(map(str, items)))
    return "\n".join(parts)


def build_documents(catalog: pd.DataFrame, fields=DEFAULT_FIELDS) -> list[str]:
    """One document per catalog row (title, description, shelves, optional author/genre)."""
    has_author = "author" in catalog.columns
    has_genre = "genre" in catalog.columns
    docs = []
    for row in catalog.itertuples(index=False):
        author = row.author if has_author else None
        genre = row.genre if has_genre else None
        docs.append(
            build_book_document(row.title, row.description, row.shelves, author, genre, fields))
    return docs
