"""Build the single text document used to represent each book."""
import pandas as pd

DEFAULT_FIELDS = ("title", "plot", "shelves")


def build_book_document(title: str, description: str, shelves: list[str],
                        author: str | None = None, fields=DEFAULT_FIELDS) -> str:
    """Book document from the selected fields. `author` (when given) is appended to
    the title as "by {author}". `fields` is any subset/order of DEFAULT_FIELDS."""
    parts = []
    if "title" in fields:
        line = f"Title: {title}"
        if author:
            line += f" by {author}"
        parts.append(line)
    if "plot" in fields:
        parts.append(f"Plot: {description}")
    if "shelves" in fields:
        items = [] if shelves is None or isinstance(shelves, float) else list(shelves)
        if len(items) > 0:
            parts.append("Themes/shelves: " + ", ".join(map(str, items)))
    return "\n".join(parts)


def build_documents(catalog: pd.DataFrame, fields=DEFAULT_FIELDS) -> list[str]:
    """Build one document per catalog row (title, description, shelves, optional author)."""
    has_author = "author" in catalog.columns
    docs = []
    for row in catalog.itertuples(index=False):
        author = row.author if has_author else None
        docs.append(build_book_document(row.title, row.description, row.shelves, author, fields))
    return docs
