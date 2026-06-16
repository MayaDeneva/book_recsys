"""Streamed ingestion of Goodreads book metadata into a catalog frame."""
import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

_COLUMNS = ["book_id", "title", "description", "language_code", "shelves", "author_id",
            "work_id"]


def _top_shelves(obj: dict, n: int = 5) -> list[str]:
    """Top-n popular shelf names (the source list is already count-descending)."""
    shelves = obj.get("popular_shelves") or []
    return [s["name"] for s in shelves[:n]]


def _primary_author_id(obj: dict) -> str:
    authors = obj.get("authors") or []
    return authors[0]["author_id"] if authors else ""


def _normalize_book(obj: dict) -> dict:
    return {
        "book_id": obj["book_id"],
        "title": obj.get("title", ""),
        "description": obj.get("description", ""),
        "language_code": obj.get("language_code", ""),
        "shelves": _top_shelves(obj),
        "author_id": _primary_author_id(obj),
        "work_id": obj.get("work_id", ""),   # groups editions of one work (dedup key)
    }


def stream_books_json(path: str | Path,
                      chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    """Yield catalog-metadata DataFrames from a (gzipped) books JSON-lines file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    buffer: list[dict] = []
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            buffer.append(_normalize_book(json.loads(line)))
            if len(buffer) >= chunksize:
                yield pd.DataFrame(buffer, columns=_COLUMNS)
                buffer = []
    if buffer:
        yield pd.DataFrame(buffer, columns=_COLUMNS)
