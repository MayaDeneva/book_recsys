"""Load Goodreads genre votes (goodreads_book_genres_initial.json) -> a top-genre string.

Each line is {"book_id": "...", "genres": {"history, biography": 12, "fiction": 5, ...}}.
We keep the top-n genre labels by vote count, comma-joined, as a clean curated genre signal
(the popular_shelves in the catalog mix genre with organizational tags like "to-read").
"""
import gzip
import json
from pathlib import Path


def top_genres(votes: dict, n: int = 3) -> str:
    """Top-n genre labels by vote count, comma-joined (highest first; ties keep input order)."""
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(label for label, _ in ranked[:n])


def stream_genres(path: str | Path, n: int = 3) -> dict:
    """Map book_id -> top-n genre string from the (gzipped) genre JSON-lines file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict = {}
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[obj["book_id"]] = top_genres(obj.get("genres") or {}, n)
    return out
