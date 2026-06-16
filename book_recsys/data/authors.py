"""Ingest Goodreads author-id -> name and attach names to a catalog."""
import gzip
import json
from pathlib import Path

import pandas as pd


def load_author_names(path) -> dict:
    """Map author_id -> name from goodreads_book_authors.json(.gz)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    names: dict = {}
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            names[obj["author_id"]] = obj.get("name", "")
    return names


def attach_author_names(catalog: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    """Return a copy of catalog with an `author` column resolved from `author_id`."""
    catalog = catalog.copy()
    catalog["author"] = catalog["author_id"].map(lambda a: name_map.get(a, ""))
    return catalog
