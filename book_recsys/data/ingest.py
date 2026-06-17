"""Streamed ingestion of Goodreads interaction JSON-lines into the normalized schema."""
import gzip
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER

_GOODREADS_DATE = "%a %b %d %H:%M:%S %z %Y"


def _parse_ts(raw: str | None) -> int:
    if not raw:
        return 0
    return int(datetime.strptime(raw, _GOODREADS_DATE).timestamp())


def _normalize_record(obj: dict) -> dict:
    raw_ts = obj.get("date_added") or obj.get("read_at")
    return {
        USER: obj["user_id"],
        BOOK: obj["book_id"],
        RATING: int(obj.get("rating") or 0),
        TS: _parse_ts(raw_ts),
    }


def stream_interactions_json(path: str | Path,
                             chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield normalized interaction DataFrames from a (gzipped) JSON-lines file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    buffer: list[dict] = []
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            buffer.append(_normalize_record(json.loads(line)))
            if len(buffer) >= chunksize:
                yield pd.DataFrame(buffer, columns=[USER, BOOK, RATING, TS])
                buffer = []
    if buffer:
        yield pd.DataFrame(buffer, columns=[USER, BOOK, RATING, TS])
