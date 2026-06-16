import gzip
import json
from datetime import datetime, timedelta, timezone

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.data.ingest import stream_interactions_json


SAMPLE = [
    {"user_id": "u0", "book_id": "b0", "rating": 5,
     "date_added": "Tue Nov 17 11:37:35 -0800 2017"},
    {"user_id": "u1", "book_id": "b1", "rating": 0,
     "date_added": "Wed Jan 03 09:00:00 -0800 2018"},
]


def _write_jsonl_gz(path, records, *, blank_line=False):
    with gzip.open(path, "wt") as f:
        for i, r in enumerate(records):
            f.write(json.dumps(r) + "\n")
            if blank_line and i == 0:
                f.write("\n")  # exercise the skip-blank-line branch


def test_streams_normalized_chunks(tmp_path):
    p = tmp_path / "inter.json.gz"
    _write_jsonl_gz(p, SAMPLE, blank_line=True)
    chunks = list(stream_interactions_json(p, chunksize=1))
    assert len(chunks) == 2  # chunksize=1 -> two chunks (blank line skipped)
    df = chunks[0]
    assert list(df.columns) == [USER, BOOK, RATING, TS]


def test_parses_fields_and_timestamp(tmp_path):
    p = tmp_path / "inter.json.gz"
    _write_jsonl_gz(p, SAMPLE)
    df = next(stream_interactions_json(p, chunksize=100))
    assert df.iloc[0][USER] == "u0"
    assert df.iloc[0][BOOK] == "b0"
    assert int(df.iloc[0][RATING]) == 5
    expected = int(datetime(2017, 11, 17, 11, 37, 35,
                            tzinfo=timezone(timedelta(hours=-8))).timestamp())
    assert int(df.iloc[0][TS]) == expected


def test_falls_back_to_read_at_then_zero(tmp_path):
    records = [
        {"user_id": "u2", "book_id": "b2", "rating": 3,
         "read_at": "Tue Nov 17 11:37:35 -0800 2017"},  # no date_added
        {"user_id": "u3", "book_id": "b3", "rating": 4},  # no timestamp at all
    ]
    p = tmp_path / "inter.json.gz"
    _write_jsonl_gz(p, records)
    df = next(stream_interactions_json(p, chunksize=100))
    assert int(df.iloc[0][TS]) > 0          # used read_at
    assert int(df.iloc[1][TS]) == 0         # no timestamp -> 0


def test_reads_plain_uncompressed_file(tmp_path):
    p = tmp_path / "inter.json"  # no .gz suffix -> plain open
    with open(p, "w") as f:
        f.write(json.dumps(SAMPLE[0]) + "\n")
    df = next(stream_interactions_json(p, chunksize=100))
    assert df.iloc[0][USER] == "u0"
