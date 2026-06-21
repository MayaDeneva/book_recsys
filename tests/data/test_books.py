import gzip
import json

from book_recsys.data.books import stream_books_json


def _write(path, records):
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


SAMPLE = [
    {
        "book_id": "b0",
        "title": "Dragon Dawn",
        "description": "A young mage rises.",
        "language_code": "eng",
        "work_id": "w0",
        "image_url": "http://img/b0.jpg",
        "authors": [{
            "author_id": "a1",
            "role": ""
        }],
        "popular_shelves": [{
            "count": "50",
            "name": "fantasy"
        }, {
            "count": "20",
            "name": "magic"
        }]
    },
    {
        "book_id": "b1",
        "title": "No Shelves",
        "description": "",
        "language_code": "eng"
    },
]


def test_streams_catalog_columns(tmp_path):
    p = tmp_path / "books.json.gz"
    _write(p, SAMPLE)
    df = next(stream_books_json(p, chunksize=100))
    assert list(df.columns) == [
        "book_id", "title", "description", "language_code", "shelves", "author_id", "work_id",
        "image_url"
    ]


def test_extracts_image_url(tmp_path):
    p = tmp_path / "books.json.gz"
    _write(p, SAMPLE)
    df = next(stream_books_json(p, chunksize=100))
    assert df.iloc[0]["image_url"] == "http://img/b0.jpg"
    assert df.iloc[1]["image_url"] == ""  # no image_url field -> ""


def test_extracts_work_id(tmp_path):
    p = tmp_path / "books.json.gz"
    _write(p, SAMPLE)
    df = next(stream_books_json(p, chunksize=100))
    assert df.iloc[0]["work_id"] == "w0"
    assert df.iloc[1]["work_id"] == ""  # no work_id field -> ""


def test_extracts_primary_author_id(tmp_path):
    p = tmp_path / "books.json.gz"
    _write(p, SAMPLE)
    df = next(stream_books_json(p, chunksize=100))
    assert df.iloc[0]["author_id"] == "a1"
    assert df.iloc[1]["author_id"] == ""  # no authors field -> ""


def test_extracts_top_shelf_names(tmp_path):
    p = tmp_path / "books.json.gz"
    _write(p, SAMPLE)
    df = next(stream_books_json(p, chunksize=100))
    assert df.iloc[0]["shelves"] == ["fantasy", "magic"]
    assert df.iloc[1]["shelves"] == []


def test_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "books.json.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps(SAMPLE[0]) + "\n")
        f.write("\n")
        f.write(json.dumps(SAMPLE[1]) + "\n")
    df = next(stream_books_json(p, chunksize=100))
    assert len(df) == 2


def test_reads_plain_uncompressed_file(tmp_path):
    p = tmp_path / "books.json"
    with open(p, "w") as f:
        f.write(json.dumps(SAMPLE[0]) + "\n")
    df = next(stream_books_json(p, chunksize=100))
    assert df.iloc[0]["book_id"] == "b0" and df.iloc[0]["author_id"] == "a1"


def test_chunks_respect_chunksize(tmp_path):
    p = tmp_path / "books.json.gz"
    _write(p, SAMPLE)
    assert len(list(stream_books_json(p, chunksize=1))) == 2  # mid-stream yield
