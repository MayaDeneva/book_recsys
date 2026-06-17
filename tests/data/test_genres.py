import gzip
import json

from book_recsys.data.genres import stream_genres, top_genres


def test_top_genres_ranks_by_votes():
    assert top_genres({"fiction": 5, "fantasy": 12, "romance": 1}, n=2) == "fantasy, fiction"


def test_top_genres_empty():
    assert top_genres({}, n=3) == ""


def test_stream_genres_reads_top_n_and_skips_blanks(tmp_path):
    p = tmp_path / "g.json.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps({"book_id": "b0", "genres": {"fantasy": 10, "fiction": 3}}) + "\n")
        f.write("\n")  # blank line skipped
        f.write(json.dumps({"book_id": "b1", "genres": {}}) + "\n")
    assert stream_genres(p, n=1) == {"b0": "fantasy", "b1": ""}


def test_stream_genres_plain_file(tmp_path):
    p = tmp_path / "g.json"
    with open(p, "w") as f:
        f.write(json.dumps({"book_id": "x", "genres": {"mystery": 2}}) + "\n")
    assert stream_genres(p) == {"x": "mystery"}
