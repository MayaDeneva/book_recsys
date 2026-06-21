import gzip
import json

import pandas as pd

from book_recsys.data.authors import attach_author_names, load_author_names


def _write(path, records):
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_author_names(tmp_path):
    p = tmp_path / "authors.json.gz"
    _write(p, [{
        "author_id": "a1",
        "name": "Ursula K. Le Guin"
    }, {
        "author_id": "a2",
        "name": "Frank Herbert"
    }])
    names = load_author_names(p)
    assert names == {"a1": "Ursula K. Le Guin", "a2": "Frank Herbert"}


def test_load_skips_blank_lines(tmp_path):
    p = tmp_path / "authors.json.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps({"author_id": "a1", "name": "X"}) + "\n")
        f.write("\n")
    assert load_author_names(p) == {"a1": "X"}


def test_attach_author_names_adds_author_column():
    catalog = pd.DataFrame({"book_id": ["b0", "b1"], "author_id": ["a1", "a9"]})
    out = attach_author_names(catalog, {"a1": "Le Guin"})
    assert list(out["author"]) == ["Le Guin", ""]  # unknown id -> ""
