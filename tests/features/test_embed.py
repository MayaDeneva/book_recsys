import numpy as np

from book_recsys.features.embed import embed_documents


class _CountingEncoder:

    def __init__(self):
        self.calls = 0

    def encode(self, docs):
        self.calls += 1
        return np.array([[float(len(d)), 1.0] for d in docs])


def test_encodes_and_writes_cache(tmp_path):
    enc = _CountingEncoder()
    cache = tmp_path / "emb.npy"
    out = embed_documents(["aa", "bbbb"], enc, cache)
    assert out.shape == (2, 2)
    assert cache.exists()
    assert enc.calls == 1


def test_second_call_uses_cache_without_encoding(tmp_path):
    enc = _CountingEncoder()
    cache = tmp_path / "emb.npy"
    first = embed_documents(["aa", "bbbb"], enc, cache)
    second = embed_documents(["aa", "bbbb"], enc, cache)
    assert enc.calls == 1  # not re-encoded
    assert np.array_equal(first, second)


def test_creates_missing_parent_dirs(tmp_path):
    enc = _CountingEncoder()
    cache = tmp_path / "nested" / "dir" / "emb.npy"
    embed_documents(["x"], enc, cache)
    assert cache.exists()
