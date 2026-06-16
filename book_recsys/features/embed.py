"""Disk-cached document embedding. The encoder is injected for testability."""
from pathlib import Path

import numpy as np


def embed_documents(documents, encoder, cache_path) -> np.ndarray:
    """Return embeddings for `documents`, caching to `cache_path` (.npy).

    If the cache exists it is loaded and the encoder is not called. Otherwise the
    encoder embeds the documents and the result is saved. `encoder` is any object
    with `.encode(list[str]) -> array-like`.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)
    vectors = np.asarray(encoder.encode(list(documents)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors
