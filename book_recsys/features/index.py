"""FAISS cosine-similarity index over book embeddings."""
import faiss
import numpy as np


def build_index(vectors) -> "faiss.Index":
    """Build an inner-product index over L2-normalized vectors (= cosine)."""
    matrix = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def search(index: "faiss.Index", queries, k: int):
    """Return (scores, indices) of the top-k nearest items for each query row."""
    matrix = np.ascontiguousarray(np.asarray(queries, dtype="float32"))
    faiss.normalize_L2(matrix)
    return index.search(matrix, k)
