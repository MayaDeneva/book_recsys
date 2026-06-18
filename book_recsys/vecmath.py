"""Small shared vector helpers used by the swipe feed and the LLM-steered ranker."""
import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zero (no divide error)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; an all-equal array maps to all zeros."""
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
