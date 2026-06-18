"""History aggregation helpers: recency weighting + (weighted) profile pooling.

Shared by the CF/content recommenders so "average my history into one vector" can become
"exponentially weight recent interactions more" without each model reimplementing it.
"""
import numpy as np


def recency_weights(timestamps, tau: float):
    """Exponential-decay weight per interaction: w_i = exp(-(t_max - t_i) / tau).

    The most recent interaction gets weight 1.0; older ones decay with scale `tau` (same time
    unit as `timestamps`). Returns a float array aligned to the input order. For the live UI
    (picks carry no timestamp) pass pick *order* as the timestamps.
    """
    t = np.asarray(timestamps, dtype=float)
    if t.size == 0:
        return t
    return np.exp(-(t.max() - t) / tau)


def aligned_weights(query, weights, pos):
    """Map `query` ids to matrix positions via `pos`, dropping unknowns. Returns (idx, w):
    `w` is the parallel weight list aligned to the kept positions, or None when `weights` is
    None (plain-mean path). Keeps each kept book's weight matched to its row.
    """
    if weights is None:
        return [pos[b] for b in query if b in pos], None
    idx, w = [], []
    for book, weight in zip(query, weights):
        if book in pos:
            idx.append(pos[book])
            w.append(weight)
    return idx, w


def weighted_profile(matrix, idx, weights=None):
    """(Weighted) mean of `matrix` rows at positions `idx` -> a 1-D profile vector.

    `matrix` may be dense ndarray or sparse; `weights` (aligned to `idx`) switches the plain
    mean for a recency/importance-weighted mean. Returns a dense 1-D ndarray either way.
    """
    rows = matrix[idx]
    if weights is None:
        return np.asarray(rows.mean(axis=0)).ravel()
    w = np.asarray(weights, dtype=float)
    return np.asarray(rows.T @ w).ravel() / w.sum()
