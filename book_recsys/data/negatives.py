"""Negative sampling for implicit-feedback training and evaluation.

Uniform (random) negatives treat any unobserved item as negative — a noisy assumption,
since the user might have liked a book they simply never saw. Popularity-weighted
negatives sample popular unread books more often: a popular book the user *didn't* read
is a more confident negative (they were likely exposed to it and still passed), and it
removes the popularity inflation uniform negatives give the popularity baseline
(Krichene & Rendle, 2020).
"""
import numpy as np


def build_cdf(weights) -> np.ndarray:
    """Cumulative distribution over a pool from per-item sampling weights (e.g. ∝ popularity)."""
    w = np.asarray(weights, dtype=float)
    return np.cumsum(w / w.sum())


def sample_negatives(pool, seen, n_neg, rng, cdf=None) -> list:
    """Draw `n_neg` items from `pool` that are not in `seen` (mutated to dedupe draws).

    Uniform when `cdf` is None, else popularity-weighted via inverse-CDF sampling.
    """
    n = len(pool)
    out: list = []
    while len(out) < n_neg:
        if cdf is None:
            i = int(rng.integers(n))
        else:
            i = min(int(np.searchsorted(cdf, rng.random())), n - 1)
        cand = pool[i]
        if cand not in seen:
            out.append(cand)
            seen.add(cand)
    return out
