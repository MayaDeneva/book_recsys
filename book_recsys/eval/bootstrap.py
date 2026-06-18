"""Bootstrap confidence intervals + paired significance for evaluation metrics.

Operates on the per-user metric scores (e.g. each user's ndcg@10), so it is cheap —
no model re-runs, just resampling numbers. Resampling users with replacement estimates
how much the mean metric would wobble under a different draw of users (non-parametric,
so no normality assumption — important since per-user recall is 0/1).
"""
import numpy as np


def bootstrap_ci(scores,
                 n_resamples: int = 1000,
                 alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean of `scores` and its (1-alpha) percentile-bootstrap CI -> (mean, lo, hi)."""
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores), size=(n_resamples, len(scores)))
    means = scores[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(scores.mean()), float(lo), float(hi)


def paired_bootstrap(scores_a,
                     scores_b,
                     n_resamples: int = 1000,
                     alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """Paired bootstrap of the per-user difference a - b (same users, aligned).

    Resamples users and averages (a - b) on each resample. Returns the mean difference,
    its CI, and `significant` (True iff the CI excludes 0 -> a reliably beats/loses to b).
    Pairing cancels easy/hard-user variation, so it is more sensitive than two separate CIs.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "significant": bool(lo > 0 or hi < 0)
    }
