"""Interaction-level filters for explicit-feedback experiments."""
import pandas as pd

from book_recsys.data.schema import RATING, TS, USER


def filter_min_rating(df: pd.DataFrame, min_rating: int) -> pd.DataFrame:
    """Keep only interactions with rating >= min_rating (drops implicit 0s below it)."""
    return df[df[RATING] >= min_rating].reset_index(drop=True)


def denoise_history_keep_targets(df: pd.DataFrame,
                                 min_rating: int = 1,
                                 n_targets: int = 2,
                                 min_items: int = 3) -> pd.DataFrame:
    """Drop low-rating events from each user's *history* while preserving their last
    `n_targets` interactions (the leave-one-out valid/test targets) regardless of rating.

    A row is kept iff its rating >= `min_rating` OR it is among the user's last `n_targets`
    interactions by timestamp. Users left with fewer than `min_items` rows are dropped
    entirely (too short for the leave-one-out split). The input is not mutated; per-user
    chronological order is preserved.
    """
    ordered = df.sort_values([USER, TS], kind="stable")
    from_end = ordered.groupby(USER, sort=False).cumcount(ascending=False)
    keep = (ordered[RATING] >= min_rating) | (from_end < n_targets)
    kept = ordered[keep]
    sizes = kept.groupby(USER, sort=False)[RATING].transform("size")
    return kept[sizes >= min_items].reset_index(drop=True)
