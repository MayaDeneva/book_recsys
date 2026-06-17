"""Interaction-level filters for explicit-feedback experiments."""
import pandas as pd

from book_recsys.data.schema import RATING


def filter_min_rating(df: pd.DataFrame, min_rating: int) -> pd.DataFrame:
    """Keep only interactions with rating >= min_rating (drops implicit 0s below it)."""
    return df[df[RATING] >= min_rating].reset_index(drop=True)
