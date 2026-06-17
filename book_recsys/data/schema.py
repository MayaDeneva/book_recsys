"""Normalized interaction schema shared across the pipeline."""
import pandas as pd

USER = "user_id"
BOOK = "book_id"
RATING = "rating"
TS = "timestamp"

REQUIRED_COLUMNS = (USER, BOOK, RATING, TS)


def validate_interactions(df: pd.DataFrame) -> None:
    """Raise ValueError if df does not carry the required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"interactions frame missing columns: {missing}")
