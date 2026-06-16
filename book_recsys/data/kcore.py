"""Iterative k-core filtering of an interaction frame."""
import pandas as pd

from book_recsys.data.schema import BOOK, USER


def k_core_filter(df: pd.DataFrame, min_user: int, min_book: int) -> pd.DataFrame:
    """Keep only users with >= min_user interactions and books with >= min_book.

    Applied iteratively until the frame is stable, because removing rows on one
    axis can push counts on the other axis below threshold.
    """
    while True:
        before = len(df)
        user_counts = df[USER].value_counts()
        keep_users = user_counts[user_counts >= min_user].index
        df = df[df[USER].isin(keep_users)]
        book_counts = df[BOOK].value_counts()
        keep_books = book_counts[book_counts >= min_book].index
        df = df[df[BOOK].isin(keep_books)]
        if len(df) == before:
            break
    return df.reset_index(drop=True)
