"""Seeded user-level sampling."""
import numpy as np
import pandas as pd

from book_recsys.data.schema import USER


def sample_users(df: pd.DataFrame, n_users: int, seed: int) -> pd.DataFrame:
    """Return all rows for n_users users chosen uniformly at random.

    If n_users >= the number of users, the frame is returned unchanged.
    """
    users = df[USER].unique()
    if n_users >= len(users):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(users, size=n_users, replace=False)
    return df[df[USER].isin(chosen)].reset_index(drop=True)
