"""Train/val/test splitting strategies."""
import pandas as pd

from book_recsys.data.schema import TS, USER


def global_time_split(df: pd.DataFrame, train_frac: float = 0.8,
                      val_frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by global wall-clock time using timestamp quantiles."""
    t1 = df[TS].quantile(train_frac)
    t2 = df[TS].quantile(train_frac + val_frac)
    train = df[df[TS] <= t1].reset_index(drop=True)
    val = df[(df[TS] > t1) & (df[TS] <= t2)].reset_index(drop=True)
    test = df[df[TS] > t2].reset_index(drop=True)
    return train, val, test


def leave_last_n_out(df: pd.DataFrame,
                     n: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's latest n interactions (by timestamp) as the test set."""
    ordered = df.sort_values([USER, TS])
    holdout = ordered.groupby(USER).tail(n)
    train = ordered.drop(holdout.index)
    return train.reset_index(drop=True), holdout.reset_index(drop=True)
