"""Reproduce SASRec's 30k-user sample and build the user×item training matrix."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from book_recsys.data.schema import BOOK, TS, USER


def reproduce_sasrec_sample(sample_df: pd.DataFrame,
                            n_users: int = 30000,
                            max_hist: int = 100,
                            seed: int = 42,
                            expect_rows: int | None = None) -> pd.DataFrame:
    """Subsample to `n_users` users (pandas RNG, matching 06_recbole.ipynb) and cap each
    user's history to the most-recent `max_hist` interactions by timestamp. If `expect_rows`
    is given, raise unless the row count matches — proves bit-identity with SASRec's set.
    """
    keep = sample_df[USER].drop_duplicates().sample(n_users, random_state=seed)
    out = sample_df[sample_df[USER].isin(keep)]
    out = (out.sort_values([USER, TS]).groupby(USER,
                                               sort=False).tail(max_hist).reset_index(drop=True))
    if expect_rows is not None and len(out) != expect_rows:
        raise ValueError(f"expected {expect_rows} interactions, got {len(out)}")
    return out


def build_matrix(train_df: pd.DataFrame,
                 min_item_count: int = 1) -> tuple[sp.csr_matrix, list, dict, np.ndarray]:
    """User×item binary CSR matrix + item vocab. Items with < `min_item_count` total
    interactions are dropped. Returns (matrix, ids, pos, counts): ids[j] is the book at
    column j, pos[book]=j, counts[j] is that item's interaction count (aligned to ids).
    """
    counts_s = train_df[BOOK].value_counts()
    kept = counts_s[counts_s >= min_item_count]
    ids = list(kept.index)
    pos = {b: j for j, b in enumerate(ids)}
    df = train_df[train_df[BOOK].isin(pos)]
    users = df[USER].astype("category")
    rows = users.cat.codes.to_numpy()
    cols = df[BOOK].map(pos).to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(users.cat.categories.size, len(ids)))
    matrix.data[:] = 1.0  # collapse any summed duplicates back to binary
    return matrix, ids, pos, kept.to_numpy().astype(np.float64)
