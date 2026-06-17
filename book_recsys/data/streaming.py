"""Fast parse-once preprocessing of the interaction files."""
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from tqdm.auto import tqdm

from book_recsys.data.kcore import k_core_filter
from book_recsys.data.sample import sample_users
from book_recsys.data.schema import BOOK, RATING, TS, USER

_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _to_unix_seconds(series) -> np.ndarray:
    dt = pd.to_datetime(series, format=_DATE_FORMAT, errors="coerce", utc=True)
    secs = (dt - _EPOCH) // pd.Timedelta(seconds=1)
    return secs.fillna(0).astype("int64").to_numpy()


def load_interactions(paths, chunksize: int = 1_000_000,
                      verbose: bool = False) -> pd.DataFrame:
    """Parse interaction files ONCE into a compact frame: user_id/book_id as category
    dtype (codes, not duplicated strings), rating int16, timestamp int64 seconds.

    Vectorized via pd.read_json; memory scales with codes + unique ids, not raw text.
    """
    users, books, ratings, times = [], [], [], []
    for path in tqdm(list(paths), desc="reading files", disable=not verbose):
        reader = pd.read_json(path, lines=True, compression="gzip", chunksize=chunksize)
        for chunk in reader:
            users.append(pd.Categorical(chunk["user_id"].astype(str)))
            books.append(pd.Categorical(chunk["book_id"].astype(str)))
            ratings.append(chunk["rating"].fillna(0).to_numpy(dtype="int16"))
            if "date_added" in chunk.columns:
                date = chunk["date_added"]
            else:
                date = pd.Series([None] * len(chunk))
            times.append(_to_unix_seconds(date))
    return pd.DataFrame({
        USER: union_categoricals(users),
        BOOK: union_categoricals(books),
        RATING: np.concatenate(ratings),
        TS: np.concatenate(times),
    })


def streaming_kcore_sample(paths, *, min_user: int, min_book: int, n_users: int,
                           seed: int, out_path, chunksize: int = 1_000_000,
                           verbose: bool = True) -> dict:
    """Parse-once k-core + seeded user sample, written to out_path parquet.

    k-core/sampling run on integer category codes (fast, low memory); the sampled rows
    are then mapped back to original ids. Returns {"n_users", "n_books"}.
    """
    df = load_interactions(paths, chunksize=chunksize, verbose=verbose)
    user_cats = df[USER].cat.categories.to_numpy()
    book_cats = df[BOOK].cat.categories.to_numpy()
    codes = pd.DataFrame({
        USER: df[USER].cat.codes.to_numpy(),
        BOOK: df[BOOK].cat.codes.to_numpy(),
        RATING: df[RATING].to_numpy(),
        TS: df[TS].to_numpy(),
    })
    del df
    core = k_core_filter(codes, min_user, min_book)
    sample = sample_users(core, n_users, seed)
    sample[USER] = user_cats[sample[USER].to_numpy()]
    sample[BOOK] = book_cats[sample[BOOK].to_numpy()]
    sample.to_parquet(str(out_path))
    return {"n_users": int(sample[USER].nunique()), "n_books": int(sample[BOOK].nunique())}
