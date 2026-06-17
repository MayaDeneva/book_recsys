"""Write interaction frames to RecBole '.inter' atomic files."""
import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER

_RECBOLE_COLUMNS = {
    USER: "user_id:token",
    BOOK: "item_id:token",
    RATING: "rating:float",
    TS: "timestamp:float",
}


def write_inter_file(df: pd.DataFrame, path) -> None:
    """Write df (USER, BOOK, RATING, TS) as a tab-separated RecBole .inter file."""
    out = df[[USER, BOOK, RATING, TS]].rename(columns=_RECBOLE_COLUMNS)
    out.to_csv(path, sep="\t", index=False)
