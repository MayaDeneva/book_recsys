import pandas as pd
import pytest

from book_recsys.data.schema import BOOK, RATING, TS, USER, validate_interactions


def test_column_constants():
    assert (USER, BOOK, RATING, TS) == ("user_id", "book_id", "rating", "timestamp")


def test_validate_passes_on_good_frame():
    df = pd.DataFrame({USER: ["u0"], BOOK: ["b0"], RATING: [5], TS: [100]})
    validate_interactions(df)  # should not raise


def test_validate_raises_on_missing_column():
    df = pd.DataFrame({USER: ["u0"], BOOK: ["b0"]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_interactions(df)
