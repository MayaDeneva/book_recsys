import pandas as pd

from book_recsys.data.filters import denoise_history_keep_targets, filter_min_rating
from book_recsys.data.schema import BOOK, RATING, TS, USER


def _df(ratings):
    return pd.DataFrame([{
        USER: "u",
        BOOK: f"b{i}",
        RATING: r,
        TS: 0
    } for i, r in enumerate(ratings)])


def _multi(rows):
    # rows: list of (user, rating, ts)
    return pd.DataFrame([{
        USER: u,
        BOOK: f"b{i}",
        RATING: r,
        TS: t
    } for i, (u, r, t) in enumerate(rows)])


def test_keeps_only_ratings_at_or_above_threshold():
    out = filter_min_rating(_df([0, 3, 4, 5]), min_rating=4)
    assert sorted(out[RATING]) == [4, 5]


def test_drops_implicit_zero_ratings():
    out = filter_min_rating(_df([0, 0, 5]), min_rating=1)
    assert list(out[RATING]) == [5]


def test_index_is_reset():
    out = filter_min_rating(_df([1, 5]), min_rating=5)
    assert list(out.index) == [0]


def test_denoise_drops_zero_rating_history_events():
    # history events b0 (r0) dropped; b1,b2 (rated) kept; last two (b3,b4) always kept
    out = denoise_history_keep_targets(_df([0, 3, 0, 4, 0]), min_rating=1, n_targets=2)
    assert list(out[RATING]) == [3, 4, 0]  # b1, b3, b4 — b0 & b2 dropped, b4 target kept


def test_denoise_preserves_last_n_targets_even_if_zero_rating():
    # the final two events are rating 0 but are the valid/test targets -> kept
    out = denoise_history_keep_targets(_df([5, 5, 5, 0, 0]), min_rating=1, n_targets=2)
    assert list(out[RATING]) == [5, 5, 5, 0, 0]


def test_denoise_drops_users_below_min_items():
    # u_short: only 1 rated history event + 2 targets would be 3, but here all-zero history
    # leaves just the 2 targets (<3) -> user dropped. u_ok keeps >=3.
    df = _multi([("short", 0, 0), ("short", 0, 1), ("short", 0, 2), ("ok", 4, 0), ("ok", 0, 1),
                 ("ok", 0, 2)])
    out = denoise_history_keep_targets(df, min_rating=1, n_targets=2, min_items=3)
    assert set(out[USER]) == {"ok"}


def test_denoise_keeps_multiple_same_day_ratings():
    # two rated events share a timestamp (same day) -> both kept, order preserved
    out = denoise_history_keep_targets(_df([4, 4, 5, 5]), min_rating=1, n_targets=2)
    assert list(out[RATING]) == [4, 4, 5, 5]


def test_denoise_does_not_mutate_input():
    df = _df([0, 3, 4, 5])
    before = df.copy()
    denoise_history_keep_targets(df, min_rating=1)
    pd.testing.assert_frame_equal(df, before)


def test_denoise_resets_index():
    out = denoise_history_keep_targets(_df([0, 4, 5, 5]), min_rating=1)
    assert list(out.index) == list(range(len(out)))
