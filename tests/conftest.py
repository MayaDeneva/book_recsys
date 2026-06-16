import pandas as pd
import pytest

from book_recsys.data.schema import BOOK, RATING, TS, USER


@pytest.fixture
def tiny_interactions():
    """10 users with rated, timestamped interactions over a small catalog."""
    data = {
        "u0": [("b0", 5, 100), ("b1", 4, 200), ("b2", 3, 300)],
        "u1": [("b0", 5, 110), ("b1", 5, 210), ("b3", 4, 310)],
        "u2": [("b1", 4, 120), ("b2", 5, 220), ("b4", 3, 320)],
        "u3": [("b0", 3, 130), ("b2", 4, 230), ("b5", 5, 330)],
        "u4": [("b1", 5, 140), ("b3", 4, 240), ("b6", 3, 340)],
        "u5": [("b0", 4, 150), ("b4", 5, 250), ("b7", 4, 350)],
        "u6": [("b2", 5, 160), ("b5", 4, 260), ("b8", 3, 360)],
        "u7": [("b1", 4, 170), ("b6", 5, 270), ("b9", 4, 370)],
        "u8": [("b0", 5, 180), ("b3", 3, 280), ("b10", 4, 380)],
        "u9": [("b2", 4, 190), ("b7", 5, 290), ("b11", 3, 390)],
    }
    rows = []
    for user, items in data.items():
        for book, rating, ts in items:
            rows.append({USER: user, BOOK: book, RATING: rating, TS: ts})
    return pd.DataFrame(rows)
