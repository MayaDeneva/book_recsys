from fastapi.testclient import TestClient

from book_recsys.api.app import create_app
from book_recsys.api.sessions import SessionStore


class FakeRecService:

    def search(self, q, limit=20):
        return ["a", "b"][:limit]

    def label(self, book_id):
        return f"Title-{book_id}"


class FakeFeed:
    """Returns x,y,z minus anything already seen/disliked (ignores scores)."""

    def next(self, liked, disliked, seen, k=10, lam=1.0):
        pool = [b for b in ["x", "y", "z"] if b not in set(seen) | set(disliked)]
        return pool[:k]


def make_client():
    return TestClient(create_app(FakeRecService(), FakeFeed(), SessionStore()))


def test_search_returns_labeled_books():
    r = make_client().get("/search", params={"q": "foo"})
    assert r.status_code == 200
    assert r.json() == [{"book_id": "a", "label": "Title-a"},
                        {"book_id": "b", "label": "Title-b"}]


def test_session_then_swipe_adapts_and_collects_reading_list():
    c = make_client()
    r = c.post("/session", json={"liked": ["a"], "lam": 1.0, "k": 10})
    body = r.json()
    sid = body["session_id"]
    assert body["cards"][0]["book_id"] == "x"

    r2 = c.post("/swipe", json={"session_id": sid, "book_id": "x", "action": "want"})
    body2 = r2.json()
    assert body2["reading_list"] == [{"book_id": "x", "label": "Title-x"}]
    assert body2["cards"][0]["book_id"] == "y"  # x now seen -> next card is y


def test_swipe_unknown_session_returns_404():
    r = make_client().post("/swipe",
                           json={"session_id": "nope", "book_id": "x", "action": "like"})
    assert r.status_code == 404


def test_swipe_bad_action_returns_400():
    c = make_client()
    sid = c.post("/session", json={"liked": []}).json()["session_id"]
    r = c.post("/swipe", json={"session_id": sid, "book_id": "x", "action": "love"})
    assert r.status_code == 400
