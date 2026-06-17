from fastapi.testclient import TestClient

from book_recsys.api.app import create_app
from book_recsys.api.sessions import SessionStore


class FakeRecService:

    def search(self, q, limit=20):
        return ["a", "b"][:limit]

    def label(self, book_id):
        return f"Title-{book_id}"

    def card(self, book_id):
        return {"book_id": book_id, "title": f"Title-{book_id}", "author": "Author",
                "description": f"Synopsis of {book_id}", "image_url": f"http://img/{book_id}"}


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
    # feed cards are rich: full synopsis + cover, not the short label
    assert body["cards"][0] == {"book_id": "x", "title": "Title-x", "author": "Author",
                                "description": "Synopsis of x", "image_url": "http://img/x"}

    r2 = c.post("/swipe", json={"session_id": sid, "book_id": "x", "action": "want"})
    body2 = r2.json()
    # the reading list keeps the compact label
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


class FakeOverview:
    def generate(self, message, history=None, history_titles=None):
        return {"intro": "ov", "categories": [
            {"header": "Top picks", "items": [{"book_id": "x", "reason": "great fit"}]}]}


def chat_client():
    return TestClient(create_app(FakeRecService(), FakeFeed(), SessionStore(),
                                 overview=FakeOverview()))


def test_chat_returns_grounded_overview_with_cards():
    r = chat_client().post("/chat", json={"message": "war books"})
    body = r.json()
    assert body["intro"] == "ov"
    item = body["categories"][0]["items"][0]
    assert item["book_id"] == "x" and item["reason"] == "great fit"
    assert item["image_url"] == "http://img/x"   # enriched to a real card


def test_chat_503_when_llm_unconfigured():
    r = make_client().post("/chat", json={"message": "hi"})   # overview defaults to None
    assert r.status_code == 503


def test_chat_blends_session_history():
    store = SessionStore()
    c = TestClient(create_app(FakeRecService(), FakeFeed(), store, overview=FakeOverview()))
    sid = c.post("/session", json={"liked": ["a"], "k": 3}).json()["session_id"]
    r = c.post("/chat", json={"message": "darker", "session_id": sid, "use_history": True})
    assert r.status_code == 200


def test_chat_unknown_session_is_ignored():
    r = chat_client().post("/chat",
                           json={"message": "x", "session_id": "nope", "use_history": True})
    assert r.status_code == 200
