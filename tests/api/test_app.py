from fastapi.testclient import TestClient

from book_recsys.api.app import create_app
from book_recsys.api.sessions import ProfileStore, SessionStore
from book_recsys.llm.steer import SteeringState


class FakeRecService:

    def search(self, q, limit=20):
        return ["a", "b"][:limit]

    def label(self, book_id):
        return f"Title-{book_id}"

    def card(self, book_id):
        return {
            "book_id": book_id,
            "title": f"Title-{book_id}",
            "author": "Author",
            "description": f"Synopsis of {book_id}",
            "image_url": f"http://img/{book_id}"
        }


class FakeFeed:
    """Returns x,y,z minus anything already seen/disliked (ignores scores)."""

    def __init__(self):
        self.last_method = "unset"
        self.last_weights = "unset"

    def methods(self):
        return ["m1", "m2"]

    def next(self, liked, disliked, seen, k=10, lam=1.0, method=None, weights=None):
        self.last_method = method
        self.last_weights = weights
        pool = [b for b in ["x", "y", "z"] if b not in set(seen) | set(disliked)]
        return pool[:k]


def make_client():
    return TestClient(create_app(FakeRecService(), FakeFeed(), SessionStore()))


def test_search_returns_labeled_books():
    r = make_client().get("/search", params={"q": "foo"})
    assert r.status_code == 200
    assert r.json() == [{"book_id": "a", "label": "Title-a"}, {"book_id": "b", "label": "Title-b"}]


def test_methods_endpoint_lists_recommenders():
    assert make_client().get("/methods").json() == ["m1", "m2"]


def test_session_passes_selected_method_to_feed():
    feed = FakeFeed()
    c = TestClient(create_app(FakeRecService(), feed, SessionStore()))
    c.post("/session", json={"liked": ["a"], "method": "m2"})
    assert feed.last_method == "m2"


def test_session_then_swipe_adapts_and_collects_reading_list():
    c = make_client()
    r = c.post("/session", json={"liked": ["a"], "lam": 1.0, "k": 10})
    body = r.json()
    sid = body["session_id"]
    # feed cards are rich: full synopsis + cover, not the short label
    assert body["cards"][0] == {
        "book_id": "x",
        "title": "Title-x",
        "author": "Author",
        "description": "Synopsis of x",
        "image_url": "http://img/x"
    }
    # the seeded reading history comes back as compact labels (drives the swipe sidebar)
    assert body["liked"] == [{"book_id": "a", "label": "Title-a"}]

    r2 = c.post("/swipe", json={"session_id": sid, "book_id": "x", "action": "want"})
    body2 = r2.json()
    # the reading list keeps the compact label
    assert body2["reading_list"] == [{"book_id": "x", "label": "Title-x"}]
    assert body2["cards"][0]["book_id"] == "y"  # x now seen -> next card is y


def test_swipe_unknown_session_returns_404():
    r = make_client().post("/swipe", json={"session_id": "nope", "book_id": "x", "action": "like"})
    assert r.status_code == 404


def test_swipe_bad_action_returns_400():
    c = make_client()
    sid = c.post("/session", json={"liked": []}).json()["session_id"]
    r = c.post("/swipe", json={"session_id": sid, "book_id": "x", "action": "love"})
    assert r.status_code == 400


class FakeOverview:

    def generate(self, message, history=None, history_titles=None):
        return {
            "intro": "ov",
            "categories": [{
                "header": "Top picks",
                "items": [{
                    "book_id": "x",
                    "reason": "great fit"
                }]
            }]
        }


def chat_client():
    return TestClient(
        create_app(FakeRecService(), FakeFeed(), SessionStore(), overview=FakeOverview()))


def test_chat_returns_grounded_overview_with_cards():
    r = chat_client().post("/chat", json={"message": "war books"})
    body = r.json()
    assert body["intro"] == "ov"
    item = body["categories"][0]["items"][0]
    assert item["book_id"] == "x" and item["reason"] == "great fit"
    assert item["image_url"] == "http://img/x"  # enriched to a real card


def test_chat_503_when_llm_unconfigured():
    r = make_client().post("/chat", json={"message": "hi"})  # overview defaults to None
    assert r.status_code == 503


def test_chat_blends_session_history():
    store = SessionStore()
    c = TestClient(create_app(FakeRecService(), FakeFeed(), store, overview=FakeOverview()))
    sid = c.post("/session", json={"liked": ["a"], "k": 3}).json()["session_id"]
    r = c.post("/chat", json={"message": "darker", "session_id": sid, "use_history": True})
    assert r.status_code == 200


def test_chat_unknown_session_is_ignored():
    r = chat_client().post("/chat",
                           json={
                               "message": "x",
                               "session_id": "nope",
                               "use_history": True
                           })
    assert r.status_code == 200


class _RaisingOverview:

    def generate(self, message, history=None, history_titles=None):
        raise RuntimeError("ollama down")


def test_chat_503_when_generation_fails():
    c = TestClient(
        create_app(FakeRecService(), FakeFeed(), SessionStore(), overview=_RaisingOverview()))
    r = c.post("/chat", json={"message": "x"})
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# /steer tests
# ---------------------------------------------------------------------------


class _RecSvc:

    def card(self, book_id):
        return {
            "book_id": book_id,
            "title": f"T{book_id}",
            "author": "",
            "description": "",
            "image_url": ""
        }

    def search(self, q, limit=20):
        return ["anchor1"]

    def label(self, book_id):
        return f"T{book_id}"


class _Steerer:

    def update(self, messages, prev, anchor_titles):
        return SteeringState(history_weight=0.5, topic="WWII", reply="Toward WWII.")


class _Ranker:

    def rank_with_reasons(self, state, history_ids, seen, k=10, anchor_id=None):
        return [("x1", "Matches your topic: WWII"), ("x2", "")]


def test_steer_returns_reply_state_and_cards():
    app = create_app(_RecSvc(), None, SessionStore(), steerer=_Steerer(), ranker=_Ranker())
    client = TestClient(app)
    resp = client.post("/steer", json={"message": "books about WWII"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Toward WWII."
    assert body["state"]["topic"] == "WWII"
    assert [c["book_id"] for c in body["cards"]] == ["x1", "x2"]
    assert body["session_id"]  # a session was created
    assert body["cards"][0]["reason"] == "Matches your topic: WWII"
    assert body["cards"][1]["reason"] == ""


def test_steer_503_when_not_configured():
    app = create_app(_RecSvc(), None, SessionStore())  # no steerer/ranker
    resp = TestClient(app).post("/steer", json={"message": "hi"})
    assert resp.status_code == 503


def test_steer_resolves_anchor_book_to_search_hit():

    class _AnchorSteerer:

        def update(self, messages, prev, anchor_titles):
            return SteeringState(anchor_book="Dune", reply="like Dune")

    class _RecordingRanker:

        def __init__(self):
            self.anchor_id = "UNSET"

        def rank_with_reasons(self, state, history_ids, seen, k=10, anchor_id=None):
            self.anchor_id = anchor_id
            return [("x1", "")]

    ranker = _RecordingRanker()
    app = create_app(_RecSvc(), None, SessionStore(), steerer=_AnchorSteerer(), ranker=ranker)
    resp = TestClient(app).post("/steer", json={"message": "more like Dune"})
    assert resp.status_code == 200
    assert ranker.anchor_id == "anchor1"  # _RecSvc.search(...) -> ["anchor1"]


def test_steer_anchor_book_with_no_search_hit_is_none():

    class _EmptySearchRec(_RecSvc):

        def search(self, q, limit=20):
            return []

    class _AnchorSteerer:

        def update(self, messages, prev, anchor_titles):
            return SteeringState(anchor_book="Nope", reply="x")

    class _RecordingRanker:

        def __init__(self):
            self.anchor_id = "UNSET"

        def rank_with_reasons(self, state, history_ids, seen, k=10, anchor_id=None):
            self.anchor_id = anchor_id
            return []

    ranker = _RecordingRanker()
    app = create_app(_EmptySearchRec(),
                     None,
                     SessionStore(),
                     steerer=_AnchorSteerer(),
                     ranker=ranker)
    resp = TestClient(app).post("/steer", json={"message": "more like Nope"})
    assert resp.status_code == 200
    assert ranker.anchor_id is None
    assert resp.json()["cards"] == []  # empty pairs -> empty cards


def test_steer_503_when_steerer_raises():

    class _BoomSteerer:

        def update(self, messages, prev, anchor_titles):
            raise RuntimeError("ollama down")

    app = create_app(_RecSvc(), None, SessionStore(), steerer=_BoomSteerer(), ranker=_Ranker())
    resp = TestClient(app).post("/steer", json={"message": "hi"})
    assert resp.status_code == 503  # LLM failure -> graceful 503


def test_users_save_then_seed_session():
    store, profiles = SessionStore(), ProfileStore()
    c = TestClient(create_app(FakeRecService(), FakeFeed(), store, profile_store=profiles))
    assert c.get("/users").json() == []
    sid = c.post("/session", json={"liked": ["x", "y"]}).json()["session_id"]
    assert c.post("/users/maya", json={
        "session_id": sid
    }).json() == {
        "name": "maya",
        "liked": ["x", "y"]
    }
    assert c.get("/users").json() == ["maya"]
    sid2 = c.post("/session", json={"user": "maya"}).json()["session_id"]  # seeded from profile
    assert store.get(sid2).liked == ["x", "y"]


def test_session_dedups_overlapping_picks_and_profile():
    store, profiles = SessionStore(), ProfileStore()
    profiles.save("maya", ["a", "b"])
    c = TestClient(create_app(FakeRecService(), FakeFeed(), store, profile_store=profiles))
    # picks ["a", "c"] overlap the profile ["a", "b"] on "a" -> "a" must appear once
    body = c.post("/session", json={"liked": ["a", "c"], "user": "maya"}).json()
    sid = body["session_id"]
    assert store.get(sid).liked == ["a", "c", "b"]
    assert [card["book_id"] for card in body["liked"]] == ["a", "c", "b"]


def test_save_user_unknown_session_is_404():
    c = TestClient(create_app(FakeRecService(), FakeFeed(),
                              SessionStore()))  # default profile store
    assert c.post("/users/maya", json={"session_id": "nope"}).status_code == 404
