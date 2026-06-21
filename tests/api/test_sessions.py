import pytest

from book_recsys.api.sessions import ProfileStore, SessionStore
from book_recsys.llm.steer import SteeringState


def test_profile_store_in_memory_save_get_names():
    ps = ProfileStore()
    assert ps.names() == [] and ps.get("maya") == []
    assert ps.save("maya", ["a", "b"]) == ["a", "b"]
    assert ps.names() == ["maya"]
    assert ps.get("maya") == ["a", "b"]


def test_profile_store_persists_to_file(tmp_path):
    path = str(tmp_path / "profiles.json")
    ProfileStore(path).save("maya", ["a", "b"])
    reloaded = ProfileStore(path)  # a fresh store backed by the same file loads it
    assert reloaded.names() == ["maya"] and reloaded.get("maya") == ["a", "b"]


def test_create_seeds_liked_and_seen():
    st = SessionStore()
    sid = st.create(["a", "b"], lam=0.5, k=5)
    s = st.get(sid)
    assert s.liked == ["a", "b"]
    assert s.seen == {"a", "b"}  # seeds are not re-recommended
    assert s.lam == 0.5 and s.k == 5
    assert s.disliked == [] and s.reading_list == []
    assert s.method == ""  # default: FeedService picks its default recommender


def test_create_stores_chosen_method():
    st = SessionStore()
    s = st.get(st.create(["a"], method="maxsim"))
    assert s.method == "maxsim"


def test_apply_actions_update_the_right_sets():
    st = SessionStore()
    sid = st.create(["a"])
    st.apply(sid, "c", "like")
    st.apply(sid, "d", "want")
    st.apply(sid, "e", "dislike")
    st.apply(sid, "f", "skip")
    s = st.get(sid)
    assert s.liked == ["a", "c", "d"]  # like + want
    assert s.reading_list == ["d"]  # want only
    assert s.disliked == ["e"]
    assert {"a", "c", "d", "e", "f"} == s.seen  # every swipe marks seen


def test_unknown_action_raises_valueerror():
    st = SessionStore()
    sid = st.create([])
    with pytest.raises(ValueError):
        st.apply(sid, "x", "love")


def test_unknown_session_raises_keyerror():
    st = SessionStore()
    with pytest.raises(KeyError):
        st.get("nope")
    with pytest.raises(KeyError):
        st.apply("nope", "x", "like")


def test_new_session_has_default_steering_and_empty_messages():
    store = SessionStore()
    sid = store.create([])
    s = store.get(sid)
    assert s.steering == SteeringState()
    assert s.messages == []


def test_ensure_returns_same_id_when_known():
    store = SessionStore()
    sid = store.create([])
    assert store.ensure(sid) == sid


def test_ensure_creates_session_when_unknown_or_none():
    store = SessionStore()
    sid = store.ensure(None)
    assert store.get(sid).liked == []
    sid2 = store.ensure("nope")
    assert sid2 != "nope"
    assert store.get(sid2).liked == []  # unknown id -> a fresh, retrievable session


def test_append_message_and_set_steering():
    store = SessionStore()
    sid = store.create([])
    store.append_message(sid, "user", "hi")
    store.set_steering(sid, SteeringState(topic="WWII"))
    s = store.get(sid)
    assert s.messages == [{"role": "user", "text": "hi"}]
    assert s.steering.topic == "WWII"


def test_sessions_have_independent_message_lists():
    store = SessionStore()
    s1 = store.create([])
    s2 = store.create([])
    store.append_message(s1, "user", "hello")
    assert store.get(s1).messages == [{"role": "user", "text": "hello"}]
    assert store.get(s2).messages == []  # default_factory -> no shared list


def test_apply_records_event_weights_like_over_want():
    from book_recsys.api.sessions import WANT_WEIGHT
    st = SessionStore()
    sid = st.create([])
    st.apply(sid, "b1", "like")
    st.apply(sid, "b2", "want")
    s = st.get(sid)
    assert s.weights == {"b1": 1.0, "b2": WANT_WEIGHT}  # like is a stronger positive than want
