import pytest

from book_recsys.api.sessions import SessionStore
from book_recsys.llm.steer import SteeringState


def test_create_seeds_liked_and_seen():
    st = SessionStore()
    sid = st.create(["a", "b"], lam=0.5, k=5)
    s = st.get(sid)
    assert s.liked == ["a", "b"]
    assert s.seen == {"a", "b"}      # seeds are not re-recommended
    assert s.lam == 0.5 and s.k == 5
    assert s.disliked == [] and s.reading_list == []


def test_apply_actions_update_the_right_sets():
    st = SessionStore()
    sid = st.create(["a"])
    st.apply(sid, "c", "like")
    st.apply(sid, "d", "want")
    st.apply(sid, "e", "dislike")
    st.apply(sid, "f", "skip")
    s = st.get(sid)
    assert s.liked == ["a", "c", "d"]          # like + want
    assert s.reading_list == ["d"]             # want only
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
    assert store.ensure("nope") != "nope"  # unknown id -> a fresh session


def test_append_message_and_set_steering():
    store = SessionStore()
    sid = store.create([])
    store.append_message(sid, "user", "hi")
    store.set_steering(sid, SteeringState(topic="WWII"))
    s = store.get(sid)
    assert s.messages == [{"role": "user", "text": "hi"}]
    assert s.steering.topic == "WWII"
