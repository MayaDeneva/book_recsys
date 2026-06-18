import pytest

from book_recsys.api.sessions import SessionStore


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


def test_ensure_returns_known_session_or_creates_new():
    store = SessionStore()
    sid1 = store.create([])
    assert store.ensure(sid1) == sid1  # known id -> same session
    sid2 = store.ensure("nope")
    assert sid2 != "nope"
    assert store.get(sid2).liked == []  # unknown id -> a fresh, retrievable session
    assert store.ensure(None) != "nope"  # None -> a fresh session


def test_sessions_have_independent_message_lists():
    store = SessionStore()
    s1 = store.create([])
    s2 = store.create([])
    store.append_message(s1, "user", "hello")
    assert store.get(s1).messages == [{"role": "user", "text": "hello"}]
    assert store.get(s2).messages == []  # default_factory -> no shared list
