"""In-memory, ephemeral swipe sessions (no DB; lost on restart)."""
import json
import os
import uuid
from dataclasses import dataclass, field

from book_recsys.llm.steer import SteeringState


class ProfileStore:
    """Named seed-user profiles: name -> list of liked book_ids. Optionally persisted to a JSON
    file so the "log in as" list survives restarts — lets you save a set of picks once and reload
    it instead of re-seeding the feed every time. Not auth; just named seeds.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._profiles: dict = {}
        if path and os.path.exists(path):
            with open(path) as handle:
                self._profiles = {k: list(v) for k, v in json.load(handle).items()}

    def names(self) -> list:
        return list(self._profiles)

    def get(self, name: str) -> list:
        """The saved liked book_ids for `name` ([] if unknown)."""
        return list(self._profiles.get(name, []))

    def save(self, name: str, liked) -> list:
        """Store `liked` under `name` (overwrites), persisting to disk when file-backed."""
        self._profiles[name] = list(liked)
        if self._path:
            with open(self._path, "w") as handle:
                json.dump(self._profiles, handle)
        return self._profiles[name]


_ACTIONS = {"like", "want", "dislike", "skip"}
WANT_WEIGHT = 0.4  # a 🔖 "want to read" is a weaker positive than a ♥ "read & liked" (event-level)


@dataclass
class Session:
    lam: float = 1.0
    k: int = 10
    method: str = ""  # which recommender drives the feed (UI toggle); "" -> FeedService default
    liked: list = field(default_factory=list)
    disliked: list = field(default_factory=list)
    reading_list: list = field(default_factory=list)
    seen: set = field(default_factory=set)
    weights: dict = field(default_factory=dict)  # book_id -> event weight (like=1.0, want<1.0)
    steering: SteeringState = field(default_factory=SteeringState)
    messages: list = field(default_factory=list)


class SessionStore:
    """Maps session_id -> Session and applies swipe actions to the stored state."""

    def __init__(self) -> None:
        self._sessions: dict = {}

    def create(self, liked, lam: float = 1.0, k: int = 10, method: str = "") -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = Session(lam=lam,
                                      k=k,
                                      method=method,
                                      liked=list(liked),
                                      seen=set(liked))
        return sid

    def get(self, session_id: str) -> Session:
        return self._sessions[session_id]  # KeyError if unknown

    def apply(self, session_id: str, book_id, action: str) -> Session:
        if action not in _ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        session = self._sessions[session_id]  # KeyError if unknown
        session.seen.add(book_id)
        if action in ("like", "want"):
            session.liked.append(book_id)
            session.weights[book_id] = 1.0 if action == "like" else WANT_WEIGHT
        if action == "want":
            session.reading_list.append(book_id)
        if action == "dislike":
            session.disliked.append(book_id)
        return session

    def ensure(self, session_id: str | None) -> str:
        if session_id is not None and session_id in self._sessions:
            return session_id
        return self.create([])

    def append_message(self, session_id: str, role: str, text: str) -> None:
        self._sessions[session_id].messages.append({"role": role, "text": text})

    def set_steering(self, session_id: str, state: SteeringState) -> None:
        self._sessions[session_id].steering = state
