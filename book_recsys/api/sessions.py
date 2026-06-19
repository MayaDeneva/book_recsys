"""In-memory, ephemeral swipe sessions (no DB; lost on restart)."""
import uuid
from dataclasses import dataclass, field

from book_recsys.llm.steer import SteeringState

_ACTIONS = {"like", "want", "dislike", "skip"}


@dataclass
class Session:
    lam: float = 1.0
    k: int = 10
    method: str = ""  # which recommender drives the feed (UI toggle); "" -> FeedService default
    liked: list = field(default_factory=list)
    disliked: list = field(default_factory=list)
    reading_list: list = field(default_factory=list)
    seen: set = field(default_factory=set)
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
