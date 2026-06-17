"""In-memory, ephemeral swipe sessions (no DB; lost on restart)."""
import uuid
from dataclasses import dataclass, field

_ACTIONS = {"like", "want", "dislike", "skip"}


@dataclass
class Session:
    lam: float = 1.0
    k: int = 10
    liked: list = field(default_factory=list)
    disliked: list = field(default_factory=list)
    reading_list: list = field(default_factory=list)
    seen: set = field(default_factory=set)


class SessionStore:
    """Maps session_id -> Session and applies swipe actions to the stored state."""

    def __init__(self) -> None:
        self._sessions: dict = {}

    def create(self, liked, lam: float = 1.0, k: int = 10) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = Session(lam=lam, k=k, liked=list(liked), seen=set(liked))
        return sid

    def get(self, session_id: str) -> Session:
        return self._sessions[session_id]   # KeyError if unknown

    def apply(self, session_id: str, book_id, action: str) -> Session:
        if action not in _ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        session = self._sessions[session_id]   # KeyError if unknown
        session.seen.add(book_id)
        if action in ("like", "want"):
            session.liked.append(book_id)
        if action == "want":
            session.reading_list.append(book_id)
        if action == "dislike":
            session.disliked.append(book_id)
        return session
