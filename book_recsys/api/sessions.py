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
    messages: list = field(default_factory=list)


class SessionStore:
    """Maps session_id -> Session and applies swipe actions to the stored state."""

    def __init__(self) -> None:
        self._sessions: dict = {}

    def create(self, liked, lam: float = 1.0, k: int = 10) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = Session(lam=lam, k=k, liked=list(liked), seen=set(liked))
        return sid

    def get(self, session_id: str) -> Session:
        return self._sessions[session_id]  # KeyError if unknown

    def ensure(self, session_id: str | None) -> str:
        """Return the session_id, creating a fresh session if unknown or None."""
        if session_id is None or session_id not in self._sessions:
            return self.create([])
        return session_id

    def append_message(self, session_id: str, role: str, text: str) -> None:
        """Append a message to the session's message list."""
        session = self._sessions[session_id]
        session.messages.append({"role": role, "text": text})

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
