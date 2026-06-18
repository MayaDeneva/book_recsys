"""LLM-driven steering: the LLM reads the chat and emits the recsys's knobs.

`SteeringState` is the running, per-session memory. `parse_steering` turns one LLM
JSON reply into the next state (robust to malformed output); `build_steer_prompt`
builds the request; `Steerer` makes the single LLM call per turn.
"""
import json
import re
from dataclasses import dataclass, field, replace


@dataclass
class SteeringState:
    history_weight: float = 1.0  # 1 = purely "like my reads", 0 = purely the topic
    topic: str | None = None
    avoid: list = field(default_factory=list)
    genre: str | None = None  # hard include-filter; set only on explicit request
    anchor_book: str | None = None  # a named book -> similar.recommend
    reply: str = ""  # one-line narration for the chat (not persistent memory)


def _clean_str(value):
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def parse_steering(raw: str, prev: SteeringState) -> SteeringState:
    """Merge one LLM JSON reply onto `prev`. Absent key -> keep prev; present (incl.
    null) -> use parsed; whole-parse failure -> a copy of prev. See module/spec."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return replace(prev)
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return replace(prev)
    if not isinstance(obj, dict):
        return replace(prev)

    history_weight = prev.history_weight
    if "history_weight" in obj:
        try:
            history_weight = min(1.0, max(0.0, float(obj["history_weight"])))
        except (TypeError, ValueError):
            history_weight = prev.history_weight

    topic = _clean_str(obj["topic"]) if "topic" in obj else prev.topic
    genre = _clean_str(obj["genre"]) if "genre" in obj else prev.genre
    anchor_book = _clean_str(obj["anchor_book"]) if "anchor_book" in obj else prev.anchor_book

    if "avoid" in obj:
        raw_avoid = obj["avoid"] if isinstance(obj["avoid"], list) else []
        avoid = [s.strip() for s in raw_avoid if isinstance(s, str) and s.strip()]
    else:
        avoid = list(prev.avoid)

    reply = _clean_str(obj.get("reply")) or "" if "reply" in obj else ""

    return SteeringState(history_weight=history_weight, topic=topic, avoid=avoid,
                         genre=genre, anchor_book=anchor_book, reply=reply)
