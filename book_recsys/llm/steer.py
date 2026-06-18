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


def _clean_str(value: object) -> str | None:
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

    reply = (_clean_str(obj["reply"]) or "") if "reply" in obj else ""

    return SteeringState(history_weight=history_weight,
                         topic=topic,
                         avoid=avoid,
                         genre=genre,
                         anchor_book=anchor_book,
                         reply=reply)


def build_steer_prompt(messages, prev: SteeringState, anchor_titles) -> str:
    """Build the LLM prompt for steering: embed prior state, recent messages, and rules."""
    lines = [
        "You steer a book recommender by choosing its settings. Read the conversation "
        "and return the UPDATED settings as JSON.",
        "",
        "Current settings (carry forward unless the conversation changes them):",
        f"- history_weight: {prev.history_weight}  (1.0 = recommend books like the "
        "reader's past reads; 0.0 = ignore past reads, follow the topic instead)",
        f"- topic: {prev.topic!r}  (the theme/subject to retrieve by; null if none yet)",
        f"- avoid: {prev.avoid}  (themes to steer away from)",
        f"- genre: {prev.genre!r}",
        f"- anchor_book: {prev.anchor_book!r}",
    ]
    if anchor_titles:
        lines.append("")
        lines.append("The reader's past reads: " + ", ".join(anchor_titles[:15]))
    lines.append("")
    lines.append("Conversation so far:")
    for msg in messages:
        lines.append(f"{msg['role']}: {msg['text']}")
    lines += [
        "",
        "Rules:",
        "- Set genre to null UNLESS the reader explicitly names a genre to restrict to.",
        "- If the request is a GIFT or for someone else, set history_weight near 0 (the "
        "past reads are the asker's, not the recipient's) and build topic from the "
        "recipient's described tastes; if a book the recipient loved is named, set "
        "anchor_book to it.",
        "- To clear a setting, return it as null (omitting a key keeps its current value).",
        "",
        'Reply with ONLY a JSON object: {"history_weight": <0..1>, "topic": <string|null>, '
        '"avoid": [<string>...], "genre": <string|null>, "anchor_book": <string|null>, '
        '"reply": "<one short sentence telling the reader what you changed>"}.',
    ]
    return "\n".join(lines)


class Steerer:
    """Single LLM call per steering turn: prompt + parse."""

    def __init__(self, client) -> None:
        self._client = client

    def update(self, messages, prev: SteeringState, anchor_titles) -> SteeringState:
        """Build prompt from state, call LLM, parse reply onto prev state."""
        raw = self._client.complete(build_steer_prompt(messages, prev, anchor_titles))
        return parse_steering(raw, prev)
