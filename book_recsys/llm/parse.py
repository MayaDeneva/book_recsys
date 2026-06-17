"""Parse a natural-language book request into structured intent via an LLM."""
import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    mood: str | None = None
    audience: str | None = None
    avoid: list[str] = field(default_factory=list)


_PROMPT = (
    "Extract the reader's intent from this book request as a JSON object with keys "
    '"mood" (string or null), "audience" (string or null), and "avoid" (list of strings). '
    "Respond with only the JSON.\nRequest: {query}\nJSON:"
)


def parse_query(query: str, client) -> ParsedQuery:
    """Ask the LLM to extract mood/audience/avoid; tolerate non-JSON responses."""
    raw = client.complete(_PROMPT.format(query=query))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return ParsedQuery()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ParsedQuery()
    return ParsedQuery(
        mood=data.get("mood"),
        audience=data.get("audience"),
        avoid=list(data.get("avoid") or []),
    )
