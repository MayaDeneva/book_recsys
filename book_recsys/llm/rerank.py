"""LLM reranking of retrieved candidates via packed, JSON-scored batches."""
import json
import re


def _build_prompt(context: str, batch, id_to_doc) -> str:
    lines = [
        "Score each book from 0-10 for how well it fits the request below.",
        "Respond with only a JSON list of {\"id\": <id>, \"score\": <0-10>}.",
        f"Request/context:\n{context}",
        "Books:",
    ]
    for book_id in batch:
        lines.append(f"- id={book_id}: {id_to_doc.get(book_id, '')}")
    return "\n".join(lines)


def _parse_scores(raw: str, batch) -> dict:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return {}
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    allowed = set(batch)
    scores = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        book_id = item.get("id")
        if book_id not in allowed:
            continue
        try:
            scores[book_id] = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            scores[book_id] = 0.0
    return scores


def rerank(context: str, candidate_ids, id_to_doc, client, k: int,
           batch_size: int = 10) -> list:
    """Return the top-k candidates ordered by LLM score (desc), ties by input order."""
    position = {book_id: i for i, book_id in enumerate(candidate_ids)}
    scores: dict = {}
    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start:start + batch_size]
        raw = client.complete(_build_prompt(context, batch, id_to_doc))
        scores.update(_parse_scores(raw, batch))
    ordered = sorted(candidate_ids, key=lambda b: (-scores.get(b, 0.0), position[b]))
    return ordered[:k]
