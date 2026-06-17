"""LLM "AI-overview" recommendations: retrieve real catalog books, then have the LLM
write a grounded, categorized overview (intro + groups + per-book reason).

The LLM returns book *ids* (chosen only from the retrieved candidates) plus a one-line
reason, so the caller can render real cards (covers) — no hallucinated titles.
"""
import json
import re

from book_recsys.llm.fusion import reciprocal_rank_fusion


def build_overview_prompt(query, history_titles, candidate_ids, id_to_doc) -> str:
    lines = ["You are a book concierge. Recommend ONLY from the catalog listed below."]
    if query:
        lines.append(f'Reader request: "{query}"')
    if history_titles:
        lines.append("Books the reader already enjoys: " + ", ".join(history_titles[:15]))
    lines.append("Catalog (use the exact id; do not invent books):")
    for book_id in candidate_ids:
        lines.append(f"- id={book_id}: {id_to_doc.get(book_id, '')}")
    lines.append('Reply with ONLY a JSON object: {"intro": "<1-2 sentence overview of the topic '
                 'or taste>", "categories": [{"header": "<short group name>", "items": '
                 '[{"id": "<id from the list>", "reason": "<one short sentence>"}]}]}. '
                 "Use 1-3 categories, 2-4 books each; ids only from the list above.")
    return "\n".join(lines)


def parse_overview(raw: str, allowed_ids) -> dict:
    """Parse the LLM JSON into {intro, categories:[{header, items:[{book_id, reason}]}]}.

    Items whose id is not in `allowed_ids` are dropped (grounding); malformed output
    yields an empty overview. Empty categories are removed.
    """
    allowed = set(allowed_ids)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"intro": "", "categories": []}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"intro": "", "categories": []}
    categories = []
    for cat in obj.get("categories") or []:
        if not isinstance(cat, dict):
            continue
        items = [{
            "book_id": it["id"],
            "reason": str(it.get("reason", ""))
        } for it in (cat.get("items") or []) if isinstance(it, dict) and it.get("id") in allowed]
        if items:
            categories.append({"header": str(cat.get("header", "")), "items": items})
    return {"intro": str(obj.get("intro", "")), "categories": categories}


class OverviewGenerator:
    """Retrieve real catalog books, then LLM-write a grounded, categorized overview."""

    def __init__(self, retriever, id_to_doc, client, n: int = 40) -> None:
        self._retriever = retriever
        self._id_to_doc = id_to_doc
        self._client = client
        self._n = n

    def _candidates(self, query, history) -> list:
        lists = []
        if history:
            lists.append(self._retriever.by_history(history, self._n))
        if query:
            lists.append(self._retriever.by_text(query, self._n))
        if not lists:
            return []
        fused = reciprocal_rank_fusion(lists) if len(lists) > 1 else lists[0]
        return fused[:self._n]

    def generate(self, query, history=None, history_titles=None) -> dict:
        history = list(history or [])
        candidates = self._candidates(query, history)
        if not candidates:
            return {"intro": "", "categories": []}
        prompt = build_overview_prompt(query, history_titles or [], candidates, self._id_to_doc)
        return parse_overview(self._client.complete(prompt), candidates)
