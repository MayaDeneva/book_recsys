"""Paradigm-3 recommender: retrieve-then-rerank with an LLM."""
from book_recsys.llm.fusion import reciprocal_rank_fusion
from book_recsys.llm.rerank import rerank


class LLMRecommender:
    """Retrieve candidates from the embedding catalog, then LLM-rerank them.

    Query shapes:
      - list of book ids   -> history only (UC1)
      - str                -> NL request only (UC5)
      - {"history","query"}-> both, fused with RRF (UC3)
    """

    def __init__(self, retriever, id_to_doc, client, retrieve_n: int = 200,
                 rerank_batch: int = 10) -> None:
        self._retriever = retriever
        self._id_to_doc = id_to_doc
        self._client = client
        self._retrieve_n = retrieve_n
        self._rerank_batch = rerank_batch

    def fit(self, train_data=None) -> "LLMRecommender":
        return self

    def _split_query(self, query):
        if isinstance(query, str):
            return [], query
        if isinstance(query, dict):
            return list(query.get("history", [])), query.get("query")
        return list(query), None

    def _context(self, history, intent) -> str:
        parts = []
        if history:
            liked = ", ".join(self._id_to_doc.get(b, str(b)) for b in history[:20])
            parts.append(f"Books the reader liked: {liked}")
        if intent:
            parts.append(f"Request: {intent}")
        return "\n".join(parts)

    def recommend(self, query, k: int) -> list:
        history, intent = self._split_query(query)
        lists = []
        if history:
            lists.append(self._retriever.by_history(history, self._retrieve_n))
        if intent:
            lists.append(self._retriever.by_text(intent, self._retrieve_n))
        if not lists:
            return []
        candidates = reciprocal_rank_fusion(lists) if len(lists) > 1 else lists[0]
        ranked = rerank(self._context(history, intent), candidates, self._id_to_doc,
                        self._client, k + len(history), self._rerank_batch)
        seen = set(history)
        return [b for b in ranked if b not in seen][:k]
