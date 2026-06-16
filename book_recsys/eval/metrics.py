"""Top-K ranking metrics for a single recommendation list (binary relevance)."""
import math
from collections.abc import Iterable, Sequence


def recall_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    relevant = set(relevant)
    dcg = 0.0
    for pos, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(pos + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended: Sequence, relevant: Iterable) -> float:
    relevant = set(relevant)
    for pos, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (pos + 1)
    return 0.0
