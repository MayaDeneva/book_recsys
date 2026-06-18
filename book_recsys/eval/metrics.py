"""Top-K ranking metrics for a single recommendation list (binary relevance)."""
import math
from collections.abc import Iterable, Mapping, Sequence


def _cosine(u, v) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return dot / (nu * nv) if nu > 0 and nv > 0 else 0.0


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


def intra_list_diversity(recommended: Sequence, vectors: Mapping, k: int) -> float:
    """Mean pairwise cosine *dissimilarity* (1 - cosine) over the top-k recommended items.

    `vectors` maps item id -> embedding (1-D sequence). Items in `recommended[:k]` without a
    vector are skipped. Fewer than two usable vectors -> 0.0 (no pairs to average). Higher =
    a more varied top-K (a beyond-accuracy quality, orthogonal to relevance).
    """
    vecs = [vectors[item] for item in recommended[:k] if item in vectors]
    if len(vecs) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            total += 1.0 - _cosine(vecs[i], vecs[j])
            pairs += 1
    return total / pairs


def serendipity_at_k(recommended: Sequence, relevant: Iterable, popularity: Mapping,
                     k: int) -> float:
    """Popularity-discounted relevance (Vargas & Castells, 2011).

    `popularity` maps item id -> popularity in (0, 1] (1 = most popular; e.g. the popularity
    percentile from `popularity_diagnostics`). A hit's unexpectedness is -log2(popularity), so
    a relevant *niche* book scores high and a relevant blockbuster ~0. Returns the mean over
    the top-k recs of 1[item in relevant] * unexpectedness. Items absent from `popularity` are
    treated as most-popular (unexpectedness 0) so unknown items never inflate the score.
    """
    relevant = set(relevant)
    top = recommended[:k]
    if not top:
        return 0.0
    total = 0.0
    for item in top:
        if item in relevant:
            total += -math.log2(popularity.get(item, 1.0))
    return total / len(top)
