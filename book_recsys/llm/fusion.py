"""Reciprocal Rank Fusion of multiple ranked lists."""


def reciprocal_rank_fusion(ranked_lists, k: int = 60) -> list:
    """Fuse ranked lists into one. score(item) = sum 1/(k + rank + 1).

    rank is 0-indexed. Higher fused score ranks first; ties keep first-seen order.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: -scores[item])


def weighted_reciprocal_rank_fusion(weighted_lists, k: int = 60) -> list:
    """Weighted RRF. score(item) = sum weight * 1/(k + rank + 1), rank 0-indexed.

    weighted_lists: iterable of (ranked_list, weight). Higher fused score ranks
    first; ties keep first-seen order. Items absent from a list contribute 0 there.
    """
    scores: dict = {}
    for ranked, weight in weighted_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    return sorted((item for item in scores if scores[item] > 0), key=lambda item: -scores[item])
