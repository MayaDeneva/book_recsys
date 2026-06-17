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
