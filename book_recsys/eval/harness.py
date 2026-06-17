"""Evaluate any Recommender on held-out relevance using ranking metrics."""
import statistics
from typing import Any

import numpy as np
import pandas as pd

from book_recsys.data.negatives import build_cdf, sample_negatives
from book_recsys.data.schema import BOOK, USER
from book_recsys.eval.metrics import mrr, ndcg_at_k, recall_at_k


def build_user_histories(train_df: pd.DataFrame) -> dict[Any, list]:
    """Map each user to the list of book ids they interacted with in training."""
    return train_df.groupby(USER)[BOOK].apply(list).to_dict()


def build_relevance(test_df: pd.DataFrame) -> dict[Any, set]:
    """Map each user to the set of held-out book ids that count as relevant."""
    return test_df.groupby(USER)[BOOK].apply(set).to_dict()


def evaluate_per_user(recommender,
                      histories: dict[Any, list],
                      relevance: dict[Any, set],
                      k: int = 10) -> dict[str, list]:
    """Per-user recall@k / ndcg@k / mrr lists (unaggregated) — feed these to bootstrap CIs.

    Each user's training history is passed as the query so recommenders can exclude
    already-seen books.
    """
    recalls, ndcgs, mrrs = [], [], []
    for user, relevant in relevance.items():
        recs = recommender.recommend(histories.get(user, []), k)
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        mrrs.append(mrr(recs, relevant))
    return {f"recall@{k}": recalls, f"ndcg@{k}": ndcgs, "mrr": mrrs}


def evaluate(recommender,
             histories: dict[Any, list],
             relevance: dict[Any, set],
             k: int = 10) -> dict[str, float]:
    """Mean recall@k, ndcg@k, mrr across all users (see evaluate_per_user for raw scores)."""
    per = evaluate_per_user(recommender, histories, relevance, k)
    return {name: statistics.fmean(vals) for name, vals in per.items()}


def evaluate_predictions(predictions: dict, relevance: dict, k: int = 10) -> dict[str, float]:
    """Mean recall@k, ndcg@k, mrr for precomputed per-user ranked book lists.

    `predictions` maps user -> ranked book ids (e.g. a RecBole batch export). Users
    in `relevance` with no prediction score 0.
    """
    recalls, ndcgs, mrrs = [], [], []
    for user, relevant in relevance.items():
        recs = predictions.get(user, [])
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        mrrs.append(mrr(recs, relevant))
    return {
        f"recall@{k}": statistics.fmean(recalls),
        f"ndcg@{k}": statistics.fmean(ndcgs),
        "mrr": statistics.fmean(mrrs),
    }


def evaluate_sampled_negatives(recommender,
                               histories: dict,
                               relevance: dict,
                               all_items,
                               n_neg: int = 100,
                               k: int = 10,
                               seed: int = 0,
                               item_weights=None) -> dict[str, float]:
    """Rank each held-out positive against `n_neg` negatives (items the user didn't
    interact with). Returns mean recall@k / ndcg@k / mrr over the small set.
    Interpretable, literature-comparable alternative to full-catalog ranking.

    `item_weights` (aligned to `all_items`) switches negatives from uniform to
    popularity-weighted — more confident negatives, and it removes the popularity
    inflation uniform sampling causes (see book_recsys.data.negatives).
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(list(all_items))
    cdf = build_cdf(item_weights) if item_weights is not None else None
    recalls, ndcgs, mrrs = [], [], []
    for user, relevant in relevance.items():
        positive = next(iter(relevant))
        seen = set(histories.get(user, [])) | {positive}
        negatives = sample_negatives(pool, seen, n_neg, rng, cdf)
        candidates = [positive] + negatives
        scores = recommender.score_items(histories.get(user, []), candidates)
        order = [candidates[i] for i in np.argsort(scores)[::-1]]
        recalls.append(recall_at_k(order, {positive}, k))
        ndcgs.append(ndcg_at_k(order, {positive}, k))
        mrrs.append(mrr(order, {positive}))
    return {
        f"recall@{k}": statistics.fmean(recalls),
        f"ndcg@{k}": statistics.fmean(ndcgs),
        "mrr": statistics.fmean(mrrs),
    }


def popularity_diagnostics(recommender,
                           histories: dict,
                           popularity_ranking,
                           catalog_size: int,
                           k: int = 10) -> dict[str, float]:
    """Quantify how popularity-skewed a recommender is — orthogonal to accuracy.

    `popularity_ranking` is books most→least popular (e.g. PopularityRecommender's order).
    Returns `mean_pop_percentile` (avg popularity percentile of recommended items;
    1.0 = always the most popular, lower = more niche; cold items count as 0) and
    `coverage` (fraction of the catalog ever recommended; higher = less head-concentrated).
    """
    n = len(popularity_ranking)
    percentile = {book: 1.0 - i / n for i, book in enumerate(popularity_ranking)}
    pcts: list = []
    recommended: set = set()
    for history in histories.values():
        recs = recommender.recommend(history, k)
        recommended.update(recs)
        pcts.extend(percentile.get(book, 0.0) for book in recs)
    return {
        "mean_pop_percentile": statistics.fmean(pcts) if pcts else 0.0,
        "coverage": len(recommended) / catalog_size,
    }


def build_cooccurrence_relevance(interactions, anchors, top_n: int = 10) -> dict:
    """For each anchor book, the top_n books most often co-read with it (relevant set).

    Behavioral ground truth for similar-to-anchor: books that users who read the anchor
    also read. Anchors absent from the data are skipped; the anchor never includes itself.
    """
    from collections import Counter

    user_books = interactions.groupby(USER)[BOOK].apply(set)
    book_users = interactions.groupby(BOOK)[USER].apply(set)
    relevance = {}
    for anchor in anchors:
        if anchor not in book_users.index:
            continue
        counts: Counter = Counter()
        for user in book_users[anchor]:
            for book in user_books[user]:
                if book != anchor:
                    counts[book] += 1
        relevance[anchor] = {book for book, _ in counts.most_common(top_n)}
    return relevance


def evaluate_similar(recommender, relevance: dict, k: int = 10) -> dict[str, float]:
    """Mean recall@k / ndcg@k / mrr for a similar-to-anchor recommender.

    relevance maps anchor_id -> set of co-read book_ids; `recommender.recommend(anchor, k)`
    returns the model's similar books for that anchor.
    """
    recalls, ndcgs, mrrs = [], [], []
    for anchor, relevant in relevance.items():
        recs = recommender.recommend(anchor, k)
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        mrrs.append(mrr(recs, relevant))
    return {
        f"recall@{k}": statistics.fmean(recalls),
        f"ndcg@{k}": statistics.fmean(ndcgs),
        "mrr": statistics.fmean(mrrs),
    }
