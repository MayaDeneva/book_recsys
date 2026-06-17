import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.eval.harness import (
    build_relevance,
    build_user_histories,
    evaluate,
    popularity_diagnostics,
)


def _df(triples):
    rows = [{USER: u, BOOK: b, RATING: 5, TS: t} for u, b, t in triples]
    return pd.DataFrame(rows)


def test_build_user_histories_groups_books():
    df = _df([("u0", "b0", 0), ("u0", "b1", 1), ("u1", "b2", 0)])
    hist = build_user_histories(df)
    assert set(hist["u0"]) == {"b0", "b1"}
    assert hist["u1"] == ["b2"]


def test_build_relevance_returns_sets():
    df = _df([("u0", "b0", 0), ("u0", "b0", 1)])
    rel = build_relevance(df)
    assert rel["u0"] == {"b0"}


class _AlwaysB0:
    def recommend(self, query, k):
        return ["b0", "b1"][:k]


def test_evaluate_returns_mean_metrics():
    train = _df([("u0", "b9", 0), ("u1", "b9", 0)])
    test = _df([("u0", "b0", 1), ("u1", "b1", 1)])
    histories = build_user_histories(train)
    relevance = build_relevance(test)
    out = evaluate(_AlwaysB0(), histories, relevance, k=2)
    assert out["recall@2"] == 1.0
    assert round(out["mrr"], 4) == round((1.0 + 0.5) / 2, 4)
    assert "ndcg@2" in out


from book_recsys.eval.harness import evaluate_predictions, evaluate_sampled_negatives


def test_evaluate_predictions_scores_precomputed_rankings():
    relevance = {"u0": {"b0"}, "u1": {"b1"}}
    predictions = {"u0": ["b0", "b9"], "u1": ["b8", "b1"]}
    out = evaluate_predictions(predictions, relevance, k=2)
    assert out["recall@2"] == 1.0
    assert round(out["mrr"], 4) == round((1.0 + 0.5) / 2, 4)
    assert set(out) == {"recall@2", "ndcg@2", "mrr"}


def test_evaluate_predictions_missing_user_scores_zero():
    out = evaluate_predictions({}, {"u0": {"b0"}}, k=5)
    assert out["recall@5"] == 0.0
    assert out["mrr"] == 0.0


def test_evaluate_sampled_negatives_perfect_recommender():
    class _Perfect:
        def score_items(self, history, items):
            return [1.0 if i == "b0" else 0.0 for i in items]
    out = evaluate_sampled_negatives(
        _Perfect(), histories={"u0": ["b9"]}, relevance={"u0": {"b0"}},
        all_items=["b0", "b1", "b2", "b3", "b4"], n_neg=3, k=10, seed=0)
    assert out["recall@10"] == 1.0
    assert set(out) == {"recall@10", "ndcg@10", "mrr"}


def test_evaluate_sampled_negatives_accepts_popularity_weights():
    class _Perfect:
        def score_items(self, history, items):
            return [1.0 if i == "b0" else 0.0 for i in items]
    out = evaluate_sampled_negatives(
        _Perfect(), histories={"u0": ["b9"]}, relevance={"u0": {"b0"}},
        all_items=["b0", "n1", "n2", "n3"], n_neg=2, k=10, seed=0,
        item_weights=[1, 5, 5, 5])   # popularity-weighted negatives path
    assert out["recall@10"] == 1.0
    assert set(out) == {"recall@10", "ndcg@10", "mrr"}


def test_evaluate_sampled_negatives_random_recommender_is_low():
    class _Zero:
        def score_items(self, history, items):
            return [0.0] * len(items)   # ties -> positive not guaranteed in top-1
    out = evaluate_sampled_negatives(
        _Zero(), histories={"u0": ["b9"]}, relevance={"u0": {"b0"}},
        all_items=["b0"] + [f"n{i}" for i in range(50)], n_neg=20, k=1, seed=0)
    assert 0.0 <= out["recall@1"] <= 1.0


from book_recsys.eval.harness import build_cooccurrence_relevance, evaluate_similar


def _inter(rows):
    return pd.DataFrame([{USER: u, BOOK: b, RATING: 5, TS: 0} for u, b in rows])


def test_cooccurrence_picks_most_coread():
    df = _inter([("u0", "a"), ("u0", "x"), ("u0", "y"),
                 ("u1", "a"), ("u1", "x"), ("u1", "z"),
                 ("u2", "a"), ("u2", "x")])
    rel = build_cooccurrence_relevance(df, ["a"], top_n=1)
    assert rel["a"] == {"x"}        # x is co-read with a 3x — the most


def test_cooccurrence_excludes_anchor_and_skips_unknown():
    df = _inter([("u0", "a"), ("u0", "x")])
    rel = build_cooccurrence_relevance(df, ["a", "zzz"], top_n=5)
    assert "a" not in rel["a"]      # the anchor itself is never "relevant"
    assert "zzz" not in rel         # an anchor not in the data is skipped


class _Sim:
    def recommend(self, anchor, k):
        return {"a": ["x", "w"], "b": ["q"]}.get(anchor, [])[:k]


def test_evaluate_similar_scores_against_coread_truth():
    out = evaluate_similar(_Sim(), {"a": {"x"}, "b": {"z"}}, k=2)
    # a: x recommended at rank 1 -> hit ; b: z not in [q] -> miss
    assert out["recall@2"] == 0.5
    assert set(out) == {"recall@2", "ndcg@2", "mrr"}


class _FixedRec:
    """Recommender that returns a fixed list regardless of history."""

    def __init__(self, recs):
        self._recs = recs

    def recommend(self, history, k):
        return self._recs[:k]


def test_popularity_diagnostics_popular_recs_score_high_percentile():
    ranking = ["p0", "p1", "p2", "p3"]  # p0 = most popular
    out = popularity_diagnostics(_FixedRec(["p0", "p1"]), {"u": []}, ranking,
                                 catalog_size=4, k=10)
    assert out["mean_pop_percentile"] > 0.8   # both near the top of the ranking
    assert out["coverage"] == 0.5             # 2 of 4 catalog items recommended


def test_popularity_diagnostics_cold_items_count_as_obscure():
    out = popularity_diagnostics(_FixedRec(["cold"]), {"u": []}, ["p0", "p1"],
                                 catalog_size=10, k=10)
    assert out["mean_pop_percentile"] == 0.0  # not in ranking -> percentile 0
    assert out["coverage"] == 0.1


def test_popularity_diagnostics_no_recs_returns_zero():
    out = popularity_diagnostics(_FixedRec([]), {"u": []}, ["p0"], catalog_size=5, k=10)
    assert out["mean_pop_percentile"] == 0.0
    assert out["coverage"] == 0.0


from book_recsys.eval.harness import evaluate_per_user


def test_evaluate_per_user_returns_unaggregated_lists():
    train = _df([("u0", "b9", 0), ("u1", "b9", 0)])
    test = _df([("u0", "b0", 1), ("u1", "b1", 1)])
    per = evaluate_per_user(_AlwaysB0(), build_user_histories(train), build_relevance(test), k=2)
    assert per["recall@2"] == [1.0, 1.0]   # one score per user, not a mean
    assert len(per["ndcg@2"]) == 2
