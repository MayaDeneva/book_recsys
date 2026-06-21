import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from book_recsys.data.schema import BOOK, RATING, USER
from book_recsys.models.hybrid.learned import LearnedHybridRecommender

HI = [f"hi{i}" for i in range(5)]
LO = [f"lo{i}" for i in range(20)]
A_SCORES = {**{b: 1.0 for b in HI}, **{b: 0.0 for b in LO}}
B_SCORES = {b: 0.5 for b in HI + LO}


class FakeScorer:
    """Fixed per-book score (ignores history). Mirrors the Recommender surface the
    hybrid needs: score_items(history, items) and recommend(history, k)."""

    def __init__(self, scores, default=float("-inf")):
        self.scores = scores
        self.default = default

    def score_items(self, history, items):
        return [self.scores.get(b, self.default) for b in items]

    def recommend(self, history, k):
        ranked = sorted(self.scores, key=lambda b: -self.scores[b])
        return [b for b in ranked if b not in set(history)][:k]


def _scorers():
    return {"A": FakeScorer(A_SCORES), "B": FakeScorer(B_SCORES)}


def _learning_train():
    """Each user: a low-A context book then a high-A positive (held out last).
    So the positive is separable on feature A and noise on feature B."""
    rows = []
    for u in range(40):
        rows.append((f"u{u}", LO[u % len(LO)], 5))
        rows.append((f"u{u}", HI[u % len(HI)], 5))
    return pd.DataFrame(rows, columns=[USER, BOOK, RATING])


def _fitted():
    return LearnedHybridRecommender(_scorers(), candidate_k=10, seed=0).fit(_learning_train())


def test_feature_weights_one_per_scorer():
    weights = _fitted().feature_weights()
    assert set(weights) == {"A", "B"}


def test_learns_predictive_feature_gets_higher_weight():
    weights = _fitted().feature_weights()
    assert weights["A"] > weights["B"]


def test_score_items_ranks_high_feature_above_low():
    scores = _fitted().score_items([LO[0]], [HI[0], LO[1]])
    assert scores[0] > scores[1]


def test_score_items_empty_items_returns_empty():
    assert _fitted().score_items([LO[0]], []) == []


def test_score_items_cold_item_in_both_scorers_is_neg_inf():
    out = _fitted().score_items([LO[0]], ["unknown_book"])
    assert out == [float("-inf")]


def test_score_items_item_cold_to_one_scorer_still_scored():
    # known to A, absent from B -> not fully unknown -> finite score (B imputed)
    hybrid = LearnedHybridRecommender({
        "A": FakeScorer({"x": 1.0}),
        "B": FakeScorer({})
    }, seed=0).fit(_learning_train())
    out = hybrid.score_items([LO[0]], ["x"])
    assert np.isfinite(out[0])


def test_recommend_returns_topk_excluding_seen():
    recs = _fitted().recommend([HI[0]], k=3)
    assert len(recs) == 3
    assert HI[0] not in recs
    assert len(set(recs)) == 3  # no duplicates across scorers


def test_recommend_empty_candidates_returns_empty():
    hybrid = LearnedHybridRecommender({
        "A": FakeScorer({}),
        "B": FakeScorer({})
    }, seed=0).fit(_learning_train())
    assert hybrid.recommend([LO[0]], k=5) == []


def test_skips_users_with_single_interaction():
    rows = _learning_train().values.tolist() + [["solo", "hi0", 5]]
    df = pd.DataFrame(rows, columns=[USER, BOOK, RATING])
    hybrid = LearnedHybridRecommender(_scorers(), seed=0).fit(df)  # must not raise
    assert set(hybrid.feature_weights()) == {"A", "B"}


def test_popularity_negative_sampling_fits_and_recommends():
    hybrid = LearnedHybridRecommender(_scorers(),
                                      candidate_k=10,
                                      neg_sampling="popularity",
                                      seed=0).fit(_learning_train())
    assert set(hybrid.feature_weights()) == {"A", "B"}
    assert len(hybrid.recommend([HI[0]], k=3)) == 3


def test_custom_tree_model_reports_feature_importances():
    hybrid = LearnedHybridRecommender(_scorers(),
                                      model=DecisionTreeClassifier(random_state=0),
                                      seed=0).fit(_learning_train())
    weights = hybrid.feature_weights()
    assert set(weights) == {"A", "B"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9  # importances sum to 1
