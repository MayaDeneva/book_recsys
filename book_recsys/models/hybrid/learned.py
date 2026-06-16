"""Learned hybrid recommender (stacking / feature augmentation).

Two stages: the component recommenders generate candidates, then a meta-model
reranks them using each component's score as a feature. Trained on leave-last-out
positives vs sampled negatives, so the meta-model learns *how* to combine the
paradigms rather than fusing their outputs with a hand-tuned weight.

`feature_weights()` exposes each component's learned contribution — for the default
logistic model these are standardized coefficients (directly comparable), so the
hybrid doubles as the "how much does each paradigm contribute" experiment.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from book_recsys.data.negatives import build_cdf, sample_negatives
from book_recsys.data.schema import BOOK, USER

_NEG_INF = float("-inf")


class LearnedHybridRecommender:
    """Rerank candidates from several component recommenders with a learned meta-model.

    scorers: ordered dict {name: recommender}; each needs score_items(history, items)
    and recommend(history, k). candidate_k caps each component's candidate pool.
    model defaults to a standardized logistic regression (interpretable coefficients);
    pass any sklearn classifier (e.g. a tree) for a non-linear combiner.
    """

    def __init__(self, scorers: dict, candidate_k: int = 100, n_neg: int = 4,
                 model=None, seed: int = 0, neg_sampling: str = "uniform") -> None:
        self._scorers = dict(scorers)
        self._names = list(self._scorers)
        self.candidate_k = candidate_k
        self.n_neg = n_neg
        self._seed = seed
        self.neg_sampling = neg_sampling
        self._model = model if model is not None else make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000))
        self._floors = None

    def _raw_features(self, history, items) -> np.ndarray:
        cols = [self._scorers[name].score_items(history, items) for name in self._names]
        return np.array(cols, dtype=float).T  # (n_items x n_scorers)

    @staticmethod
    def _column_floors(x: np.ndarray) -> np.ndarray:
        """Per-feature finite minimum — the value imputed for items a component can't
        score (-inf), i.e. "cold to this model" ≈ that model's weakest affinity."""
        floors = []
        for j in range(x.shape[1]):
            finite = x[:, j][np.isfinite(x[:, j])]
            floors.append(float(finite.min()) if finite.size else 0.0)
        return np.array(floors)

    def _impute(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for j in range(x.shape[1]):
            x[~np.isfinite(x[:, j]), j] = self._floors[j]
        return x

    def fit(self, train_data) -> "LearnedHybridRecommender":
        histories = train_data.groupby(USER)[BOOK].apply(list)
        counts = train_data[BOOK].value_counts()
        pool = counts.index.to_numpy()
        cdf = build_cdf(counts.to_numpy()) if self.neg_sampling == "popularity" else None
        rng = np.random.default_rng(self._seed)
        feats, labels = [], []
        for books in histories:
            if len(books) < 2:
                continue
            history, positive = books[:-1], books[-1]
            negatives = sample_negatives(pool, set(books), self.n_neg, rng, cdf)
            feats.append(self._raw_features(history, [positive] + negatives))
            labels.extend([1] + [0] * self.n_neg)
        x = np.vstack(feats)
        self._floors = self._column_floors(x)
        self._model.fit(self._impute(x), np.array(labels))
        return self

    def _final_estimator(self):
        return self._model.steps[-1][1] if isinstance(self._model, Pipeline) else self._model

    def feature_weights(self) -> dict:
        """Each component's learned contribution: standardized coefficients for a
        linear model, else feature importances for a tree."""
        clf = self._final_estimator()
        vals = np.ravel(clf.coef_) if hasattr(clf, "coef_") else np.ravel(clf.feature_importances_)
        return dict(zip(self._names, (float(v) for v in vals)))

    def score_items(self, history, item_ids) -> list:
        if not item_ids:
            return []
        raw = self._raw_features(history, item_ids)
        fully_unknown = ~np.isfinite(raw).any(axis=1)
        proba = self._model.predict_proba(self._impute(raw))[:, 1]
        return [_NEG_INF if fully_unknown[i] else float(proba[i]) for i in range(len(item_ids))]

    def recommend(self, query, k: int) -> list:
        seen = set(query)
        candidates: list = []
        for name in self._names:
            for b in self._scorers[name].recommend(query, self.candidate_k):
                if b not in seen and b not in candidates:
                    candidates.append(b)
        if not candidates:
            return []
        scores = self.score_items(query, candidates)
        order = np.argsort(scores)[::-1]
        return [candidates[i] for i in order[:k]]
