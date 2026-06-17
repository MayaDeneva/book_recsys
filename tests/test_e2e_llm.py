import numpy as np
import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.eval.harness import build_relevance, build_user_histories, evaluate
from book_recsys.llm.retrieve import Retriever
from book_recsys.models.llm.recommender import LLMRecommender

BOOK_IDS = ["b0", "b1", "b2", "b3"]
EMB = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
DOCS = {b: f"doc {b}" for b in BOOK_IDS}


class _Encoder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


class _RankClient:
    def complete(self, prompt):
        return '[{"id":"b1","score":9},{"id":"b2","score":2}]'


def test_llm_recommender_scores_through_harness():
    retriever = Retriever(BOOK_IDS, EMB, encoder=_Encoder())
    rec = LLMRecommender(retriever, DOCS, _RankClient(), retrieve_n=4).fit()

    train = pd.DataFrame([{USER: "u0", BOOK: "b0", RATING: 5, TS: 0}])
    holdout = pd.DataFrame([{USER: "u0", BOOK: "b1", RATING: 5, TS: 1}])
    histories = build_user_histories(train)
    relevance = build_relevance(holdout)

    metrics = evaluate(rec, histories, relevance, k=3)
    assert set(metrics) == {"recall@3", "ndcg@3", "mrr"}
    assert metrics["recall@3"] == 1.0  # b1 retrieved (near b0) and reranked top
