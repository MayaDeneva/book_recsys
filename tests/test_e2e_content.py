import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.eval.harness import build_relevance, build_user_histories, evaluate
from book_recsys.features.document import build_documents
from book_recsys.features.vectorize import tfidf_matrix
from book_recsys.models.content.content import ContentRecommender
from book_recsys.models.content.similar import SimilarItemsRecommender


CATALOG = pd.DataFrame({
    "book_id": ["b0", "b1", "b2", "b3"],
    "title": ["Dragon", "Wizard", "Romance", "Mage"],
    "description": ["a young mage rises against darkness",
                    "a young mage battles darkness",
                    "two lovers meet in spring",
                    "a mage studies ancient darkness"],
    "shelves": [["fantasy"], ["fantasy"], ["romance"], ["fantasy"]],
})


def _build_content_recommender():
    docs = build_documents(CATALOG)
    matrix, _ = tfidf_matrix(docs, max_features=50)
    return ContentRecommender(CATALOG["book_id"].tolist(), matrix.toarray())


def test_content_recommender_scores_through_harness():
    rec = _build_content_recommender().fit()
    train = pd.DataFrame([{USER: "u0", BOOK: "b0", RATING: 5, TS: 0}])
    holdout = pd.DataFrame([{USER: "u0", BOOK: "b1", RATING: 5, TS: 1}])
    histories = build_user_histories(train)
    relevance = build_relevance(holdout)
    metrics = evaluate(rec, histories, relevance, k=3)
    assert set(metrics) == {"recall@3", "ndcg@3", "mrr"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_similar_items_finds_a_fantasy_neighbor():
    docs = build_documents(CATALOG)
    matrix, _ = tfidf_matrix(docs, max_features=50)
    rec = SimilarItemsRecommender(CATALOG["book_id"].tolist(), matrix.toarray()).fit()
    neighbors = rec.recommend("b0", k=3)
    assert "b2" != neighbors[0]  # the romance book is not the closest to a fantasy book
