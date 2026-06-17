from book_recsys.data.kcore import k_core_filter
from book_recsys.data.sample import sample_users
from book_recsys.data.schema import validate_interactions
from book_recsys.data.splits import leave_last_n_out
from book_recsys.eval.harness import build_relevance, build_user_histories, evaluate
from book_recsys.models.classical.popularity import PopularityRecommender


def test_full_pipeline_runs_end_to_end(tiny_interactions):
    df = tiny_interactions
    validate_interactions(df)
    core = k_core_filter(df, min_user=2, min_book=1)
    sampled = sample_users(core, n_users=8, seed=7)
    train, holdout = leave_last_n_out(sampled, n=1)

    rec = PopularityRecommender().fit(train)
    histories = build_user_histories(train)
    relevance = build_relevance(holdout)
    metrics = evaluate(rec, histories, relevance, k=5)

    assert set(metrics) == {"recall@5", "ndcg@5", "mrr"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
