from book_recsys.eval.experiment import results_table, run_experiments


class _FixedRec:

    def __init__(self, recs):
        self._recs = recs

    def recommend(self, query, k):
        return self._recs[:k]


def test_run_experiments_scores_each_config():
    histories = {"u0": ["b9"]}
    relevance = {"u0": {"b0"}}
    configs = {
        "hits": _FixedRec(["b0", "b1"]),  # b0 relevant at rank 1
        "misses": _FixedRec(["b8", "b7"]),  # nothing relevant
    }
    results = run_experiments(configs, histories, relevance, k=2)
    assert results["hits"]["recall@2"] == 1.0
    assert results["misses"]["recall@2"] == 0.0
    assert set(results["hits"]) == {"recall@2", "ndcg@2", "mrr"}


def test_results_table_is_config_by_metric():
    results = {"a": {"recall@2": 1.0, "mrr": 1.0}, "b": {"recall@2": 0.0, "mrr": 0.0}}
    df = results_table(results)
    assert list(df.index) == ["a", "b"]
    assert "recall@2" in df.columns
    assert df.loc["a", "recall@2"] == 1.0
