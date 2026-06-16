"""Run a set of named recommender configurations through the eval harness."""
import pandas as pd

from book_recsys.eval.harness import evaluate


def run_experiments(recommenders: dict, histories: dict, relevance: dict,
                    k: int = 10) -> dict:
    """Score each named recommender; returns {name: {metric: value}}."""
    return {
        name: evaluate(rec, histories, relevance, k)
        for name, rec in recommenders.items()
    }


def results_table(results: dict) -> pd.DataFrame:
    """Tabulate run_experiments output as a (config x metric) DataFrame."""
    return pd.DataFrame(results).T
