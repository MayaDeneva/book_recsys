import pandas as pd

from book_recsys.eval.ledger import compare, log_results


def _table(ndcg):
    """A tiny results table shaped like results_table(): method index, metric columns."""
    return pd.DataFrame({"ndcg@10": ndcg}, index=["svd", "hybrid"])


def test_log_creates_ledger_and_tags_every_row(tmp_path):
    path = tmp_path / "ledger.csv"
    led = log_results(_table([0.10, 0.20]), path, protocol="popneg", embed="bge")
    assert path.exists()
    assert set(led["method"]) == {"svd", "hybrid"}
    # both config tags land on every row
    assert list(led["protocol"]) == ["popneg", "popneg"]
    assert list(led["embed"]) == ["bge", "bge"]


def test_log_appends_a_different_config(tmp_path):
    path = tmp_path / "ledger.csv"
    log_results(_table([0.10, 0.20]), path, protocol="popneg", embed="bge")
    led = log_results(_table([0.12, 0.22]), path, protocol="popneg", embed="bge+genre")
    # nothing clobbered: 2 methods x 2 embed variants
    assert len(led) == 4
    assert set(led["embed"]) == {"bge", "bge+genre"}


def test_rerunning_same_config_replaces_rows(tmp_path):
    path = tmp_path / "ledger.csv"
    log_results(_table([0.10, 0.20]), path, protocol="popneg", embed="bge")
    led = log_results(_table([0.11, 0.21]), path, protocol="popneg", embed="bge")
    # same (method, tags) -> updated in place, not duplicated
    assert len(led) == 2
    svd = led.loc[led["method"] == "svd", "ndcg@10"].iloc[0]
    assert svd == 0.11


def test_compare_pivots_one_metric_across_a_config_axis(tmp_path):
    path = tmp_path / "ledger.csv"
    log_results(_table([0.10, 0.20]), path, protocol="popneg", embed="bge")
    led = log_results(_table([0.12, 0.22]), path, protocol="popneg", embed="bge+genre")
    tbl = compare(led, metric="ndcg@10", index="method", columns="embed")
    assert tbl.loc["svd", "bge"] == 0.10
    assert tbl.loc["svd", "bge+genre"] == 0.12
