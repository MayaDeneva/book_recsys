"""One append-only results ledger so every run accumulates in the same comparable place.

Each call to log_results() drops a run's (method x metric) table into a single CSV, tagging
every row with the config that produced it (protocol, n_users, embed variant, split, ...).
Re-running the same (method, tags) combo replaces its rows instead of duplicating them, so
the ledger is idempotent across reruns. compare() then pivots one metric across any config
axis — e.g. +genre vs no-genre NDCG@10 — without hand-diffing per-run files.
"""
from pathlib import Path

import pandas as pd


def log_results(table: pd.DataFrame, path, key: str = "method", **tags) -> pd.DataFrame:
    """Append a results `table` (index = method, columns = metrics) to the CSV ledger at
    `path`, tagging every row with **tags (e.g. protocol="popneg", n_users=30000,
    embed="bge+genre"). Re-running the same (key, tags) combo replaces its rows. Returns the
    full ledger."""
    rows = table.reset_index(names=key)
    for name, value in tags.items():
        rows[name] = value
    path = Path(path)
    if path.exists():
        rows = pd.concat([pd.read_csv(path), rows], ignore_index=True)
    ident = [key, *tags]
    rows = rows.drop_duplicates(subset=ident, keep="last").reset_index(drop=True)
    rows.to_csv(path, index=False)
    return rows


def compare(ledger: pd.DataFrame,
            metric: str = "ndcg@10",
            index: str = "method",
            columns: str = "embed") -> pd.DataFrame:
    """Pivot the ledger into an (index x columns) table of one metric — e.g. method rows,
    embedding-variant columns, NDCG@10 cells — to read an ablation off directly."""
    return ledger.pivot_table(index=index, columns=columns, values=metric)
