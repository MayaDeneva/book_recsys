"""Headless Mult-VAE training + eval mirror of notebooks/10_autoencoder.ipynb.

Run from the repo root once artifacts/sample.parquet exists. The notebook is the primary
runner; this script is for resume-friendly headless/Kaggle runs.
"""
import argparse
import os

import pandas as pd

from book_recsys.data.negatives import build_cdf
from book_recsys.data.schema import BOOK
from book_recsys.data.splits import leave_last_n_out
from book_recsys.eval.harness import (build_relevance, build_user_histories,
                                      evaluate_sampled_negatives, popularity_diagnostics)
from book_recsys.models.autoencoder.data import reproduce_sasrec_sample
from book_recsys.models.autoencoder.recommender import MultVaeRecommender


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="artifacts/sample.parquet")
    p.add_argument("--n-users", type=int, default=30000)
    p.add_argument("--max-hist", type=int, default=100)
    p.add_argument("--expect-rows", type=int, default=2_273_496)
    p.add_argument("--latent", type=int, default=200)
    p.add_argument("--hidden", type=int, default=600)
    p.add_argument("--beta-cap", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-item-count", type=int, default=1)
    p.add_argument("--pop-discount", type=float, default=0.0)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-dir", default="artifacts")
    p.add_argument("--n-eval-users", type=int, default=2000)
    args = p.parse_args()

    sample = pd.read_parquet(args.sample)
    sample = reproduce_sasrec_sample(sample,
                                     args.n_users,
                                     args.max_hist,
                                     expect_rows=args.expect_rows)
    train, test = leave_last_n_out(sample, n=1)

    rec = MultVaeRecommender(hidden=args.hidden,
                             latent=args.latent,
                             dropout=args.dropout,
                             beta_cap=args.beta_cap,
                             epochs=args.epochs,
                             lr=args.lr,
                             min_item_count=args.min_item_count,
                             pop_discount=args.pop_discount,
                             device=args.device,
                             ckpt_dir=args.ckpt_dir)
    rec.fit(train)

    histories = build_user_histories(train)
    relevance = build_relevance(test)
    eval_users = list(relevance)[:args.n_eval_users]
    relevance = {u: relevance[u] for u in eval_users}
    all_items = sample[BOOK].unique()
    weights = sample[BOOK].value_counts().reindex(all_items).to_numpy()

    headline = evaluate_sampled_negatives(rec,
                                          histories,
                                          relevance,
                                          all_items,
                                          n_neg=100,
                                          k=10,
                                          seed=0,
                                          item_weights=weights)
    pop = popularity_diagnostics(rec, {u: histories.get(u, [])
                                       for u in eval_users},
                                 list(sample[BOOK].value_counts().index),
                                 catalog_size=len(all_items),
                                 k=10)
    print("headline (popularity-matched neg):", headline)
    print("diagnostics:", pop)


if __name__ == "__main__":  # pragma: no cover
    main()
