"""UC1 evaluation: LLM retrieve-then-rerank vs classical baselines, same harness.

Scores every method through book_recsys.eval.metrics on an identical, seeded
subsample of users (the LLM is ~20 Ollama calls/user, so we subsample). Each
held-out last book is ranked; we report recall@10 / ndcg@10 / mrr.

The LLM is retrieval-limited (reranks only the top `retrieve_n` candidates), so
its numbers carry a retrieval ceiling the full-catalog baselines do not — a
documented caveat, not a bug.

Usage:
    python scripts/eval_llm_uc1.py --n 3            # quick smoke
    python scripts/eval_llm_uc1.py --n 50 --out artifacts/llm_uc1.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from book_recsys.eval.harness import build_relevance, build_user_histories
from book_recsys.eval.metrics import mrr, ndcg_at_k, recall_at_k

ART = "artifacts"
K = 10


def score_user(recs, relevant, k):
    return recall_at_k(recs, relevant, k), ndcg_at_k(recs, relevant, k), mrr(recs, relevant)


def mean_scores(rows):
    arr = np.array(rows, dtype=float)
    return {"recall@%d" % K: arr[:, 0].mean(), "ndcg@%d" % K: arr[:, 1].mean(),
            "mrr": arr[:, 2].mean()}


def evaluate_method(rec, users, histories, relevance, label, progress=False):
    rows, fails = [], 0
    t0 = time.time()
    for i, u in enumerate(users, 1):
        try:
            recs = rec.recommend(histories.get(u, []), K)
        except Exception as exc:  # noqa: BLE001 — keep going, report at end
            fails += 1
            recs = []
            if progress:
                print("    user %s failed: %r" % (u, exc))
        rows.append(score_user(recs, relevance[u], K))
        if progress:
            print("  [%s] %d/%d  elapsed %.0fs" % (label, i, len(users), time.time() - t0),
                  flush=True)
    out = mean_scores(rows)
    out["_users"], out["_fails"], out["_secs"] = len(users), fails, round(time.time() - t0, 1)
    return out


def pick_users(histories, relevance, n, seed):
    eligible = [u for u in relevance if histories.get(u)]
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    return eligible[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--retrieve-n", type=int, default=200)
    ap.add_argument("--doc-chars", type=int, default=200,
                    help="truncate each book document in the rerank prompt (memory/latency)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="ollama/qwen2.5vl:7b")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--baselines", action="store_true", help="also score svd/hybrid/popularity")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("loading sample + split ...", flush=True)
    sample = pd.read_parquet("%s/sample.parquet" % ART)
    from book_recsys.data.splits import leave_last_n_out
    train, holdout = leave_last_n_out(sample, n=1)
    histories = build_user_histories(train)
    relevance = build_relevance(holdout)
    users = pick_users(histories, relevance, args.n, args.seed)
    print("scoring %d users (seed %d)" % (len(users), args.seed), flush=True)

    results = {}

    if args.baselines:
        print("loading models.joblib (classical baselines) ...", flush=True)
        import gc

        import joblib
        models = joblib.load("%s/models.joblib" % ART)
        for name in ("popularity", "svd", "hybrid_cf_content"):
            if name in models:
                results[name] = evaluate_method(models[name], users, histories, relevance, name)
                print("  %-18s %s" % (name, results[name]), flush=True)
        del models
        gc.collect()

    print("building retriever + LLM recommender ...", flush=True)
    catalog = pd.read_parquet("%s/catalog.parquet" % ART)
    emb = np.load("%s/embeddings.npy" % ART)
    from book_recsys.features.document import build_documents
    from book_recsys.llm.clients import LiteLLMClient
    from book_recsys.llm.retrieve import Retriever
    from book_recsys.models.llm.recommender import LLMRecommender

    book_ids = catalog["book_id"].tolist()
    docs = build_documents(catalog)
    if args.doc_chars:
        docs = [d[:args.doc_chars] for d in docs]
    id_to_doc = dict(zip(book_ids, docs))
    retriever = Retriever(book_ids, emb, encoder=None)  # UC1 = history only, no text encode
    client = LiteLLMClient(args.model, api_base=args.api_base, timeout=120)
    llm = LLMRecommender(retriever, id_to_doc, client, retrieve_n=args.retrieve_n).fit()

    results["llm_rerank"] = evaluate_method(llm, users, histories, relevance,
                                            "llm_rerank", progress=True)

    print("\n=== UC1 results (%d users, retrieve_n=%d) ===" % (len(users), args.retrieve_n))
    print("%-20s %10s %10s %10s" % ("method", "recall@%d" % K, "ndcg@%d" % K, "mrr"))
    for name, r in results.items():
        print("%-20s %10.4f %10.4f %10.4f" % (name, r["recall@%d" % K], r["ndcg@%d" % K],
                                              r["mrr"]))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print("\nsaved -> %s" % args.out)


if __name__ == "__main__":
    main()
