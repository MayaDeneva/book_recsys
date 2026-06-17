"""UC3 / UC5 qualitative demo: LLM retrieve-then-rerank on intent queries.

This is where the LLM is designed to win (unlike UC1, whose history-mean retrieval
ceiling is ~1.2%). Text queries are encoded with bge-small (matching the 384-d
catalog) and reranked by the local Ollama model. Output = top-K titles per query,
for the report / slides / UI.

    python scripts/demo_llm_uc35.py --only 1     # smoke one query
    python scripts/demo_llm_uc35.py              # full curated set
"""
import argparse

import numpy as np
import pandas as pd

ART = "artifacts"


def dedup_titles(book_ids, id_to_title, k):
    seen, out = set(), []
    for b in book_ids:
        t = id_to_title.get(b, str(b))
        key = t.lower().split(" (")[0].strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= k:
            break
    return out


def find_ids(catalog, substrings):
    ids = []
    for s in substrings:
        hit = catalog[catalog["title"].str.contains(s, case=False, na=False)]
        if len(hit):
            ids.append(hit.iloc[0]["book_id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="run only the first N queries")
    ap.add_argument("--retrieve-n", type=int, default=60)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--doc-chars", type=int, default=200)
    ap.add_argument("--model", default="ollama/qwen2.5vl:7b")
    ap.add_argument("--api-base", default="http://localhost:11434")
    args = ap.parse_args()

    print("loading catalog + embeddings ...", flush=True)
    catalog = pd.read_parquet("%s/catalog.parquet" % ART)
    emb = np.load("%s/embeddings.npy" % ART)
    id_to_title = dict(zip(catalog["book_id"], catalog["title"]))

    from book_recsys.features.document import build_documents
    from book_recsys.llm.clients import LiteLLMClient
    from book_recsys.llm.retrieve import Retriever
    from book_recsys.models.llm.recommender import LLMRecommender
    from sentence_transformers import SentenceTransformer

    docs = [d[:args.doc_chars] for d in build_documents(catalog)]
    id_to_doc = dict(zip(catalog["book_id"], docs))

    print("loading bge-small query encoder ...", flush=True)
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")  # 384-d, matches the catalog
    retriever = Retriever(catalog["book_id"].tolist(), emb, encoder=encoder)
    client = LiteLLMClient(args.model, api_base=args.api_base, timeout=120)
    rec = LLMRecommender(retriever, id_to_doc, client,
                         retrieve_n=args.retrieve_n, rerank_batch=args.batch).fit()

    # UC3 needs a coherent reader history; build a fantasy reader from title lookups.
    fantasy_hist = find_ids(catalog, ["Harry Potter and the Sorcerer",
                                       "The Hobbit", "Eragon", "Percy Jackson"])

    queries = [
        ("UC5", "a comforting cozy mystery for my grandmother"),
        ("UC5", "a gift for a 12-year-old who just finished Percy Jackson and loves Greek myths"),
        ("UC5", "an uplifting non-fiction book about resilience and overcoming hardship"),
        ("UC5", "a fast-paced space opera for a teenager who loves Star Wars"),
        ("UC3", {"history": fantasy_hist, "query": "something darker and more adult than usual"}),
        ("UC3", {"history": fantasy_hist, "query": "a light, funny read to take a break"}),
    ]
    if args.only:
        queries = queries[:args.only]

    for tag, q in queries:
        if isinstance(q, dict):
            hist_titles = dedup_titles(q["history"], id_to_title, 10)
            print("\n[%s] history=%s + mood=%r" % (tag, hist_titles, q["query"]), flush=True)
        else:
            print("\n[%s] query=%r" % (tag, q), flush=True)
        recs = rec.recommend(q, 10)
        for i, t in enumerate(dedup_titles(recs, id_to_title, 10), 1):
            print("   %2d. %s" % (i, t))


if __name__ == "__main__":
    main()
