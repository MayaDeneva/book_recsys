"""Backfill a `genre` column into artifacts/catalog.parquet from the Goodreads genre file,
so the embedding/document pipeline can run the +genre ablation.

Setup (one-time): download the genre file from the UCSD Goodreads page
  https://mengtingwan.github.io/data/goodreads.html  ->  goodreads_book_genres_initial.json.gz
and put it under data/. Then:

  python scripts/enrich_catalog_genres.py

After that:
  - TF-IDF/BoW +genre ablation works immediately (build_documents(catalog,
    fields=("title", "genre", "plot", "shelves")) in 05_ablations).
  - For the bge/semantic +genre variant, RE-EMBED with the same `fields` (03_embed_local)
    and compare against the current embeddings (which omit genre).
"""
import glob

import pandas as pd

from book_recsys.data.genres import stream_genres

CATALOG = "artifacts/catalog.parquet"
RAW = next(iter(glob.glob("data/goodreads_book_genres_initial.json*")), None)
if RAW is None:
    raise FileNotFoundError(
        "data/goodreads_book_genres_initial.json[.gz] not found — download it from "
        "https://mengtingwan.github.io/data/goodreads.html")

catalog = pd.read_parquet(CATALOG)
genres = stream_genres(RAW)
catalog["genre"] = catalog["book_id"].map(genres).fillna("")
catalog.to_parquet(CATALOG)
have = int((catalog["genre"].str.len() > 0).sum())
print(f"catalog now has `genre` for {have:,}/{len(catalog):,} books -> {CATALOG}")
print("sample:", catalog.loc[catalog['genre'].str.len() > 0, 'genre'].head(3).tolist())
