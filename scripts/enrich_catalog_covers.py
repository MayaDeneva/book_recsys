"""One-off: backfill `image_url` into artifacts/catalog.parquet from the raw books file.

The catalog was built before image_url was extracted; this adds it for the existing catalog
book_ids without re-running the whole preprocess pipeline. Future catalog builds include it
automatically (see book_recsys/data/books.py _COLUMNS). Goodreads' `nophoto` placeholder is
kept as-is here and filtered at display time in RecommenderService.card().

    python scripts/enrich_catalog_covers.py
"""
import pandas as pd

from book_recsys.data.books import stream_books_json

CATALOG = "artifacts/catalog.parquet"
RAW = "data/books.json.gz"

catalog = pd.read_parquet(CATALOG)
wanted = set(catalog["book_id"])
covers: dict = {}
scanned = 0
for chunk in stream_books_json(RAW):
    for book_id, url in zip(chunk["book_id"], chunk["image_url"]):
        if book_id in wanted:
            covers[book_id] = url
    scanned += len(chunk)
    print(f"\rscanned {scanned:,} books, matched {len(covers):,}/{len(wanted):,}",
          end="", flush=True)
print()
catalog["image_url"] = catalog["book_id"].map(covers).fillna("")
catalog.to_parquet(CATALOG)
have = int((catalog["image_url"].str.len() > 0).sum())
print("catalog now has image_url for %d/%d books -> %s" % (have, len(catalog), CATALOG))
