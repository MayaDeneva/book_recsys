"""Collapse duplicate editions to works.

UCSD Goodreads catalogs *editions* (each its own book_id) rather than works, so one book
appears many times with near-identical text. `collapse_editions` maps every edition to its
work's canonical (highest-interaction) edition, remaps the interactions, dedupes per work,
and realigns the embeddings — turning edition-level items into work-level items.
"""
from book_recsys.data.schema import BOOK, TS, USER


def collapse_editions(interactions, catalog, embeddings, work_of):
    """Collapse editions to works. `work_of` maps book_id -> work_id (a book absent from
    the map is treated as its own work). Returns (interactions, catalog, embeddings) where
    every book_id is a work's canonical edition and `embeddings` is realigned row-for-row
    to the returned catalog. `catalog` gains a `work_id` column.
    """
    book_ids = list(catalog[BOOK])
    works = [work_of.get(b, b) for b in book_ids]
    counts = interactions[BOOK].value_counts()

    cat = catalog.copy()
    cat["work_id"] = works
    cat["_count"] = cat[BOOK].map(counts).fillna(0.0)
    canonical = (cat.sort_values("_count", ascending=False, kind="stable")
                    .groupby("work_id")[BOOK].first())   # highest-interaction edition / work
    remap = {b: canonical[w] for b, w in zip(book_ids, works)}

    pos = {b: i for i, b in enumerate(book_ids)}
    kept = (cat[cat[BOOK].isin(set(canonical.values))]
            .drop(columns="_count").reset_index(drop=True))
    emb = embeddings[[pos[b] for b in kept[BOOK]]]

    inter = interactions.copy()
    inter[BOOK] = inter[BOOK].map(remap)
    inter = (inter.sort_values(TS, kind="stable")
             .drop_duplicates([USER, BOOK], keep="last").reset_index(drop=True))
    return inter, kept, emb
