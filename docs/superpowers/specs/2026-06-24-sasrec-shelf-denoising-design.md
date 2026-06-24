# SASRec shelf-denoising spike — design

**Date:** 2026-06-24 · **Author:** Maya Deneva · **Status:** approved, pre-implementation
**Course fit:** DL deadline (2026-06-27) — a sequential-model rigor finding.

## Motivation

Documentation review surfaced real sequential signal in the Goodreads sample: a book's
predecessor shares its genre **~35%** of the time vs **~7%** for a random book from the same
user's history. But SASRec underperforms its potential (best single model under the
popularity-matched headline, yet it *loses* to svd on full-catalog leave-1-out). Hypothesis:
the sequence is polluted by **"added-to-shelf" events** whose ordering reflects browsing, not
taste.

### What the data actually says (measured on `artifacts/sample.parquet`, 12.6M interactions)

- **53.8% of interactions are `rating == 0`** — shelved/unrated. The dump's timestamp is
  `date_added` (see `book_recsys/data/ingest.py`), so a SASRec sequence is literally
  "order books were added to shelves."
- **Timestamps are second-precise**, so *exact-second* ties are rare (2.6%) — but at **day**
  granularity **85.4%** of interactions fall on a day where the user logged another book
  (mean 3.88/day, max 4,091). The intra-day second-level order is browse-session order, not
  reading order.
- **The rating filter does not subsume the session noise:** after `rating >= 1`, **77%** of
  remaining interactions are still in multi-book days. "Drop unrated" and "collapse same-day
  bursts" are partially independent levers.
- **Denoising costs sequence length:** `rating >= 1` keeps 46% of interactions (median
  events/user 104 → 53); `rating >= 4` keeps 33% (median 39). The length loss may offset the
  signal gain — that trade is the open question this spike measures.

## Decision: one lever only — drop "added-to-shelf" events

Per explicit scope call: **denoise only the added-to-shelf events; keep every rated event even
if a user rated several books the same day** (multiple ratings/day is real signal, not noise).
**No same-day collapse.**

Operationalized as the existing, tested
`book_recsys/data/filters.py::filter_min_rating(sample, 1)` — drop `rating == 0`.

**Known caveat (record in the report):** `rating == 0` conflates *want-to-read (never read)*
with *read-but-unrated*, because `ingest.py`'s normalized schema discards the raw `is_read`
flag. The proxy therefore also drops some genuinely-read books. The faithful `is_read`-exact
denoise requires re-ingesting the raw interactions JSON and re-running k-core → sample →
work-collapse — **out of scope for the spike**, deferred as the rigorous follow-up *iff* the
spike pays off.

## Approach

A spike, not a deliverable build. Smallest change that yields a defensible finding.

1. **Denoise** the `.inter` construction in `notebooks/06_recbole.ipynb`: apply
   `filter_min_rating(sample, 1)` to `sample` *before* sequences are built. Everything
   downstream (config, train, export, harness scoring) is unchanged.
2. **Train** one SASRec on the denoised `.inter`, cloud GPU (RecBole still needs CUDA — this
   spike deliberately does **not** touch the UI/RecBole serving blocker).
3. **Score** through the existing shared harness. Filtering shifts each user's held-out target
   to their last *rated* book, so for apples-to-apples we **re-score the in-process baselines**
   (svd / hybrid / max-sim / content_emb) on the **same denoised users + targets** via the
   same-draw path cell 21 already implements — identical users + candidate sets.
4. **Compare** denoised-SASRec vs the existing raw-SASRec numbers vs the re-scored baselines,
   on the **popularity-matched headline** protocol (NDCG@10), plus full-catalog as the anchor.

## Output

A "raw vs shelf-denoised SASRec" mini-table added to `reports/model_report.md` (Neural —
SASRec section), with the conflation caveat and the sequence-length trade noted.

## Decision gate (post-spike)

- **Denoising clearly helps** → invest in (a) the `is_read`-exact re-ingest, and/or (b) a
  UI-servable native-torch sequential model (mirror `book_recsys/models/autoencoder/`, trains +
  serves on M4 MPS, no RecBole) that finally closes the UI blocker.
- **No change / worse** → report the null result ("shelf-denoising trades sequential signal for
  context length; net-neutral on next-item") — a legitimate finding for the DL report.

## Scope guard (YAGNI — explicitly out)

No GRU4Rec. No same-day session collapse. No full ablation grid. No UI/serving work. No
`is_read` re-ingest. All of those are gated behind a positive spike result.

## Validity / risks

- **Target shift:** the denoised eval predicts "last *rated* book," not "last *any* book"; the
  re-scored baselines keep it apples-to-apples *within this run*, but the absolute numbers are
  not directly comparable to the 50k raw-target tables — footnote this.
- **Checkpoint vocab:** the denoised `.inter` has a different item vocabulary than the shipped
  `SASRec.pth`; this run trains a fresh checkpoint (the existing one is the raw baseline).
- **Sequence-length confound:** if denoised loses, it may be length, not denoising — note that
  the `is_read`-exact version (which keeps read-but-unrated books) would lose less length and is
  the cleaner re-test.
