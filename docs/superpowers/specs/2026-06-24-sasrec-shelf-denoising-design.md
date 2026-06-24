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
`book_recsys/data/filters.py::filter_min_rating(..., 1)` — drop `rating == 0`, applied to the
**input history only** (the last two interactions per user are preserved as eval targets; see
Approach).

**Known caveat (record in the report):** `rating == 0` conflates *want-to-read (never read)*
with *read-but-unrated*, because `ingest.py`'s normalized schema discards the raw `is_read`
flag. The proxy therefore also drops some genuinely-read books. The faithful `is_read`-exact
denoise requires re-ingesting the raw interactions JSON and re-running k-core → sample →
work-collapse — **out of scope for the spike**, deferred as the rigorous follow-up *iff* the
spike pays off.

## Approach — denoise the *input history only*, keep the target fixed (directly comparable)

A spike, not a deliverable build. Smallest change that yields a defensible, directly-comparable
finding. The key design choice: **filter only the history the model reads to predict; never
touch the held-out target.** Then exactly one variable changes (the input sequence), the target
set is identical to the existing tables, and denoised-SASRec drops straight in — no shifted
targets, no baseline re-scoring.

1. **Denoise the history, preserve targets** in the `.inter` construction in
   `notebooks/06_recbole.ipynb`: per user (sorted by timestamp) keep a row iff
   `rating >= 1` **OR** it is among that user's **last two** interactions. The last two are the
   leave-one-out **test** (last) and **valid** (2nd-to-last) targets — preserved regardless of
   rating so the held-out target is byte-identical to the raw run. Earlier history is denoised.
   Implementation reuses `filter_min_rating` on the history slice; the last-two rows are
   re-appended.
2. **Equalize the user set:** drop the **651 users (1.30%)** whose denoised history is empty
   (no rated event before their last item) from **every** compared method's row, so all numbers
   are over the identical user set.
3. **Train** one SASRec on the denoised `.inter`, cloud GPU (RecBole still needs CUDA — this
   spike deliberately does **not** touch the UI/RecBole serving blocker).
4. **Compare** denoised-SASRec vs the **existing** raw-SASRec and baseline rows (same users,
   same targets, same candidate sets) on the **popularity-matched headline** (NDCG@10), plus
   full-catalog as the anchor. No re-score of baselines needed beyond the user-set equalization.

## Output

A "raw vs shelf-denoised SASRec" mini-table added to `reports/model_report.md` (Neural —
SASRec section), directly comparable to the existing rows, with the two caveats below recorded.

**Caveat to record — 68% of targets are shelf-adds.** 68.4% of the held-out last-book targets
are themselves `rating==0` shelf-adds. Keeping the original target (what makes the result
comparable) means we mostly predict the *next shelf-add*. This applies **equally** to every
method scored on these targets, so the **relative** head-to-head is fair; it only bounds the
**absolute** interpretation. It is a property of the dataset, not the denoise. *(Optional
secondary view, if wanted: a "predict next *rated* book" cut — the cleaner task — reported as a
separate non-comparable table.)*

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

- **Target preserved → directly comparable:** denoising touches only the input history; the
  held-out target is the user's true last book, identical to the existing tables. No target
  shift, no baseline re-score (beyond dropping the 651 empty-history users from every row).
- **Noisy targets bound absolute (not relative) reading:** 68.4% of targets are `rating==0`
  shelf-adds; fair across methods, but the absolute NDCG reflects "predict the next shelf-add."
- **Checkpoint vocab:** the denoised `.inter` has a different item vocabulary than the shipped
  `SASRec.pth`; this run trains a fresh checkpoint (the existing one is the raw baseline).
- **Sequence-length confound:** if denoised loses, it may be length, not denoising — note that
  the `is_read`-exact version (which keeps read-but-unrated books) would lose less length and is
  the cleaner re-test.
