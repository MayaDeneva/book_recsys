# Model Report — Book Recommender Systems

**Author:** Maya Deneva · Dataset: UCSD Goodreads (Wan & McAuley, 2018/2019)
*Living document — accumulates methodology, EDA, and per-use-case results as methods land.*

---

## 1. Methodology

**Preprocessing (all-genre):**
- Source: per-genre interaction files (`goodreads_interactions_<genre>.json.gz`, all 8 genres) + full book-metadata graph.
- k-core filter: users ≥ 20 interactions, books ≥ 10 (true iterative k-core, streamed).
- User sample: 50,000 users (seed 42).
- After edition→work collapse (items are **works**, not editions): **468,628 works · ~12.6M interactions · 50,000 users** (`sample.parquet`, `catalog.parquet`). *(Pre-collapse: ~701k editions / 15.7M interactions.)*

**Content representation:** book document = `Title + Plot + Themes/shelves`, embedded with `BAAI/bge-small-en-v1.5` (384-d), cached (`embeddings.npy`) + FAISS cosine index.

### Evaluation protocols — and which we report

Split: **leave-last-1-out** per user (predict each user's chronologically-last book). One shared
harness scores every method — Recall@{5,10,20}, **NDCG@10 (headline)**, MRR — with paired
**bootstrap 95% CIs** (1,000 resamples).

**"Negatives" means two different things — don't conflate them:**
- **Eval negatives** = *how we score*: what the held-out book is ranked against (the **protocol**).
- **Training negatives** = *how a model was trained* (e.g. `hybrid_cf_content_popneg` is the hybrid
  *trained* on popularity-weighted negatives — a **model variant**, not a protocol).

Three scoring protocols, reported with deliberately different weight:

| protocol | held-out book ranked against | role here |
|---|---|---|
| **Popularity-matched negatives** (a.k.a. popularity-weighted) | 100 *popular* distractors | **HEADLINE** — interpretable (random ≈ 0.10) *and* not gamed by popularity |
| **Full-catalog** | all ~468k works | **honesty anchor** — no sampling bias, but brutally hard → tiny absolute numbers |
| **Uniform random negatives** | 100 *random* distractors | **caveat only** — flatters popularity; kept as a one-paragraph demo, never a results table |

The full-catalog-vs-sampled contrast is itself a finding (the winner flips with protocol —
Krichene & Rendle 2020), so we keep the anchor; but every head-to-head ranking is read off the
**popularity-matched** headline.

---

## 2. EDA findings (`notebooks/01_eda.ipynb`)

- **Ratings are ~54% implicit (rating 0)** — shelved/read but unrated; only ~7.2M of 15.7M carry an explicit 1–5, skewed to 4s/5s. → motivates the explicit-positive (rating ≥ 4) ablation.
- **Heavy-tailed users:** median ~120 interactions/user, mean 314, max 67,759.
- **Long-tail book popularity**; user–book matrix density ≈ **0.04%** (sparsity ~0.9996) → motivates CF + content fusion.
- **~9% of books have empty descriptions** → content falls back to title + shelves.
- **Non-English books present** (language not filtered) → optional English-only ablation.
- Timestamps span 2001–2017 → supports time-based splits.

---

## 3. Results

### UC1 — Established taste (history → next book)

**Headline protocol — popularity-matched sampled negatives** (each held-out book ranked against
100 popularity-matched distractors, see §1). All methods on the same N=2,000 draw, current
work-collapsed artifacts, k=10; random@10 ≈ 0.099. `reports/study_maxsim_popneg.csv`.

| method | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| **max-sim** (per-book) | 0.3550 | **0.2354** | **0.2172** |
| hybrid_cf_content | **0.3935** | 0.2150 | 0.1853 |
| svd | 0.3540 | 0.1929 | 0.1694 |
| content_emb | 0.2585 | 0.1436 | 0.1309 |
| popularity | 0.1405 | 0.0676 | 0.0696 |
| _SASRec (sequential; 30k-user run — reference)_ | _0.4159_ | _0.2438_ | _0.2127_ |

Headline metric **NDCG@10**. (SASRec is from its own 30k-user run at the same protocol — a
reference row, not a same-draw cell; full discussion under *Neural* below.)

**What wins, and why:**
- **Max-similarity is the best non-sequential method** (NDCG 0.235): +9% over the hybrid, +64% over
  mean-pool `content_emb` on the *same* embeddings. It scores each candidate by its max similarity
  to *any* liked book instead of averaging the history into one popularity-drifting centroid — which
  is also the fix for the UI's "same recs every time". Lands in SASRec's range.
- **Hybrid keeps the best recall@10** (0.394) — gets the true book *into* the top-10 most often but
  ranks it lower than max-sim → a recall-vs-MRR trade; both reported.
- **CF+content fusion beats either alone** (hybrid 0.215, svd 0.193 > content 0.144). The learned
  hybrid's coefficients are **cf 13.76 / content 0.52** — CF dominates, but content's non-zero
  weight adds *complementary* (not redundant) signal.
- **Popularity collapses to ≈ random** (recall 0.14 vs 0.099) — once the negatives are *also*
  popular, popularity has no discriminating power (the Krichene & Rendle effect; note below).

**Cost & caveat.** Max-sim is ~720 ms/user full-catalog (≈10× svd — the per-history matmul); the
sampled-neg path is fast. Max-sim clusters around franchises / same-author works (distinct works —
the catalog is edition-collapsed — but repetitive) → pair with an MMR / per-author diversity
re-rank. *(Full-catalog max-sim NDCG + p50/p95 latency: pending a run.)*

> **Why not uniform random negatives?** Ranking against 100 *random* (mostly unpopular) negatives
> flatters every method — popularity alone scores recall@10 ≈ 0.69 ≈ svd — and can reorder methods
> (Krichene & Rendle 2020). We keep that protocol **only** as this one-paragraph cautionary
> demonstration, never as a results table; all headline comparisons use popularity-matched negatives.

**Honesty anchor — full-catalog leave-1-out.** Rank the held-out book against **all ~468k works**
(no sampling, no popularity bias, but brutally hard → tiny absolute numbers; the *ordering* is the
signal). Auto-refreshed by `08_evaluation.ipynb`:

<!-- UC1_TABLE_START -->
| method | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| svd | 0.0440 | 0.0222 | 0.0157 |
| svd_rating>=4 | 0.0400 | 0.0216 | 0.0161 |
| popularity | 0.0193 | 0.0107 | 0.0081 |
| content_tfidf_full | 0.0047 | 0.0031 | 0.0027 |
| content_emb | 0.0013 | 0.0006 | 0.0004 |
<!-- UC1_TABLE_END -->

- **The winner flips with the protocol — that's the finding, not a bug.** On full-catalog, svd
  significantly beats the hybrid (paired bootstrap, see *Beyond-accuracy* below); under
  popularity-matched negatives the hybrid/max-sim win. Same models, opposite order → the *protocol*
  decides (Krichene & Rendle 2020). We report **both**: popularity-matched as the headline,
  full-catalog as the unbiased anchor.
- **Content's strength is elsewhere.** History-averaging makes `content_emb` near-useless for
  full-catalog next-item — its strength is similarity (UC4) and LLM retrieval — yet it's the niche /
  coverage workhorse (see *Beyond-accuracy*). Explicit positives help CF slightly
  (`svd_rating≥4` ≈ `svd`); field/representation choices (TF-IDF ≫ BoW; +plot/+shelves) all sit at
  the floor here and are decided on the UC4 ruler instead — see §4.

**Learned hybrid — design notes.** `LearnedHybridRecommender` reranks CF ∪ content candidates using
each component's score as a feature; popularity is **deliberately excluded** from the default
features (a learned model would give it a positive weight and amplify the popularity tilt).
Confirmed on an earlier consistent run, the `+pop` variant was *worse* (NDCG 0.154 < 0.169) — so
excluding popularity is empirically right, not just principled. Popularity-skew diagnostic
(`popularity_diagnostics`): CF-family methods sit at pop-percentile ≈ 0.99–1.0 with ≤ 0.2% catalog
coverage; `content_emb` at 0.56 with ~4× the coverage — the accuracy↔coverage trade, quantified.
*(The +pop / feature-weight numbers are from an earlier run; they'll be regenerated in the single
`08` pass alongside the headline table.)*

### UC4 — Similar-to-anchor
Content-embedding neighbours vs **behavioural co-read** ground truth (top-10 co-read per anchor; 500 anchors with ≥50 readers), k=10.

| embeddings | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| content (bge-small, no author) | 0.0482 | 0.0496 | 0.1043 |

**Analysis:**
- Content similarity **meaningfully** predicts co-reading — ~340× the random baseline (10/468k) — but only ~5% of an anchor's most-co-read books are among its content-nearest.
- **Co-reading ≠ content similarity:** people co-read across topics (popular books) and by **author loyalty** — signals the title+plot+shelves text only partly captures.
- **A/B pending:** author-enriched embeddings should lift this (same-author books are both content-near *and* heavily co-read). [author re-embed in progress] A content+CF hybrid is the other expected lift.

**Field ablation on the UC4 ruler** (TF-IDF, `notebooks/05_ablations.ipynb` → `reports/study_uc4_field_ablation.csv`) — *which document fields help item-item similarity?* Run on UC4 because UC1 can't discriminate content recipes (every field set is at the full-catalog floor, NDCG ≈ 5e-4; see §3 UC1):

| fields | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| **title** | **0.0900** | **0.1049** | **0.2155** |
| title+plot | 0.0256 | 0.0308 | 0.0804 |
| title+plot+shelves | 0.0380 | 0.0470 | 0.1160 |

- **`title` alone dominates** — co-read ground truth is driven by same-series / same-author books, which title (+author) tokens capture directly. `plot` *dilutes* that signal; adding `shelves` back recovers ~half the loss.
- **Shelves/genre tokens *do* carry item-similarity signal** (0.031 → 0.047 NDCG) — the opposite of the UC1 genre result (a noise-level tie). Judged on the right ruler, genre matters; UC1 just couldn't see it.
- **Re-embed decision:** the dense embeddings already include shelves, so re-embedding *to add genre* buys nothing new. The real UC4 lever is **up-weighting title/series** (the catalog is already edition-collapsed) — not a new content field.

### P3 — LLM (retrieve-then-rerank): UC5 / UC3 / UC1

Pipeline: bge-small retrieves catalog candidates → Reciprocal Rank Fusion → a self-hosted
LLM (Qwen 2.5 via Ollama) reranks / writes a grounded overview. UC3/UC5 are open-ended
(no ground-truth relevance labels exist for "a gift for…"), so they're shown **qualitatively
by example** — and the point is that **classical/neural can't take a text query at all**, so
this is the paradigm's exclusive territory.

**UC5 — zero-shot gift** (free-text request, no user history). Top picks:
- *"a comforting cozy mystery for my grandmother"* → Aunt Dimity series · Gumshoe Granny
  Investigates · Murder of a Sweet Old Lady — cozy + mystery + grandmother, all three captured.
- *"a gift for a 12-year-old who finished Percy Jackson and loves Greek myths"* → Percy Jackson
  & the Olympians · Percy Jackson's Greek Gods / Greek Heroes · The McElderry Book of Greek Myths.
- *"a fast-paced space opera for a teenager who loves Star Wars"* → Star Wars novelizations
  (A New Hope, Jedi Academy). Intent nailed with **zero collaborative signal**.

**UC3 — mood + history** (a fantasy reader's history ⊕ a mood, fused via RRF then reranked):
- history + *"something darker and more adult than usual"* → keeps the fantasy thread (HP Goblet
  of Fire, The Two Towers, The Silmarillion) but shifts darker (Darker Than You Think, Black Ice,
  Draw the Dark).
- history + *"a light, funny read to take a break"* → We Were On a Break · Exit Laughing · The
  Dr. Pepper Prophecies. The taste ⊕ intent blend is something neither CF nor a lone query can do.

**UC1 — LLM rerank (where the LLM *loses*, and why).** On history→next-item the LLM is
**retrieval-ceiling-bound**: its candidates come from history-mean-embedding retrieval, whose
recall@200 is only **~1.2%** (the held-out next book rarely reaches the top-200), so no reranker
can recover it — `content_emb` history-averaging fails UC1 for the same reason (see UC1 anchor).
The LLM's strength is *intent* (UC3/UC5), not next-item prediction.

**The 3-paradigm answer** (the study's headline question — *where does an LLM beat
classical/neural, and where not?*): each paradigm wins where its inductive bias fits —
- **Classical CF / hybrid** → full-catalog next-item (established taste);
- **Neural (SASRec)** → sequential next-item under the popularity-matched eval;
- **LLM** → zero-shot intent & mood (UC3/UC5), where there is no collaborative signal to use.
No single paradigm dominates — which is exactly the point of the benchmark.

### Neural — SASRec (sequential transformer, RecBole)

SASRec trained via RecBole on a **30k-user subsample** (history capped to the most recent
100 per user — see EDA heavy-tail; SASRec attends only to the last ~50 anyway), **BPR loss +
1 sampled negative** (the original-paper regime, also avoids the CE full-softmax OOM on a
16 GB GPU). Its top-K / per-user score rows are exported and scored through the **same eval
harness** as every other method (leave-last-1-out, identical held-out targets), so the
numbers are directly comparable. *(Mult-VAE and GRU4Rec are deferred to the DL deadline.)*

**The headline: the winner flips with the eval protocol** — and that *is* the finding.

*Popularity-matched sampled negatives (the headline protocol), 30k users, k=10* — random@10 ≈ 0.099:

| method | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| **SASRec** | **0.4159** | **0.2438** | **0.2127** |
| content_emb | 0.2720 | 0.1509 | 0.1368 |
| hybrid_cf_content | 0.3085 | 0.1479 | 0.1264 |
| svd | 0.2809 | 0.1443 | 0.1293 |

*Full-catalog leave-1-out (same overlapping users), k=10:*

| method | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| svd | 0.0476 | 0.0245 | 0.0176 |
| hybrid_cf_content | 0.0432 | 0.0230 | 0.0170 |
| SASRec | 0.0410 | 0.0173 | 0.0103 |

**Analysis:**
- **Under the popularity-matched protocol, SASRec decisively wins** — NDCG@10 **0.244**,
  ~**60% above** the best classical (content_emb 0.151, hybrid 0.148, svd 0.144). The gap over
  30k users is far outside sampling noise (paired bootstrap: not close).
- **Why:** popularity-matched negatives strip the **popularity crutch** — the true book is
  ranked against 100 *popular* distractors, so svd/popularity can't score by liking bestsellers.
  SASRec models **reading order and recency** ("given your last N books, what comes next"),
  which is exactly the signal that separates your actual next book from popular-but-unrelated
  ones. The sequential inductive bias wins on its home turf.
- **But on full-catalog leave-1-out, classical CF edges ahead** (svd 0.0245 > hybrid 0.0230 >
  SASRec 0.0173): ranking the held-out book against *all* ~468k items is a different task, and
  the in-process CF/hybrid handle it slightly better than SASRec's exported scores.
- **The protocol decides the winner** — same models, opposite ranking (Krichene & Rendle 2020):
  we report both rather than cherry-picking. Each paradigm wins where its bias fits —
  sequential/neural under popularity-matched, CF/hybrid under full-catalog, the LLM on zero-shot
  intent (UC3/UC5).
- **Caveats:** SASRec is on a 30k-user subsample with history capped at 100; the classical
  methods are scored on the **same users and same candidate sets** for fairness, so the
  comparison is apples-to-apples within each protocol. Per-method `N` differs from the
  full-50k classical tables above (footnote when collating the final cross-paradigm grid).

### Beyond-accuracy, significance, cold-start & latency
Full-catalog leave-1-out, **N = 800** users, k=10 (`notebooks/08_evaluation.ipynb` →
`reports/study_beyond_accuracy.csv`, `study_latency.csv`).

<!-- BEYOND_ACCURACY_START -->
| method | NDCG@10 | 95% CI | intra-list diversity | catalog coverage |
|---|---|---|---|---|
| svd | 0.0295 | [0.0201, 0.0390] | 0.2228 | 0.0021 |
| hybrid_cf_content | 0.0206 | [0.0131, 0.0293] | 0.2226 | 0.0024 |
| content_emb | 0.0000 | [0.0000, 0.0000] | 0.0932 | 0.0034 |
<!-- BEYOND_ACCURACY_END -->

- **Paired bootstrap (1000 resamples, NDCG@10):** svd **beats** hybrid by 0.0088 [0.0031, 0.0147]
  (significant) and hybrid beats content_emb by 0.0206 [0.0131, 0.0293] (significant). Under
  full-catalog — CF's favourable protocol — the ordering **svd > hybrid > content is statistically
  real**, the mirror image of the popularity-matched headline where the hybrid/SASRec/max-sim win.
  The *protocol*, not sampling noise, decides the winner — now with CIs to prove it.
- **Intra-list diversity:** content_emb's top-10 are the *least* internally diverse (0.093 vs 0.223)
  — a tight topical cluster around the history centroid — yet it has the **highest catalog coverage**
  (0.0034, ~1.6× svd). Across users it spreads wide; within a list it's narrow. CF lists are more
  internally varied but headline-concentrated (low coverage).
- **Serendipity (popularity-discounted relevance) ≈ 0 for every method** — under full-catalog
  leave-1-out the hit rate is ~2–3% and the rare hits are popular books (low unexpectedness), so the
  metric can't discriminate here. It needs a denser, more niche relevance signal (multi-item held-out,
  e.g. UC2/UC4) to be informative; recorded as a null result with that caveat.

**Latency (per query, full-catalog rank on the dev laptop, CPU):**

<!-- LATENCY_START -->
| method | p50 (ms) | p95 (ms) |
|---|---|---|
| svd | 64.5 | 148.3 |
| content_emb | 73.0 | 125.4 |
| hybrid_cf_content | 134.7 | 235.5 |
<!-- LATENCY_END -->

The hybrid's candidate-gen + rerank roughly doubles latency (135 ms p50) vs the single-matvec
baselines (~65–73 ms); max-sim is ~720 ms (≈10× svd, the per-history matmul). All are within an
interactive budget; the LLM rerank (P3) is ~1000× slower (15–60 s/query, self-hosted Qwen) — the
**"winner ≠ deployable"** story, quantified.

**Cold-start (relative).** k-core (users ≥ 20, books ≥ 10 interactions) **removes all literal cold
users (< 10) and cold items (< 10) by construction** — so the spec's cold-start sub-eval can't run on
this sample (itself a finding about the sampling design). As a proxy, splitting on history length
(cold = bottom quartile, < 44 interactions, n=196): hybrid NDCG@10 **0.0169 (cold) vs 0.0218 (warm)**
— warmer users get ~30% better recs, the expected CF cold-gradient, but mild because even "cold"
users here still have ≥ 19 interactions.

---

## 4. Ablations
**Done:**
- ✅ **Document fields (UC4 ruler — the decidable one):** title (NDCG 0.105) ≫ +plot (0.031) < +plot+shelves (0.047). Title/series tokens dominate item-similarity; shelves recover signal that plot dilutes. (UC1 can't decide this — every field set is at the full-catalog floor ≈ 5e-4.) See §3 UC4.
- ✅ **TF-IDF vs BoW:** 0.0015 vs 0.0002 (full fields, full-catalog UC1) — TF-IDF wins clearly.
- ✅ **Interactions:** all vs rating ≥ 4 — explicit positives help CF slightly.
- ✅ **Eval protocol:** full-catalog vs uniform- vs popularity-matched negatives — the winner flips with protocol (§3), with the Krichene & Rendle caveat; uniform demoted to a methodology note.
- ✅ **Aggregation:** mean-pool vs **per-book max-sim** (max-sim wins the headline UC1; §3) — the "escape the centroid" result.
- ✅ **Beyond-accuracy + significance + latency + cold-start:** intra-list diversity, serendipity (null), catalog coverage, 95% bootstrap CIs + paired significance, p50/p95 latency, relative cold-start (§3).

**To do:**
- **Embeddings:** bge-small (current) vs bge-large (quality).
- **English-only vs all-languages** catalog.

### Recency-weighted vs flat aggregation (UC2 lever)  *(wired; numbers pending a run)*
`RecencyWeightedRecommender` wraps svd/content so recent history weighs exponentially more (the
simple form of the spec's multi-scale recency). It's a drop-in `Recommender`, scored through the
same harness (`notebooks/08_evaluation.ipynb`, full-catalog `evaluate_per_user` → `study_recency_ablation.csv`).
History is chronological as the harness builds it, so the latest interactions get the most weight.

| base | flat NDCG@10 | recency NDCG@10 |
|---|---|---|
| content_emb | _pending_ | _pending_ |
| svd | _pending_ | _pending_ |

*Read-off:* recency > flat → temporal signal matters even for order-agnostic CF/content; ≈ flat →
SASRec's learned attention captures something exponential decay can't. Either direction is a UC2 finding.

### +author / +genre document-field ablation  *(pending dense re-embed)*

Scored on **UC4** (item-similarity), not UC1 (which is at the floor — see §3). The UC4 field
ablation above is the live TF-IDF version; the bge rows below each need a re-embed (notebook 03).
Hold fixed across every row: same protocol, same N, same split, same model.

| representation | document fields | NDCG@10 (UC4) |
|---|---|---|
| TF-IDF | title + plot + shelves | 0.0470 |
| bge (current) | title + plot + shelves | _TBD_ |
| bge +author | title (+**author**) + plot + shelves | _TBD_ |
| bge +genre | title + **genre** + plot + shelves | _TBD_ |

*Read-off:* positive ΔNDCG from baseline → the field carries similarity signal the embedding wasn't
already getting from shelves; ~0 → shelves already encode it (genre) or the model ignores it (author).

## 5. Caveats
- **Numbers come from runs at different N / draws / artifact versions** (the headline UC1 table is a fresh N=2,000 run on work-collapsed artifacts; the full-catalog anchor is N=800–1,500; SASRec is a 30k-user run). The final cross-paradigm grid must be **regenerated in one `08` pass** at a single N before publication.
- Embeddings are bge-small (384-d) for speed; bge-large is a quality ablation.
- Full-catalog max-sim NDCG, p50/p95 latency, and the recency / +author / +genre ablations are wired but **pending a run**.
