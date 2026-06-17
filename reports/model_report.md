# Model Report — Book Recommender Systems

**Author:** Maya Deneva · Dataset: UCSD Goodreads (Wan & McAuley, 2018/2019)
*Living document — accumulates methodology, EDA, and per-use-case results as methods land.*

---

## 1. Methodology

**Preprocessing (all-genre):**
- Source: per-genre interaction files (`goodreads_interactions_<genre>.json.gz`, all 8 genres) + full book-metadata graph.
- k-core filter: users ≥ 20 interactions, books ≥ 10 (true iterative k-core, streamed).
- User sample: 50,000 users (seed 42).
- Result: **15,708,425 interactions · 50,000 users · 701,085 books** (`sample.parquet`, `catalog.parquet`).

**Content representation:** book document = `Title + Plot + Themes/shelves`, embedded with `BAAI/bge-small-en-v1.5` (384-d), cached (`embeddings.npy`) + FAISS cosine index.

**Evaluation:** leave-last-1-out per user; rank against the **full 701k catalog** (hardest, most honest setting — no sampled negatives); metrics recall@10 / NDCG@10 / MRR via one shared harness for every method.

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
Leave-last-1-out, full-catalog ranking, k=10. (Table auto-refreshed by `notebooks/08_evaluation.ipynb`.)

<!-- UC1_TABLE_START -->
| method | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| svd | 0.0440 | 0.0222 | 0.0157 |
| svd_rating>=4 | 0.0400 | 0.0216 | 0.0161 |
| popularity | 0.0193 | 0.0107 | 0.0081 |
| content_tfidf_full | 0.0047 | 0.0031 | 0.0027 |
| content_emb | 0.0013 | 0.0006 | 0.0004 |
<!-- UC1_TABLE_END -->

**Analysis** (1,500-user sample):
- **Collaborative filtering (SVD) wins UC1** — ~2× popularity. Co-occurrence signal beats both popularity and content for next-item.
- **Explicit positives help CF:** `svd_rating≥4` (NDCG 0.0081) > `svd` on all interactions (0.0075). Filtering the ~54% implicit zeros denoises the factorization — a small, consistent lift. *(answers "which interactions?")*
- **Content field ablation — more fields help:** `title` only → **0.0000** (useless), `+plot` → 0.0013, `+plot+shelves` → **0.0015** (best content). Title alone carries no next-item signal; plot + shelves add real signal. *(answers "which fields?")*
- **TF-IDF ≫ BoW:** full-field TF-IDF 0.0015 vs BoW 0.0002 — term weighting (down-weighting common words) matters; raw counts are ~8× worse. *(the DL-brief TF-IDF-vs-BoW comparison)*
- **Embedding history-averaging fails UC1** (0.0000) — verified *not* a bug: embeddings are high-quality (nearest-neighbour test returns same-series/topical books), but averaging a long, diverse history into one centroid washes out the signal (held-out targets rank deep, ~200k–530k of 701k). Its strength is **similarity (UC4)** and **LLM retrieval**, not history-averaging.
- **Absolute numbers are low by design** — predicting the one held-out book out of 701k via full-catalog leave-last-1-out is brutally hard; *every* method scores low. The **relative ordering** is the signal (sampled-negatives eval would give friendlier absolute numbers).

> Takeaway: pure content-based is a **baseline** for UC1 (and the cold-start/retrieval workhorse elsewhere), not the UC1 winner. Embeddings' strength is **similarity (UC4)** and **LLM retrieval**, not history-averaging.


### UC1 — sampled-negatives evaluation
Same methods, but each held-out book ranked against **100 random negatives** (not the full 701k) — interpretable, literature-comparable. 2,000 users, k=10.

| method | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| svd | 0.7020 | 0.5593 | 0.5244 |
| svd_rating>=4 | 0.6410 | 0.5054 | 0.4734 |
| popularity | 0.6900 | 0.5045 | 0.4578 |
| content_tfidf_full | 0.5415 | 0.3267 | 0.2800 |
| content_emb | 0.3250 | 0.1882 | 0.1673 |

**Analysis:**
- **SVD wins on NDCG@10 / MRR (0.56 / 0.52)** — best at ranking the true book to the top, robust across *both* eval protocols.
- **Methodology caveat made concrete (Krichene & Rendle, 2020):** under sampled-negatives **popularity jumps to recall@10 0.69 ≈ SVD's 0.70** (the held-out book usually beats 100 *random* negatives on popularity alone), vs a distant 0.0087 on full-catalog; and **`svd_rating≥4` flips** (better than `svd` full-catalog, worse here). Sampled metrics can reorder methods — so we report **both** protocols.
- random baseline for 1-in-101 ≈ 0.10, so `content_emb` at 0.33 is ~3× random — weak but working (not the 0.0000 that full-catalog made it look).

### UC1 — popularity-weighted sampled-negatives  *(the fair eval)*
Negatives sampled **∝ popularity** instead of uniformly: the held-out book is ranked against
100 *popular* distractors, so the metric rewards genuine personalization rather than
bestseller-detection. ~2,000 users, k=10. (random@10 for 1-in-101 ≈ **0.099**.)

| method | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| **hybrid_cf_content** | **0.3233** | **0.1692** | **0.1479** |
| hybrid_cf_content_pop | 0.2980 | 0.1541 | 0.1331 |
| content_emb | 0.2540 | 0.1467 | 0.1362 |
| svd | 0.2620 | 0.1346 | 0.1226 |
| popularity | 0.0947 | 0.0415 | 0.0476 |

**Analysis — the most honest UC1 ranking:**
- **Popularity collapses to ≈ random** (recall@10 **0.095** vs random **0.099**): its uniform-negatives
  score (0.69) was a sampling artifact (Krichene & Rendle, 2020) — once the negatives are *also*
  popular, popularity has no discriminating power. The caveat, demonstrated on our own data.
- **The learned hybrid wins** (NDCG **0.169**) — CF+content fusion beats SVD (0.135) and content
  (0.147) *alone*. The headline UC1 result: stacking the paradigms > any single paradigm.
- **Popularity as a feature *hurts*:** `hybrid_cf_content` (0.169) **>** `hybrid_cf_content_pop` (0.154).
  Adding popularity *lowered* accuracy under the fair eval → excluding it from the default is
  **empirically** correct, not just principled.
- **SVD's lead evaporates; content catches up:** under uniform negatives SVD ≫ content
  (0.56 vs 0.19 NDCG); here content_emb **edges** SVD on NDCG/MRR (0.147/0.136 vs 0.135/0.123).
  SVD was partly riding the popularity tilt — remove the crutch and semantic content holds up.
- **Ablation — training negatives:** `hybrid_cf_content_popneg` (hybrid *trained* on popularity
  negatives, same `[cf, content]` features) — does harder training lift the fair-eval score?
  *(pending run)*.

### UC1 — learned hybrid (stacking / feature augmentation)
A meta-model reranks candidates from **CF ∪ content**, using each component's score as a
feature (`LearnedHybridRecommender`). Two stages: candidate generation (top-N from each
component) → learned rerank, trained on leave-last-out positives vs sampled negatives.
`feature_weights()` reports each paradigm's learned contribution. **Results: see the
popularity-weighted table above** — `hybrid_cf_content` is the top UC1 method (NDCG 0.169).

**Design notes:**
- **Popularity deliberately excluded from the default feature set.** A learned model gives
  popularity a positive weight (popular books are likelier positives), which would *amplify*
  the popularity tilt the baselines already show. The default hybrid uses only
  `[cf_score, content_score]`; popularity is a **measured ablation** (`hybrid_cf_content_pop`).
  **Confirmed:** under popularity-weighted negatives the `+pop` variant is *worse*
  (NDCG 0.154 < 0.169) — popularity as a feature hurt, so the default was the right call.
- **Feature weights = the contribution experiment.** Standardized logistic coefficients are
  directly comparable, so the `[cf, content]` weights quantify how much each paradigm
  contributes — *(pending: paste the `feature_weights()` printout from 07_models)*.
- **Popularity-skew diagnostic** (`popularity_diagnostics`, run on every method):
  *mean popularity-percentile* of recommended items (1.0 = always the most popular; lower =
  more niche) + *catalog coverage* (fraction of the catalog ever recommended). This is what
  finally **quantifies** the "recs feel popularity-swayed" concern and shows whether the
  hybrid / `+pop` variant trades personalization for popularity — *(pending run)*.

### UC4 — Similar-to-anchor
Content-embedding neighbours vs **behavioural co-read** ground truth (top-10 co-read per anchor; 500 anchors with ≥50 readers), k=10.

| embeddings | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| content (bge-small, no author) | 0.0482 | 0.0496 | 0.1043 |

**Analysis:**
- Content similarity **meaningfully** predicts co-reading — ~340× the random baseline (10/701k) — but only ~5% of an anchor's most-co-read books are among its content-nearest.
- **Co-reading ≠ content similarity:** people co-read across topics (popular books) and by **author loyalty** — signals the title+plot+shelves text only partly captures.
- **A/B pending:** author-enriched embeddings should lift this (same-author books are both content-near *and* heavily co-read). [author re-embed in progress] A content+CF hybrid is the other expected lift.
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
can recover it — `content_emb` history-averaging fails UC1 for the same reason (see UC1 table).
The LLM's strength is *intent* (UC3/UC5), not next-item prediction.

**The 3-paradigm answer** (the study's headline question — *where does an LLM beat
classical/neural, and where not?*): each paradigm wins where its inductive bias fits —
- **Classical CF / hybrid** → full-catalog next-item (established taste);
- **Neural (SASRec)** → sequential next-item under the fair sampled-negative eval;
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

*Popularity-weighted sampled-negatives (the fair eval), 30k users, k=10* — random@10 ≈ 0.099:

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
- **Under the fair (popularity-weighted) protocol, SASRec decisively wins** — NDCG@10 **0.244**,
  ~**60% above** the best classical (content_emb 0.151, hybrid 0.148, svd 0.144). The gap over
  30k users is far outside sampling noise (paired bootstrap: not close).
- **Why:** popularity-weighted negatives strip the **popularity crutch** — the true book is
  ranked against 100 *popular* distractors, so svd/popularity can't score by liking bestsellers.
  SASRec models **reading order and recency** ("given your last N books, what comes next"),
  which is exactly the signal that separates your actual next book from popular-but-unrelated
  ones. The sequential inductive bias wins on its home turf.
- **But on full-catalog leave-1-out, classical CF edges ahead** (svd 0.0245 > hybrid 0.0230 >
  SASRec 0.0173): ranking the held-out book against *all* ~468k items is a different task, and
  the in-process CF/hybrid handle it slightly better than SASRec's exported scores.
- **The protocol decides the winner** — same models, opposite ranking. This is the Krichene &
  Rendle (2020) theme demonstrated for the neural model: sampled-negatives (SASRec's native
  training/eval regime) vs full-catalog give different orderings, so we report **both** rather
  than cherry-picking. Each paradigm wins where its bias fits — sequential/neural under
  sampled-neg, CF/hybrid under full-catalog, and the LLM on zero-shot intent (UC3/UC5).
- **Caveats:** SASRec is on a 30k-user subsample with history capped at 100; the classical
  methods are scored on the **same users and same candidate sets** for fairness, so the
  comparison is apples-to-apples within each protocol. Per-method `N` differs from the
  full-50k classical tables above (footnote when collating the final cross-paradigm grid).

---

## 4. Ablations
**Done (UC1, see §3 table):**
- ✅ **Document fields:** title (0.0000) → +plot (0.0013) → +shelves (0.0015). More fields help; title alone useless.
- ✅ **TF-IDF vs BoW:** 0.0015 vs 0.0002 (full fields) — TF-IDF wins clearly.
- ✅ **Interactions:** all (svd 0.0075) vs rating ≥ 4 (svd 0.0081) — explicit positives help.

**To do:**
- **Content profile:** flat mean vs **recency-weighted** (needs timestamped history; the UC2 lever — expected to lift content/UC1).
- **Embeddings:** bge-small (current) vs bge-large (quality).
- **Eval:** full-catalog vs sampled-negatives (interpretability).
- **English-only vs all-languages** catalog.

## 5. Caveats
- UC1 numbers are from a 1,500-user sample — to be re-run on all 50k for the final table.
- Embeddings are bge-small (384-d) for speed; bge-large is a quality ablation.
