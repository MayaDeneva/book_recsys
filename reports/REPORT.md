# Model Report — Book Recommender

**Maya Deneva · UCSD Goodreads.** Items are **works** (editions collapsed 701k → 468k). 30k users /
2.27M interactions; task = history → next book (leave-last-1-out). Headline = **popularity-matched
sampled negatives** (1 positive vs 100 *popular* decoys; random ≈ 0.10 NDCG@10).

## Headline results (popularity-matched negatives)

Same protocol, same users (N=3000), sorted by NDCG@10:

| model | type | NDCG@10 | Recall@10 | strong / weak |
|---|---|---|---|---|
| **Hybrid (Mult-VAE ⊕ TF-IDF)** | RRF ensemble | **0.304** | **0.504** | best overall / only helps with a complementary signal |
| **Mult-VAE (α=1)** | autoencoder | 0.303 | 0.490 | best single model + de-biasable / needs the α knob |
| max-sim | content (bge) | 0.243 | 0.370 | tail + sparse users / weak vs CF on random negs |
| learned hybrid (CF+content) | stacking | 0.214 | 0.387 | blends signals / no clear edge |
| **SVD** | CF (matrix factorization) | 0.179 | 0.345 | co-read behaviour / pathologically popularity-biased |
| content_emb | content | 0.146 | 0.259 | tail / weak headline accuracy |
| Popularity | baseline | 0.059 | 0.129 | cheap / bestsellers only |

### Best model — hybrid with the *complementary* content signal

Fusing Mult-VAE with **TF-IDF** max-sim **beats Mult-VAE alone**; fusing with **bge** does not
(same N=2000 draw, popularity-matched):

| model | NDCG@10 | Recall@10 |
|---|---|---|
| **Mult-VAE ⊕ TF-IDF max-sim** | **0.317** | **0.525** |
| Mult-VAE (α=1) | 0.305 | 0.497 |
| Mult-VAE ⊕ bge max-sim | 0.299 | 0.482 |
| bge max-sim (standalone) | 0.243 | 0.371 |
| TF-IDF max-sim (standalone) | 0.227 | 0.381 |

The **weaker standalone content model (TF-IDF) makes the *better* hybrid.** TF-IDF adds exact
author/series/title overlap — orthogonal to the autoencoder's behavioural co-occurrence signal —
whereas bge's semantic genre similarity *overlaps* what the VAE already captures.
**Fusion helps only when the added signal is complementary, not redundant.**

## Individual models (✚ strong / ✖ weak)

- **Popularity** (baseline) — recommends the globally most-read books. ✚ trivial, hard to beat on accuracy-only protocols. ✖ zero personalization, pure bestseller bias (pop-pctile 1.000), coverage ~0.0002.
- **SVD** (CF, matrix factorization) — factorizes the user×item matrix; ranks by latent co-read patterns. ✚ captures "people like you also read…"; **best on full-catalog** (0.024) and uniform negatives. ✖ **pathologically popularity-biased** (pop-pctile 0.999) — essentially a bestseller machine; blind to cold/tail items, collapses on the popularity-matched headline (0.179).
- **TF-IDF + cosine** (content) — similarity over IDF-weighted text (title+author+plot+shelves). ✚ sharp on exact author/series/title overlap; **best standalone content vectorizer** (0.328 uniform). ✖ lexical only — misses same-vibe books that use different words.
- **BoW + cosine** (content) — same text, raw counts, no IDF. ✚ simplest possible content baseline. ✖ ubiquitous tokens dominate → **~4× worse than TF-IDF** (0.085).
- **bge embeddings** (`content_emb` mean-pool / `max-sim` max-pool) — dense *semantic* similarity. ✚ **reaches the tail** (pop-pctile 0.586), semantic discovery, robust for sparse users; `max-sim` (per-book nearest, no centroid drift) is the best standalone content model on the headline (0.243) and the **"more like this" workhorse** (UC4 MRR 0.25). ✖ blurs exact author/series matches → weaker raw accuracy; `content_emb` mean-pooling drifts to the popular centroid (≈0 full-catalog).
- **Mult-VAE** (autoencoder) — reconstructs the user's whole interaction vector through a non-linear bottleneck (multinomial likelihood, β-annealed KL). ✚ **best single model** on the headline (0.303); **widest coverage by far** (0.040) and de-biasable via α — the least bestseller-biased CF model. ✖ popularity-collapses without the α knob; weak on brutal full-catalog single-item ranking (0.004) precisely *because* it's de-biased; vocabulary-bound (can't score cold items).
- **SASRec** (sequential transformer, RecBole) — treats history as an *ordered* sequence (self-attention). ✚ captures order (sequels/series). ✖ **does not beat classical SVD** (full-catalog 3rd, 0.015); **40% user-coverage gap** (in-vocabulary only); framework-heavy, not serving-friendly. Stands in for the RNN (GRU4Rec).

## Fusion & re-ranking experiments (✚ strong / ✖ weak)

- **RRF (Reciprocal Rank Fusion)** — the fusion *mechanism*: combine components by *rank* (∑ weight/(k+rank)), not raw score. ✚ fuses incomparable score scales and different catalog coverage cleanly — a narrow-but-strong model re-ranks the head, a full-coverage model supplies the tail. ✖ throws away score magnitude (rank-only); needs a fusion weight.
- **Mult-VAE ⊕ TF-IDF max-sim** (RRF) — **best overall (0.304–0.317)**. ✚ adds TF-IDF's *orthogonal* exact author/series/title overlap on top of the autoencoder's behavioural co-occurrence → genuine lift over Mult-VAE alone. ✖ helps **only** because the signal is complementary; needs a tuned fusion weight.
- **Mult-VAE ⊕ bge max-sim** (RRF) — the **negative control**. ✚ — in principle adds semantic content. ✖ bge's genre similarity *overlaps* what the VAE already learns → **no lift** (0.299, *below* the VAE alone). Proves fusion needs complementarity, not just "more signal."
- **Mult-VAE ⊕ SVD** (RRF) — **two collaborative models fused** *(numbers pending the next nb08 run)*. ✚ hypothesis: SVD's full-catalog/popularity strength + the VAE's headline/coverage strength → a model robust across *both* protocols. ✖ likely **redundant** (same CF paradigm) — SVD's bestseller pull may drag the de-biased VAE back toward popularity on the headline. The experiment that tests "is two-CF fusion complementary or redundant?".
- **Learned hybrid: SVD ⊕ content** (stacking — logistic reranker over each component's score) — `hybrid_cf_content`. ✚ *learns* the blend from data; interpretable coefficients double as a "which paradigm contributes" experiment (cf ≫ content, ~15:0.5). ✖ the CF feature dominates, so it **inherits SVD's popularity bias** (pop-pctile ~0.999); no headline edge (0.214); sklearn-version-fragile (pickled).
- **Learned hybrid + popularity feature** (`hybrid_cf_content_pop`) — adds an explicit popularity score as a third feature. ✚ **tops the uniform-negative protocol** (0.622). ✖ explicitly amplifies bias; protocol-specific win only.
- **Learned hybrid, popularity-sampled negatives** (`hybrid_cf_content_popneg`) — same stack, trained against *popular* negatives. ✚ that harder training **de-biases the reranker** (cf weight 15→0.9) → better on the popularity-matched protocol (0.214 vs 0.187). ✖ trades raw accuracy for de-biasing (lower on uniform/full-catalog).
- **α popularity discount** (inference re-rank, not a fusion) — subtract `α·log(popularity)` from every score. ✚ **cheapest, biggest single lever** — *more* accuracy **and** *less* bias at once (Mult-VAE 0.171→0.322; bias 0.986→0.722). ✖ protocol-dependent (helps popularity-matched, hurts uniform/full-catalog).

*(Two more re-ranking levers have their own sections below: **recency weighting** (UC2) and **event-level weighting** (like > want, Kunlun-inspired).)*

## Req 3 — TF-IDF vs BoW (content, same text fields; uniform-negatives run)

| vectorizer (title+author+plot+shelves, cosine) | NDCG@10 |
|---|---|
| **TF-IDF** | **0.328** |
| embeddings (bge-small) | 0.169 |
| **BoW** | 0.085 |

**TF-IDF beats BoW ~4×** — IDF down-weights ubiquitous tokens; raw counts (BoW) drown in them.

## Req 5 — Neural vs classical (full-catalog, same users)

Full-catalog ranks against the whole catalog (tiny numbers by design — read the *order*). Scored
on the 1810/3000 users SASRec covers (RecBole only predicts for in-vocabulary users — a real
coverage gap; the others cover everyone).

| model | NDCG@10 | Recall@10 |
|---|---|---|
| **SVD (CF, classical)** | **0.024** | 0.045 |
| learned hybrid (SVD ⊕ content) | 0.017 | 0.036 |
| SASRec (sequential transformer\*) | 0.015 | 0.036 |
| Mult-VAE (autoencoder) | 0.004 | 0.009 |
| content_emb | 0.000 | 0.000 |

**Neural does *not* win full-catalog.** SASRec lands 3rd — below classical SVD *and* the learned
hybrid — and can't score 40% of users. Mult-VAE is low here *because* it's de-biased (α=1): the same
property that won it the popularity-matched headline (0.30) sinks it on the popularity-rewarding
full-catalog protocol. The **protocol decides the winner**, not "neural vs classical."

\*SASRec stands in for the RNN (GRU4Rec): same sequential task.

## Anti-popularity (full-catalog diagnostics)

Mean popularity-percentile of recs (1.0 = always the top bestseller) + catalog coverage:

| model | pop-pctile ↓ | coverage ↑ |
|---|---|---|
| Popularity / SVD / learned hybrid | ~1.00 | ~0.003 |
| Mult-VAE (α=1) | 0.722 | **0.040** |
| content_emb | **0.586** | 0.007 |

The α discount lifts Mult-VAE NDCG **0.171 → 0.322** *and* cuts its bias (0.986 → 0.722).

## Takeaways

- **Mult-VAE (α=1) wins the popularity-robust headline** *and* has by far the widest coverage (0.04)
  — the least bestseller-biased CF model.
- **SVD is accurate but pathologically popularity-biased** (pop-pctile 0.999) — it essentially
  recommends bestsellers, so it wins only on random-negative / full-catalog protocols.
- **Content reaches the tail** (content_emb pop-pctile 0.586) but is weak on raw accuracy;
  **TF-IDF ≫ BoW**.
- **The protocol decides the winner** — SVD/SASRec lead full-catalog, Mult-VAE/content lead the
  popularity-matched headline (Krichene & Rendle 2020). We report on the popularity-robust headline.
- **Fusion beats the best single model — but only with a *complementary* signal.** Mult-VAE ⊕
  **TF-IDF** max-sim (0.317) > Mult-VAE alone (0.305); Mult-VAE ⊕ **bge** does not. Complementarity
  (lexical exact-match, orthogonal to the VAE) matters more than raw component strength — the weaker
  standalone content model made the stronger hybrid.

## Use-case levers (UC2 recency, UC4 similar-to-anchor)

**UC2 — recency-weighted vs flat history** (exponential decay, τ=5; NDCG@10):

| base model | flat | recency-weighted |
|---|---|---|
| content_emb | 0.0017 | **0.0239** (~14×) |
| SVD | 0.0274 | **0.0426** (+55%) |

Down-weighting old interactions is a large, cheap win — flat mean-pooling drifts the profile toward
the popular centroid; recency recovers the *recent* taste. Validates the **locality bias** (recent
events carry more signal), the same premise behind *Kunlun*'s temporal embeddings / sliding window.

**UC4 — "more like this" (content neighbours vs co-read ground truth):** Recall@10 **0.093**,
NDCG@10 **0.116**, MRR **0.251** — ~1000× the content full-catalog numbers. Content embeddings are
weak at predicting a user's next book from scratch but **strong at item–item similarity**: right
tool for the right job.

## Event-level weighting (inspired by *Kunlun*, Meta 2026)

Kunlun's **event-level personalization** allocates more model capacity to high-value events (a
purchase outweighs an impression). We borrow the *idea* at inference, not the systems machinery:
the swipe UI distinguishes a **♥ "read & liked"** from a **🔖 "want to read"**, so we treat them as
**different-strength positives**. Each history item carries a weight (like = 1.0, want = 0.4) that
scales its entry in the Mult-VAE's input interaction vector (a weighted multi-hot) and its column
in the max-sim similarity — so a liked book pulls the recommendations harder than a merely-wanted
one. It reuses the same per-history `weights` channel as the recency lever and is gated by a
`weight_aware` flag, so plain recommenders are unaffected.

*Caveats: pickled sklearn models are version-fragile (refit on the target runtime); SASRec via
RecBole isn't serving-friendly — deployment uses the package-native Mult-VAE / content / hybrid.*
