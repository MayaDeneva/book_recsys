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

## What each model does (✚ strength / ✖ weakness)

- **Popularity** (baseline) — recommends the globally most-read books. ✚ trivial, hard to beat. ✖ zero personalization, pure bestseller bias.
- **SVD** (CF) — factorizes the user×item matrix; ranks by latent co-read patterns. ✚ captures "people like you also read…". ✖ **pathologically popularity-biased** (pop-pctile 0.999), blind to cold/tail items and sparse users.
- **TF-IDF + cosine** (content) — similarity over IDF-weighted text. ✚ sharp on exact author/series/title overlap; **best standalone content vectorizer**. ✖ lexical only — misses same-vibe books with different words.
- **BoW + cosine** (content) — same, raw counts, no IDF. ✚ simplest. ✖ generic tokens dominate → ~4× worse than TF-IDF.
- **bge embeddings** (`content_emb` / `max-sim`) — dense *semantic* similarity. ✚ **reaches the tail** (most niche, pop-pctile 0.586), semantic discovery, robust for sparse users; max-pooled (`max-sim`) is the best standalone content model on the headline. ✖ blurs exact author/series matches → weaker raw accuracy; mean-pooling drifts to the centroid.
- **Mult-VAE** (autoencoder) — reconstructs the user's full interaction vector through a non-linear bottleneck. ✚ **best single model** on the headline; de-biasable via α; widest catalog coverage. ✖ popularity-collapses without α; weak on brutal full-catalog single-item ranking.
- **SASRec** (sequential transformer) — treats history as an ordered sequence (self-attention). ✚ captures order (sequels/series). ✖ framework-heavy (RecBole), not serving-friendly; full-catalog only here.
- **Hybrid: Mult-VAE ⊕ TF-IDF max-sim** (RRF) — **best overall (0.317)**. ✚ fuses the autoencoder's behavioural signal with TF-IDF's orthogonal exact-overlap. ✖ only helps with a *complementary* signal (bge fusion doesn't); needs a tuned fusion weight.
- **Learned hybrid** (CF+content stacking, logistic reranker) — learns to blend paradigms. ✚ principled feature combination. ✖ no clear edge here; sklearn-version-fragile.
- **α popularity discount** (inference re-rank) — subtract `α·log(popularity)` from scores. ✚ **cheapest, biggest lever** — more accuracy *and* less bias (0.171→0.322). ✖ protocol-dependent (helps on popularity-matched, hurts on uniform/full-catalog).

## Req 3 — TF-IDF vs BoW (content, same text fields; uniform-negatives run)

| vectorizer (title+author+plot+shelves, cosine) | NDCG@10 |
|---|---|
| **TF-IDF** | **0.328** |
| embeddings (bge-small) | 0.169 |
| **BoW** | 0.085 |

**TF-IDF beats BoW ~4×** — IDF down-weights ubiquitous tokens; raw counts (BoW) drown in them.

## Req 5 — Neural vs classical (full-catalog, same users)

Full-catalog ranks against all ~248k items (tiny numbers by design — read the order).

| model | NDCG@10 | coverage |
|---|---|---|
| SVD (CF) | 0.021 | 0.003 |
| SASRec (sequential transformer\*) | 0.017 | — |
| Mult-VAE (autoencoder) | 0.004 | **0.040** |

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

*Caveats: pickled sklearn models are version-fragile (refit on the target runtime); SASRec via
RecBole isn't serving-friendly — deployment uses the package-native Mult-VAE / content / hybrid.*
