# Model Report — Book Recommender

**Maya Deneva · UCSD Goodreads.** Items are **works** (editions collapsed: 701k → 468k). Sample:
30k users / 2.27M interactions. Task: history → next book (leave-last-1-out).
**Headline metric:** NDCG@10 / Recall@10 on **popularity-matched sampled negatives** (1 positive vs
100 popular decoys, N=2000; random ≈ 0.10 NDCG@10).

## Models & results

| model | type | NDCG@10 | Recall@10 | strong at | weak at |
|---|---|---|---|---|---|
| Popularity | baseline | ‹fill› | ‹fill› | cheap, hard to beat | only bestsellers |
| **SVD** | CF (matrix factorization) | ‹fill› | ‹fill› | co-read behaviour | sparse users, tail |
| **TF-IDF + cosine** | content (title+author+plot+shelves) | ‹fill› | ‹fill› | tail / cold items | pure behaviour signal |
| **BoW + cosine** | content (same fields) | ‹fill› | ‹fill› | simple | no IDF → ~≤ TF-IDF |
| max-sim (embeddings) | content | ~0.237 | ~0.37 | tail, robust for sparse users | below de-biased CF |
| **Mult-VAE** (α=0) | autoencoder | 0.171 | 0.322 | non-linear CF | popularity-collapsed |
| **Mult-VAE** (α=1) | autoencoder + pop-discount | **0.287** | **0.474** | best accuracy + de-biasable | needs the α knob |
| Hybrid (Mult-VAE ⊕ content) | ensemble (RRF) | ‹fill› | ‹fill› | de-popularizes the VAE | dilutes the best ranking |

‹Fill SVD / TF-IDF / BoW / Popularity / Hybrid from `study_sampled_popneg.csv`.›

## Neural — full-catalog (same users; SASRec only comparable here)

SASRec's predictions are full-catalog, so this table ranks against all ~248k items (tiny absolute
numbers by design — read the **order**). `study_neural.csv`.

| model | NDCG@10 | Recall@10 |
|---|---|---|
| SASRec (sequential transformer) | 0.0174 | 0.0395 |
| Mult-VAE (autoencoder) | ‹fill› | ‹fill› |
| SVD / content | ‹fill› | ‹fill› |

*SASRec stands in for the RNN (GRU4Rec): same sequential task.*

## Anti-popularity (project goal)

The popularity discount **α** is the strongest lever: it lifts Mult-VAE NDCG@10 **0.171 → 0.287**
*and* de-biases it (mean popularity-percentile of recs **0.986 → 0.739**; 1.0 = always the top
bestseller). Content goes furthest into the tail:

| model | pop-percentile ↓ | coverage ↑ |
|---|---|---|
| Mult-VAE (α=1) | 0.739 | 0.015 |
| max-sim (content) | **0.577** | 0.010 |
| Hybrid | 0.704 | 0.013 |

## Takeaways

- **Mult-VAE (α=1) is the best single model** on the headline; the α popularity-discount is a cheap
  inference-time win — more accuracy *and* less bestseller bias.
- **Content (TF-IDF / embeddings) owns the tail** and is most robust for sparse-history users; ‹CF
  vs content order once filled›.
- **Hybrid de-popularizes but doesn't beat the best component** — RRF fusion dilutes a strong ranker.
- ‹SASRec vs Mult-VAE order once §neural filled›.

*Caveats: sampled vs full-catalog metrics aren't interchangeable (Krichene & Rendle 2020); pickled
sklearn models are version-fragile (refit on the target runtime); SASRec via RecBole isn't
serving-friendly — deployment uses the package-native Mult-VAE / content / hybrid.*
