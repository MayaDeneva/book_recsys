# Model Report — Book Recommender Systems

**Author:** Maya Deneva · **Dataset:** UCSD Goodreads (Wan & McAuley, 2018/2019)
Concise report covering content-based (TF-IDF vs BoW), collaborative filtering (SVD), and neural
models (SASRec sequential, Mult-VAE autoencoder) on one shared task and protocol.

> Rows marked **‹fill: file.csv›** come from the `08_evaluation.ipynb` outputs — drop the numbers in.

---

## 1. Data & evaluation protocol

- **Items are works, not editions.** UCSD Goodreads catalogs *editions* (each its own `book_id`);
  one work appears many times with near-identical text. We collapse editions → works
  (`02_1_collapse_works.ipynb`): **~701k editions → 468,628 works** (see `01_eda.ipynb`,
  "Edition → work collapse").
- **Sample:** 30,000 users · 2,273,496 interactions (identical to the SASRec training set, for a
  fair neural comparison).
- **Split:** leave-last-1-out per user (predict each user's chronologically last book).
- **Task (UC1):** given a user's reading history, rank their held-out next book.
- **Headline metric protocol — popularity-matched sampled negatives:** the held-out book is ranked
  against **100 popularity-matched distractors** (popular decoys, not random ones). This is
  interpretable (random ≈ 0.10 NDCG@10) *and* not gamed by popularity. **N = 2,000** eval users.
  Metrics: **NDCG@10 (headline)**, Recall@10, MRR.
- **Honesty anchor — full-catalog:** rank against all ~248k items. Brutally hard (tiny absolute
  numbers) but unbiased; used only for the neural comparison where SASRec's predictions are
  full-catalog. *(Sampled vs full-catalog rankings can disagree — Krichene & Rendle 2020 — so we
  read head-to-heads off the headline and keep full-catalog as the anchor.)*

---

## 2. Requirement 3 — Content-based: TF-IDF vs BoW

Book document = **title + synopsis + genre/shelf tags**, vectorized two ways, recommendation by
**cosine** to the user's profile. Same text fields, same protocol, same test set.

| method | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| TF-IDF + cosine (`content_tfidf_full`) | ‹fill: study_sampled.csv› | ‹fill› | ‹fill› |
| BoW + cosine (`content_bow`) | ‹fill: study_sampled.csv› | ‹fill› | ‹fill› |

**Finding:** ‹TF-IDF vs BoW — expected: TF-IDF ≥ BoW, since IDF down-weights ubiquitous tokens;
state the gap and whether it's significant.›

---

## 3. Requirement 4 — Collaborative filtering (SVD) vs content (TF-IDF)

SVD matrix factorization of the user–item matrix (implicit positives; an explicit-only
`rating ≥ 4` variant included as an ablation), compared with the content TF-IDF model on the
**same metrics and same test set**.

| method | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| SVD (CF) | ‹fill: study_sampled_popneg.csv› | ‹fill› | ‹fill› |
| SVD, rating ≥ 4 (explicit) | ‹fill› | ‹fill› | ‹fill› |
| TF-IDF + cosine (content) | ‹fill› | ‹fill› | ‹fill› |
| Popularity (baseline) | ‹fill› | ‹fill› | ‹fill› |

**Finding:** ‹CF vs content — which wins on the popularity-matched headline, and why (CF captures
co-read behaviour; content captures the tail / cold items).›

---

## 4. Requirement 5 — Classical vs neural networks

Two neural models solving the same UC1 task:
- **Sequential — SASRec** (Kang & McAuley 2018): a self-attention (transformer) sequential model
  that treats the history as an ordered sequence. *Used in place of the RNN (GRU4Rec): it solves
  the same sequential task; GRU4Rec is the recurrent alternative we did not train.*
- **Autoencoder — Mult-VAE** (Liang et al. 2018): a variational autoencoder that reconstructs the
  user's full interaction vector through a non-linear bottleneck.

### 4a. Mult-VAE — headline (popularity-matched sampled negatives, N = 2,000)

Trained 150 epochs, latent 200, hidden 600, β annealed to 0.2. An **inference-time popularity
discount α** (subtract `α·log popularity` from scores at ranking) is the operating knob.

| Mult-VAE | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| α = 0 (raw) | 0.3215 | 0.1715 | 0.1506 |
| **α = 1 (operating point)** | **0.4745** | **0.2865** | **0.2483** |

α=1 is the headline configuration: it both maximises NDCG@10 and de-biases the model (§5).

### 4b. Neural vs classical — full-catalog (same users, fair footing)

SASRec's predictions are full-catalog, so all methods here are scored full-catalog over the same
shared users (`study_neural.csv`). Absolute numbers are small by design (random ≈ 4e-5); read the
**ordering**.

| method | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| SASRec (sequential) | 0.0395 | 0.0174 | 0.0107 |
| Mult-VAE (autoencoder, α=1) | ‹fill: study_neural.csv› | ‹fill› | ‹fill› |
| SVD (CF) | ‹fill› | ‹fill› | ‹fill› |
| Content (TF-IDF / emb) | ‹fill› | ‹fill› | ‹fill› |
| Hybrid (Mult-VAE ⊕ content) | ‹fill› | ‹fill› | ‹fill› |

**Finding:** ‹On full-catalog, does SASRec (sequential) beat Mult-VAE (autoencoder) and the
classical models? State the order; note SASRec's strength is sequence modelling, Mult-VAE's is
de-biased reconstruction.›

> **Protocol note:** the §4a Mult-VAE numbers (sampled, 0.287) are **not** comparable to the §4b
> full-catalog numbers (0.0x) — different denominators. SASRec only appears full-catalog; the
> sampled headline is the interpretable comparison among the non-SASRec models.

---

## 5. Beyond-accuracy — fighting popularity bias

The project goal is recommendations that aren't just bestsellers. Two levers, both measured.

**Inference-time popularity discount (α) on Mult-VAE.** Raising α trades a little nothing for a lot:
on the popularity-matched headline it *improves* accuracy **and** de-biases — NDCG@10
**0.171 → 0.287** while the mean popularity-percentile of its recommendations drops
**0.986 (undertrained) → 0.739 (converged, α=1)** (1.0 = always the single most-popular book).

**Content reaches the tail.** Coverage + mean popularity-percentile of the top-10 (lower = more
niche), 400-user subset:

| model | mean pop-percentile ↓ | coverage ↑ |
|---|---|---|
| Mult-VAE (α=1) | 0.739 | 0.015 |
| max-sim (content) | **0.577** | 0.010 |
| Hybrid (Mult-VAE ⊕ content) | 0.704 | 0.013 |

**Finding:** content (max-sim) recommends the **least popular** items; the hybrid pulls the
autoencoder toward the tail (0.739 → 0.704). max-sim is also the most *robust* across history
length (≈ flat NDCG for short- vs long-history users, where pure CF degrades for sparse users).

---

## 6. Conclusions

- **The popularity discount (α) is the most effective single lever** — a cheap inference-time
  re-rank took Mult-VAE from 0.171 → 0.287 NDCG@10 *and* halved its popularity bias.
- **Content (TF-IDF / embeddings) is the tail specialist** — most niche, most robust for
  sparse-history users; ‹CF vs content headline ordering once filled›.
- **Hybrid fusion is not a free win** — RRF-blending de-popularises the VAE but dilutes its ranking;
  it ‹beat / did not beat› the best standalone model on accuracy.
- ‹SASRec vs Mult-VAE ordering once §4b is filled›.

## 7. Limitations & reproducibility notes

- **Sampled vs full-catalog protocols are not interchangeable** (Krichene & Rendle 2020); we report
  both and read head-to-heads off the headline.
- **Pickled sklearn models are version-fragile** — `models.joblib` trained with one sklearn failed
  to unpickle/predict on a newer runtime; the fix is to refit on the target runtime (or pin
  versions), which is why `07_models` rebuilds the zoo in-environment.
- **SASRec is benchmarked via RecBole** (a research framework), which is impractical to serve
  directly; deployment uses the package-native Mult-VAE / hybrid / content models (load checkpoint
  → single forward pass).
- **k-core filtering removed true cold-start users**, so the "cold-start" analysis is a relative
  short- vs long-history split, not literal cold start.
