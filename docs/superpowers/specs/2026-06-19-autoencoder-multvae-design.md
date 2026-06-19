# Mult-VAE Autoencoder Recommender — Design

**Date:** 2026-06-19 · **Course scope:** P1 neural CF (the "Mult-VAE" cell of the 3-paradigm benchmark)
**Status:** approved design → implementation

## 1. Goal & motivation

Add the **autoencoder** collaborative-filtering model named in the P1 paradigm: **Mult-VAE**
(Liang et al., 2018, *Variational Autoencoders for Collaborative Filtering*). It must:

- Recommend books from a user's reading history (UC1: history → next book).
- Optimize the headline metrics the report cares about: **NDCG@10 and Recall@{5,10,20}**.
- Resist the "only recommends bestsellers" failure mode — i.e. score well on **serendipity@k
  and catalog coverage**, not just accuracy. The multinomial likelihood is inherently
  tail-friendly; a tunable inference-time popularity-discount knob lets us push further.
- Slot into the existing `Recommender` protocol and eval harness with **zero changes** to either,
  so results land directly in the existing UC1 tables and are **directly comparable to SASRec**.

## 2. Why Mult-VAE (and how it differs from SVD)

Both SVD and Mult-VAE are "fold-in" recommenders: neither stores a per-user vector, so a
brand-new history is scored at inference with no retraining. The *math* differs fundamentally:

| | `SvdRecommender` | `MultVaeRecommender` |
|---|---|---|
| User input | mean of read books' latent factors | full multi-hot 0/1 vector over all items |
| Transform | linear (dot product) | nonlinear MLP: encode → sample → decode |
| Models "A+B together ≠ A + B apart"? | no | yes |
| Output | rank by `factors · user_mean` | reconstruct a score for **every** item at once |

The nonlinear bottleneck is exactly what lets Mult-VAE beat linear CF on NDCG *and* tail
behavior — it is not constrained to recommend near the centroid of what you've read.

## 3. Model (`book_recsys/models/autoencoder/model.py`)

`MultVAE(nn.Module)` — pure torch, device-agnostic (caller moves it to MPS/CPU):

- **Encoder** MLP `I → hidden → 2K`, `tanh` activation, **input dropout** `p` (the denoising
  corruption). Input vector is L2-normalized per user before the first layer (paper convention).
- **Reparameterize** `z = μ + ε·σ`, `ε ~ N(0,I)` (training only; inference uses `μ`).
- **Decoder** MLP `K → hidden → I`, `tanh` hidden, **logits** out (no final activation).
- **Loss** = multinomial NLL `-Σ_i x_i · log softmax(logits)_i` + `β · KL(q(z|x) ‖ N(0,I))`.
- `forward(x)` returns `(logits, μ, logσ²)`; `loss(x, logits, μ, logσ², beta)` returns the scalar.

Default config (paper-standard, all overridable): `I = vocab size`, `hidden = 600`, `K = 200`,
`dropout = 0.5`, `β_cap = 0.2`.

## 4. β-annealing schedule (`train.py`)

Linear warm-up of the KL weight, the paper's key trick:
`beta(step) = β_cap · min(1.0, step / anneal_steps)`. Pure function `anneal_beta(step,
anneal_steps, beta_cap) -> float`, unit-tested independently of the training loop.

## 5. Recommender wrapper (`book_recsys/models/autoencoder/recommender.py`)

`MultVaeRecommender` implements the `Recommender` protocol:

- `__init__(..., hidden=600, latent=200, dropout=0.5, beta_cap=0.2, epochs, batch_size, lr,
  anneal_steps, min_item_count=1, pop_discount=0.0, device=None, seed=42)`.
- `fit(train_df)`:
  1. Build item vocab from `train_df` (items with ≥ `min_item_count` interactions; default 1 =
     full vocab). Store `_ids` (vocab list) and `_pos` (id→index), like `SvdRecommender`.
  2. Build the sparse user×item **binary** matrix (implicit feedback; ratings ignored — a read is
     a positive). Deterministic row/col ordering.
  3. Compute per-item popularity (interaction counts) for the popularity-discount knob.
  4. Train `MultVAE` via `train.py` (see §6), store the trained module in eval mode.
- `recommend(history, k)`:
  1. Build the 0/1 input vector from `history` over `_pos` (unknown ids dropped), L2-normalize.
  2. One forward pass using `μ` → logits over all items.
  3. **Mask seen** items (`-inf`), apply popularity discount `score_i -= α · log(pop_i)`
     (`α = pop_discount`, `α=0` → vanilla), return top-k ids.
- `score_items(history, item_ids)`: forward once, return the logits (post-discount) for the given
  ids — drives `evaluate_sampled_negatives`. Unknown ids → `-inf`.

The wrapper is the only public entry point; everything else (matrix build, training) is internal.

## 6. Training loop (`book_recsys/models/autoencoder/train.py`)

`train_multvae(model, matrix, *, epochs, batch_size, lr, anneal_steps, beta_cap, device,
val_matrix=None, seed) -> model`:

- Adam optimizer; iterate user-row minibatches; per step compute `beta = anneal_beta(...)`,
  forward, `loss`, backward, step.
- Device: caller passes `"cuda"` (Kaggle / NVIDIA laptop) / `"mps"` / `"cpu"`; default auto-detects
  (cuda → mps → cpu).
- **AMP/fp16** toggle (`amp=True`) for CUDA — ~2× faster on Kaggle T4, big-matmul friendly.
- **Checkpoint everything, lose nothing** (explicit requirement):
  - write `multvae_last.pt` **every epoch** (atomic temp-file rename so a mid-write kill can't
    corrupt it) — a Kaggle GPU cutoff at any epoch keeps all progress;
  - track val NDCG@10 and write `multvae_best.pt` whenever it improves;
  - each checkpoint stores `state_dict` + full config + epoch + optimizer state + the vocab
    (`_ids`/`_pos`), so a resume or a fresh `MultVaeRecommender.load` is exact and self-contained;
  - `--resume`/`RESUME` continues from `multvae_last.pt` (restores optimizer + epoch + RNG);
  - all checkpoints land in `artifacts/` (gitignored, so big weights never hit git) and, on Kaggle,
    are written to `/kaggle/working/` so they persist as downloadable run output.
- Optional early stop on validation NDCG@10 (held-out interactions) if `val_matrix` given.
- Deterministic: seed torch + numpy RNG.
- `save(model, path)` / `load(path, device)` checkpoint helpers (state_dict + config) →
  `artifacts/multvae.pt`.

**Runner = notebook (project convention).** All logic lives in the package; the **notebook
`notebooks/10_autoencoder.ipynb` is the runner**, exactly like `06_recbole.ipynb`. A top **config
cell** sets the knobs (`N_USERS=30000`, `MAX_HIST=100`, `LATENT=200`, `HIDDEN=600`, `BETA_CAP=0.2`,
`DROPOUT=0.5`, `LR`, `EPOCHS`, `DEVICE`, `AMP`, `MIN_ITEM_COUNT`); cells below call the package
(`build_matrix` → `train_multvae` → `MultVaeRecommender` → eval harness). Re-running an experiment =
edit the config cell, re-run. The same notebook runs on Kaggle (cuda+amp) and the NVIDIA laptop. A
thin `scripts/train_multvae.py` mirroring the notebook is **optional** (headless/resume convenience),
not the primary path.

## 7. Data — same 30k users as SASRec (exact reproduction)

To get a **fair Mult-VAE vs SASRec** row, reproduce SASRec's training set bit-for-bit from
`artifacts/sample.parquet` (the materialized 50k-user work-collapsed artifact):

```python
N_USERS, MAX_HIST, SEED = 30000, 100, 42
keep = sample["user_id"].drop_duplicates().sample(N_USERS, random_state=SEED)   # pandas RNG
sample = sample[sample["user_id"].isin(keep)]
sample = (sample.sort_values(["user_id", "timestamp"])
                .groupby("user_id", sort=False).tail(MAX_HIST).reset_index(drop=True))
assert len(sample) == 2_273_496   # proves identity with recbole_data/goodreads/goodreads.inter
```

This yields **30,000 users · 2,273,496 interactions · 248,627 items**. Lives in
`book_recsys/models/autoencoder/data.py` (`reproduce_sasrec_sample(sample_df)`), unit-testable and
called from the notebook; the model/training code stays agnostic and trains on whatever matrix it's
handed.

**Vocab feasibility:** full 248,627-item output layer ≈ 300M params (~1.2 GB fp32, ~3.6 GB with
Adam state) — trainable on an M4 but tight. **Default = full vocab** (fair to SASRec). If MPS
memory/throughput bites, `min_item_count` (e.g. 5) caps the vocab; items below threshold then
can't be scored by the AE — a known autoencoder limitation we report honestly.

**Splits:** leave-last-1-out per user (matches the report's UC1 protocol and SASRec).

## 8. Serendipity / anti-popularity

Two layers, both reported:

1. **Inherent** — the multinomial likelihood normalizes probability across items rather than
   rewarding each popular item independently, so Mult-VAE is structurally less head-biased than
   pointwise models. Measured via `serendipity_at_k`, `popularity_diagnostics` (coverage,
   mean-pop-percentile), `intra_list_diversity`.
2. **Tunable** — inference-time `score_i -= α · log(pop_i)`. Sweeping `α` traces the
   accuracy↔serendipity tradeoff (a report figure). `α=0` is vanilla Mult-VAE.

## 9. Evaluation & experiments

Reuse the harness untouched:

- **Headline:** popularity-matched sampled-negative **NDCG@10 / Recall@{5,10,20} / MRR**, on the
  **same user draw as SASRec/max-sim** (`evaluate_sampled_negatives` with popularity weights).
- **Anchor:** full-catalog ranking (honesty anchor).
- **Diagnostics:** serendipity@10, coverage, mean-pop-percentile, intra-list diversity.

**Hyperparameter experiments — manual, results-driven, NOT a pre-baked grid.** Run **one baseline
first** (K=200, hidden=600, β_cap=0.2, dropout=0.5 — paper default), watch how training/validation
NDCG behaves, then decide which knob to try next by hand. No blind sweeping. The implementation's
job is to make this cheap: **every hyperparameter is a variable in the notebook's config cell**
(`LATENT`, `HIDDEN`, `BETA_CAP`, `DROPOUT`, `LR`, `EPOCHS`) so a re-run is edit-cell-and-rerun, and
each run's config + metrics are appended to a results log so manual comparisons stay honest and
reproducible.

Cheap experiments to reach for once the baseline is in:
- `BETA_CAP = 0.0` → collapses Mult-VAE to a denoising AE (the variational-vs-not ablation).
- the **α serendipity sweep** `α ∈ {0, 0.25, 0.5, 1.0}` — **inference-only** re-ranking on the
  baseline checkpoint, **zero extra training** — always worth running for the tail story.

Report the baseline + whatever variations get tried; results append to `reports/model_report.md`
UC1 tables and a new "Mult-VAE" subsection.

## 10. Package layout & deps

```
book_recsys/models/autoencoder/
  __init__.py
  data.py          # reproduce 30k-user sample (assert 2,273,496), build_matrix
  model.py         # MultVAE(nn.Module): forward, loss
  train.py         # anneal_beta, train_multvae, checkpoint save/load/resume
  recommender.py   # MultVaeRecommender (Recommender protocol)
notebooks/10_autoencoder.ipynb  # RUNNER: config cell → train → eval → figures (Kaggle + laptop)
scripts/train_multvae.py        # OPTIONAL thin headless mirror of the notebook
```

`torch` added to `pyproject.toml` as optional dep `[autoencoder]` (mirrors `[recbole]`); core
install stays light. Checkpoint → `artifacts/multvae.pt`.

## 11. Testing (TDD, 100% coverage on new modules)

Tests written **before** implementation (`tests/models/autoencoder/`):

- `anneal_beta`: 0 at step 0, `β_cap` past `anneal_steps`, linear in between, clamps.
- `MultVAE.loss`: multinomial NLL term vs hand-computed value on a 2-item toy; KL term vs
  closed form; `β` scales the KL contribution.
- vocab + sparse-matrix build: correct shape, binary, `min_item_count` filtering, deterministic.
- `recommend`: masks seen items; popularity-discount reorders toward niche items as `α↑`; unknown
  history ids dropped; returns ≤ k ids.
- `score_items`: known ids get finite scores, unknown → `-inf`; discount applied.
- **End-to-end** (small fixture, 6 users × 8 items): train 2 epochs on MPS/CPU → assert a held-out
  positive ranks above a never-seen cold item; checkpoint save→load round-trips identical scores.

Heavy training paths are covered via the tiny fixture (same approach as existing `test_e2e_*`).

## 12. Risks & mitigations

- **MPS memory** with 248k-item output → `min_item_count` cap fallback; fp32; reasonable batch.
- **Exact-data drift** vs SASRec → the `assert len == 2_273_496` guard makes any drift loud.
- **Cold items** (below vocab threshold) unscoreable → reported as a limitation; default full vocab
  avoids it entirely for the headline comparison.

## 13. Workflow

New `autoencoder` git worktree → writing-plans → TDD implementation → train → eval → report.
