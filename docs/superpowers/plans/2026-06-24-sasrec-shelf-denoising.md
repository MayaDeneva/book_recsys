# SASRec shelf-denoising spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether denoising SASRec's *input history* (dropping `rating==0` "added-to-shelf" events while keeping the held-out target fixed) improves next-item prediction, as a directly-comparable row in the existing study tables.

**Architecture:** A new tested package function (`denoise_history_keep_targets`) cleans each user's history but preserves their last two interactions (the leave-one-out valid/test targets), so the eval target set is byte-identical to the raw run. `06_recbole.ipynb` calls it behind a `DENOISE_HISTORY` flag, equalizes the user set across runs, trains SASRec on cloud GPU, and scores through the existing shared harness. Result is a "raw vs shelf-denoised" table in `reports/model_report.md`.

**Tech Stack:** Python 3.12, pandas, RecBole (SASRec, CUDA-only — cloud GPU), the project's existing eval harness.

**Spec:** `docs/superpowers/specs/2026-06-24-sasrec-shelf-denoising-design.md`

## Global Constraints

- **Python 3.12.** Style: `yapf` (column_limit 99), `isort` (line width 99), `mypy` clean.
- **100% test coverage** (`coverage report --fail-under=100`) — the new function must be fully covered.
- **Schema constants only:** import `USER, BOOK, RATING, TS` from `book_recsys.data.schema`; never hardcode column names.
- **Denoise the input history ONLY.** Never drop, reorder, or alter any user's last two interactions — they are the eval targets and must stay identical to the raw run.
- **One lever only:** drop `rating == 0`. No same-day session collapse, no GRU4Rec, no UI/serving work, no `is_read` re-ingest (all out of scope per spec).
- **Equalize the user set:** every compared row (raw-SASRec, denoised-SASRec, baselines) is scored over the *identical* set of users that survive denoising.

---

### Task 1: `denoise_history_keep_targets` package function

**Files:**
- Modify: `book_recsys/data/filters.py`
- Test: `tests/data/test_filters.py`

**Interfaces:**
- Consumes: `book_recsys.data.schema.{USER, RATING, TS}`; existing `filter_min_rating`.
- Produces: `denoise_history_keep_targets(df: pd.DataFrame, min_rating: int = 1, n_targets: int = 2, min_items: int = 3) -> pd.DataFrame` — returns a new frame (index reset) where, per user (sorted by `TS` ascending), a row is kept iff `RATING >= min_rating` **or** it is among that user's last `n_targets` interactions; users left with fewer than `min_items` rows are dropped entirely. Input is not mutated; per-user time order preserved.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_filters.py` (reuse the existing `_df` helper for single-user cases; add a multi-user helper):

```python
from book_recsys.data.filters import denoise_history_keep_targets


def _multi(rows):
    # rows: list of (user, rating, ts)
    return pd.DataFrame([{USER: u, BOOK: f"b{i}", RATING: r, TS: t}
                         for i, (u, r, t) in enumerate(rows)])


def test_denoise_drops_zero_rating_history_events():
    # history events b0 (r0) dropped; b1,b2 (rated) kept; last two (b3,b4) always kept
    out = denoise_history_keep_targets(_df([0, 3, 0, 4, 0]), min_rating=1, n_targets=2)
    assert list(out[RATING]) == [3, 4, 0]  # b1, b3, b4 — b0 & b2 dropped, b4 target kept


def test_denoise_preserves_last_n_targets_even_if_zero_rating():
    # the final two events are rating 0 but are the valid/test targets -> kept
    out = denoise_history_keep_targets(_df([5, 5, 5, 0, 0]), min_rating=1, n_targets=2)
    assert list(out[RATING]) == [5, 5, 5, 0, 0]


def test_denoise_drops_users_below_min_items():
    # u_short: only 1 rated history event + 2 targets would be 3, but here all-zero history
    # leaves just the 2 targets (<3) -> user dropped. u_ok keeps >=3.
    df = _multi([("short", 0, 0), ("short", 0, 1), ("short", 0, 2),
                 ("ok", 4, 0), ("ok", 0, 1), ("ok", 0, 2)])
    out = denoise_history_keep_targets(df, min_rating=1, n_targets=2, min_items=3)
    assert set(out[USER]) == {"ok"}


def test_denoise_keeps_multiple_same_day_ratings():
    # two rated events share a timestamp (same day) -> both kept, order preserved
    out = denoise_history_keep_targets(_df([4, 4, 5, 5]), min_rating=1, n_targets=2)
    assert list(out[RATING]) == [4, 4, 5, 5]


def test_denoise_does_not_mutate_input():
    df = _df([0, 3, 4, 5])
    before = df.copy()
    denoise_history_keep_targets(df, min_rating=1)
    pd.testing.assert_frame_equal(df, before)


def test_denoise_resets_index():
    out = denoise_history_keep_targets(_df([0, 4, 5, 5]), min_rating=1)
    assert list(out.index) == list(range(len(out)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_filters.py -v -k denoise`
Expected: FAIL with `ImportError`/`AttributeError: cannot import name 'denoise_history_keep_targets'`.

- [ ] **Step 3: Implement the function**

Append to `book_recsys/data/filters.py`:

```python
from book_recsys.data.schema import TS, USER


def denoise_history_keep_targets(df: pd.DataFrame,
                                 min_rating: int = 1,
                                 n_targets: int = 2,
                                 min_items: int = 3) -> pd.DataFrame:
    """Drop low-rating events from each user's *history* while preserving their last
    `n_targets` interactions (the leave-one-out valid/test targets) regardless of rating.

    A row is kept iff its rating >= `min_rating` OR it is among the user's last `n_targets`
    interactions by timestamp. Users left with fewer than `min_items` rows are dropped
    entirely (too short for the leave-one-out split). The input is not mutated; per-user
    chronological order is preserved.
    """
    ordered = df.sort_values([USER, TS], kind="stable")
    from_end = ordered.groupby(USER, sort=False).cumcount(ascending=False)
    keep = (ordered[RATING] >= min_rating) | (from_end < n_targets)
    kept = ordered[keep]
    sizes = kept.groupby(USER, sort=False)[RATING].transform("size")
    return kept[sizes >= min_items].reset_index(drop=True)
```

(Hoist the `from book_recsys.data.schema import ...` line to the top of the module with the existing import, merging into `from book_recsys.data.schema import RATING, TS, USER`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_filters.py -v -k denoise`
Expected: PASS (6 tests).

- [ ] **Step 5: Coverage + lint**

Run:
```bash
coverage run -m pytest tests/data/test_filters.py && coverage report --include='*/data/filters.py' --show-missing --fail-under=100
yapf -i book_recsys/data/filters.py tests/data/test_filters.py && isort book_recsys/data/filters.py tests/data/test_filters.py --line-width 99
mypy book_recsys/data/filters.py
```
Expected: 100% on `filters.py`, no style diff, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add book_recsys/data/filters.py tests/data/test_filters.py
git commit -m "feat(data): denoise_history_keep_targets — drop rating-0 history, keep eval targets"
```

---

### Task 2: Wire denoise into `06_recbole.ipynb` + equalize the user set

**Files:**
- Modify: `notebooks/06_recbole.ipynb` (cell 4 — the sample-prep / `.inter`-writing cell)

**Interfaces:**
- Consumes: `denoise_history_keep_targets` (Task 1); existing `leave_last_n_out`, `build_relevance`, `write_inter_file`.
- Produces: a `DENOISE_HISTORY` flag and an equalized `sample` such that the SAME user set is used whether the flag is on or off.

- [ ] **Step 1: Edit cell 4** — insert the equalization + denoise block immediately after the `MAX_HIST` history-cap block and **before** `train, holdout = leave_last_n_out(...)`.

Add this code (uses the already-imported `pandas as pd`; add the import line to the cell's import block at top of the notebook / cell 3):

```python
from book_recsys.data.filters import denoise_history_keep_targets
```

Then, after the history-cap `print(...)` and before `leave_last_n_out`:

```python
# --- Shelf-denoising spike (spec 2026-06-24) -------------------------------------------
# Drop rating==0 "added-to-shelf" events from each user's HISTORY only; the last two
# interactions (RecBole valid+test targets) are always kept, so the held-out target is
# identical with the flag on or off -> the two runs are directly comparable.
DENOISE_HISTORY = True   # flip to False to produce the comparable raw-SASRec control row

# Equalize the user set across BOTH runs: keep only users who survive denoising (>=3 items
# after dropping unrated history). Apply to `sample` regardless of the flag so raw and
# denoised rows are over the IDENTICAL users (spec: drop the ~1.3% empty-history users).
survivors = set(denoise_history_keep_targets(sample, min_rating=1, n_targets=2,
                                             min_items=3)[USER])
dropped = sample[USER].nunique() - len(survivors)
print(f"user-set equalization: dropping {dropped} users "
      f"({dropped / sample[USER].nunique():.2%}) with empty denoised history")
sample = sample[sample[USER].isin(survivors)].reset_index(drop=True)

if DENOISE_HISTORY:
    sample = denoise_history_keep_targets(sample, min_rating=1, n_targets=2, min_items=3)
    print(f"after shelf-denoise: {sample[USER].nunique()} users, {len(sample):,} interactions")
# --------------------------------------------------------------------------------------
```

- [ ] **Step 2: Add a sanity-assertion cell** immediately after cell 4 to prove the target is preserved (this is the local, no-GPU validation that the wiring is correct):

```python
# Validity guard: each user's held-out LAST interaction must be unchanged by denoising.
_raw_last = (pd.read_parquet(find_data("sample.parquet"))
             .sort_values([USER, "timestamp"]).groupby(USER).tail(1)
             .set_index(USER)[BOOK])
_kept_last = holdout.set_index(USER)[BOOK]
common = _kept_last.index.intersection(_raw_last.index)
assert (_kept_last.loc[common] == _raw_last.loc[common]).all(), \
    "denoise changed a held-out target — targets must be preserved"
print(f"target-preservation check passed for {len(common):,} users")
```

- [ ] **Step 3: Local dry-run of the cell logic** (no GPU, tiny synthetic frame) to confirm the block runs and preserves targets. Run in a scratch Python session:

```bash
python - <<'PY'
import pandas as pd
from book_recsys.data.filters import denoise_history_keep_targets
from book_recsys.data.splits import leave_last_n_out
from book_recsys.data.schema import USER, BOOK, RATING, TS
df = pd.DataFrame([
    {USER:"u",BOOK:"a",RATING:0,TS:1},{USER:"u",BOOK:"b",RATING:4,TS:2},
    {USER:"u",BOOK:"c",RATING:0,TS:3},{USER:"u",BOOK:"d",RATING:0,TS:4},
])
out = denoise_history_keep_targets(df, min_rating=1, n_targets=2, min_items=3)
_, holdout = leave_last_n_out(out, n=1)
assert holdout.iloc[0][BOOK] == "d"            # last item preserved as target
assert "a" not in set(out[BOOK])               # rating-0 history dropped
assert set(out[BOOK]) == {"b","c","d"}         # c,d kept as last-two targets
print("dry-run OK:", list(out[BOOK]))
PY
```
Expected: `dry-run OK: ['b', 'c', 'd']`.

- [ ] **Step 4: Commit the notebook wiring**

```bash
git add notebooks/06_recbole.ipynb
git commit -m "feat(06): shelf-denoise SASRec history behind DENOISE_HISTORY flag, equalize users"
```

---

### Task 3: Run both SASRec trainings (cloud GPU) + collate the comparison table

**Files:**
- Modify: `reports/model_report.md` (Neural — SASRec section)
- Produces (run artifacts): two `sasrec_results.json` / score exports — one per flag setting.

> **Environment note:** RecBole requires CUDA — run `06_recbole.ipynb` on Kaggle/Colab, not the M4. This task is notebook execution, so it has no pytest cycle; its deliverable is the two result sets + the report table. Keep `N_USERS = 30000` and `random_state=42` so the subsample matches the existing study.

- [ ] **Step 1: Control run (raw, equalized).** In `06_recbole.ipynb` set `DENOISE_HISTORY = False`, run end-to-end (train → export top-K → cell 21 popularity-matched + full-catalog scoring). Save the metrics as `sasrec_raw_eq` (rename the output JSON to `sasrec_results_raw_eq.json`).

- [ ] **Step 2: Treatment run (denoised).** Set `DENOISE_HISTORY = True`, restart kernel, run end-to-end. Save as `sasrec_results_denoised.json`. The in-process baselines from cell 21 (svd/hybrid/max-sim/content_emb) are scored on the same equalized users automatically — record those once (they are the shared reference, identical across both runs).

- [ ] **Step 3: Collate** the two SASRec rows + the baseline rows into a new sub-table under the SASRec section of `reports/model_report.md`:

```markdown
#### Shelf-denoising ablation (input-history denoise, fixed target — directly comparable)

Same 30k subsample, equalized to the {N} users surviving denoising; held-out target is each
user's true last book (unchanged by the filter), popularity-matched negatives, k=10.
`sasrec_results_raw_eq.json` / `sasrec_results_denoised.json`.

| method | recall@10 | ndcg@10 | mrr |
|---|---|---|---|
| SASRec (raw history) | _r_ | _r_ | _r_ |
| SASRec (shelf-denoised history) | _d_ | _d_ | _d_ |
| max-sim (reference) | _._ | _._ | _._ |

**Caveat (spec §Output):** 68% of held-out targets are themselves `rating==0` shelf-adds —
fair across all methods (relative comparison valid), but it bounds the absolute reading; the
denoise cleans the *input ordering*, not the target. **Finding:** {denoising lifts/does-not-move
NDCG@10 by Δ}; {one-line interpretation per the spec decision gate}.
```

Fill `{N}`, the metric cells, and `{...}` from the run outputs.

- [ ] **Step 4: Commit**

```bash
git add reports/model_report.md notebooks/06_recbole.ipynb
git commit -m "report: SASRec shelf-denoising ablation (raw vs denoised history, fixed target)"
```

---

## Self-Review

**Spec coverage:**
- One-lever denoise (drop rating==0, keep multi-rated days) → Task 1 (`min_rating=1`, no session collapse). ✓
- Input-history-only, targets preserved → Task 1 (`n_targets=2`) + Task 2 Step 2 guard. ✓
- Directly comparable / fixed target / no baseline re-score → Task 2 equalization + Task 3 shared baseline rows. ✓
- Equalize user set (drop ~1.3%) → Task 2 `survivors`. ✓
- Cloud-GPU train, popularity-matched headline + full-catalog anchor → Task 3. ✓
- 68%-shelf-add-target caveat recorded → Task 3 Step 3 table caveat. ✓
- Report mini-table → Task 3. ✓
- Out-of-scope guards (GRU4Rec, session collapse, UI, is_read) → Global Constraints. ✓

**Placeholder scan:** the `_r_/_d_/{N}/{...}` tokens in Task 3 are run-output fill-ins (values unknowable until the GPU run), explicitly labelled — not code placeholders. All code steps contain complete code.

**Type consistency:** `denoise_history_keep_targets(df, min_rating, n_targets, min_items)` signature is identical in Task 1 definition and all Task 2 call sites. Schema constants `USER/BOOK/RATING/TS` used throughout.
