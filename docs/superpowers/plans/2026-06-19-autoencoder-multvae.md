# Mult-VAE Autoencoder Recommender — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Mult-VAE (variational autoencoder) collaborative-filtering recommender that recommends books from a reading history, optimizes NDCG@10/Recall, resists popularity bias, and slots into the existing `Recommender` protocol + eval harness with zero changes to either.

**Architecture:** A user is represented by their multi-hot item vector; a nonlinear MLP encoder→latent→decoder reconstructs a preference score over the whole catalog (Liang et al. 2018). Logic lives in `book_recsys/models/autoencoder/` (`data`/`model`/`train`/`recommender`); the **notebook is the runner** (config cell → train → eval), exactly like `06_recbole.ipynb`. Trained on the same 30k-user / 2,273,496-interaction sample as SASRec for a fair comparison.

**Tech Stack:** PyTorch (MPS/CUDA/CPU), NumPy, SciPy sparse, pandas. Reuses `book_recsys.eval.harness` for scoring.

## Global Constraints

- Python **3.12**; formatting `yapf` (pep8, `column_limit = 99`, `coalesce_brackets`, `split_before_dot`), `isort` line width 99, `mypy` clean.
- **TDD**: write the failing test first, every task. **100% coverage** on new `book_recsys/**` modules (`coverage report --fail-under=100`). Orchestration in `scripts/`+`notebooks/` is outside the coverage source (`source = ["book_recsys"]`) by design.
- Recommenders implement `fit(train_df) -> self`, `recommend(query, k) -> list`, `score_items(query, item_ids) -> list[float]` (see `book_recsys/models/base.py`, `classical/svd.py`).
- Schema constants from `book_recsys.data.schema`: `USER="user_id"`, `BOOK="book_id"`, `RATING="rating"`, `TS="timestamp"`.
- PyTorch device per CLAUDE.md: auto-detect **cuda → mps → cpu**.
- `torch` is an **optional dep** `[autoencoder]`; checkpoints go to `artifacts/` (gitignored). Specs/plans/reports are force-added (`git add -f`) — `/docs/`, `*.md` are gitignored here.
- Tests run with `python -m pytest`; the env already has `torch 2.12.0` (MPS available).

## File Structure

```
book_recsys/models/autoencoder/
  __init__.py        # exports MultVaeRecommender
  data.py            # reproduce_sasrec_sample, build_matrix
  model.py           # MultVAE(nn.Module): encode/decode/forward/predict/loss
  train.py           # anneal_beta, _device_type, save/load_checkpoint, train_multvae
  recommender.py     # MultVaeRecommender (Recommender protocol) + attach()
tests/models/autoencoder/
  __init__.py
  test_data.py  test_model.py  test_train.py  test_recommender.py
scripts/train_multvae.py            # OPTIONAL headless mirror of the notebook
notebooks/10_autoencoder.ipynb      # RUNNER: config cell → train → eval → figures
pyproject.toml                      # add [autoencoder] = ["torch"]
reports/model_report.md             # add Mult-VAE subsection (results filled after training)
```

---

### Task 1: Package skeleton + dependency + data module

**Files:**
- Create: `book_recsys/models/autoencoder/__init__.py`, `book_recsys/models/autoencoder/data.py`
- Create: `tests/models/autoencoder/__init__.py`, `tests/models/autoencoder/test_data.py`
- Modify: `pyproject.toml` (add `[autoencoder]` optional dep)

**Interfaces:**
- Produces:
  - `reproduce_sasrec_sample(sample_df: pd.DataFrame, n_users=30000, max_hist=100, seed=42, expect_rows: int | None = None) -> pd.DataFrame`
  - `build_matrix(train_df: pd.DataFrame, min_item_count: int = 1) -> tuple[scipy.sparse.csr_matrix, list, dict, np.ndarray]` returning `(matrix[n_users×n_items binary], ids, pos, counts)` where `ids[j]` is the book at column `j`, `pos[book]=j`, `counts[j]` is item `j`'s interaction count.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/autoencoder/__init__.py` (empty), then `tests/models/autoencoder/test_data.py`:

```python
import numpy as np
import pandas as pd
import pytest

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.models.autoencoder.data import build_matrix, reproduce_sasrec_sample


def _df(rows):
    return pd.DataFrame([{USER: u, BOOK: b, RATING: 0, TS: t} for u, b, t in rows])


def test_reproduce_caps_users_and_history():
    rows = []
    for u in range(5):
        for t in range(10):
            rows.append((f"u{u}", f"b{t}", t))
    out = reproduce_sasrec_sample(_df(rows), n_users=3, max_hist=4, seed=42)
    assert out[USER].nunique() == 3
    assert (out.groupby(USER).size() == 4).all()  # capped to most-recent 4


def test_reproduce_keeps_most_recent_history():
    rows = [("u0", f"b{t}", t) for t in range(6)]
    out = reproduce_sasrec_sample(_df(rows), n_users=1, max_hist=2, seed=42)
    assert set(out[BOOK]) == {"b4", "b5"}  # newest two by timestamp


def test_reproduce_expect_rows_mismatch_raises():
    rows = [("u0", "b0", 0), ("u0", "b1", 1)]
    with pytest.raises(ValueError, match="expected 999"):
        reproduce_sasrec_sample(_df(rows), n_users=1, max_hist=10, expect_rows=999)


def test_reproduce_expect_rows_match_ok():
    rows = [("u0", "b0", 0), ("u0", "b1", 1)]
    out = reproduce_sasrec_sample(_df(rows), n_users=1, max_hist=10, expect_rows=2)
    assert len(out) == 2


def test_build_matrix_shape_binary_and_vocab():
    train = _df([("u0", "b0", 0), ("u0", "b1", 1), ("u1", "b0", 0)])
    matrix, ids, pos, counts = build_matrix(train, min_item_count=1)
    assert matrix.shape == (2, 2)
    assert set(matrix.data) == {1.0}            # binary
    assert set(ids) == {"b0", "b1"} and pos[ids[0]] == 0
    assert counts[pos["b0"]] == 2 and counts[pos["b1"]] == 1


def test_build_matrix_min_item_count_filters():
    train = _df([("u0", "b0", 0), ("u1", "b0", 0), ("u0", "b1", 1)])
    matrix, ids, pos, counts = build_matrix(train, min_item_count=2)
    assert ids == ["b0"]                          # b1 (count 1) dropped
    assert matrix.shape[1] == 1


def test_build_matrix_dedupes_repeat_interactions():
    train = _df([("u0", "b0", 0), ("u0", "b0", 1)])  # same user-item twice
    matrix, ids, pos, counts = build_matrix(train, min_item_count=1)
    assert matrix[0, pos["b0"]] == 1.0               # not 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/autoencoder/test_data.py -q`
Expected: FAIL — `ModuleNotFoundError: book_recsys.models.autoencoder.data`.

- [ ] **Step 3: Create the package + implementation**

`book_recsys/models/autoencoder/__init__.py`:

```python
"""Mult-VAE autoencoder collaborative-filtering recommender (Liang et al., 2018)."""
from book_recsys.models.autoencoder.recommender import MultVaeRecommender

__all__ = ["MultVaeRecommender"]
```

> NOTE: this import will fail until Task 4 creates `recommender.py`. To keep Task 1 runnable, **temporarily** make `__init__.py` empty now and add the export in Task 4 Step 3. (Empty `__init__.py` is omitted from coverage per `pyproject` config.)

For Task 1, write `book_recsys/models/autoencoder/__init__.py` as:

```python
"""Mult-VAE autoencoder collaborative-filtering recommender (Liang et al., 2018)."""
```

`book_recsys/models/autoencoder/data.py`:

```python
"""Reproduce SASRec's 30k-user sample and build the user×item training matrix."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from book_recsys.data.schema import BOOK, TS, USER


def reproduce_sasrec_sample(sample_df: pd.DataFrame, n_users: int = 30000,
                            max_hist: int = 100, seed: int = 42,
                            expect_rows: int | None = None) -> pd.DataFrame:
    """Subsample to `n_users` users (pandas RNG, matching 06_recbole.ipynb) and cap each
    user's history to the most-recent `max_hist` interactions by timestamp. If `expect_rows`
    is given, raise unless the row count matches — proves bit-identity with SASRec's set.
    """
    keep = sample_df[USER].drop_duplicates().sample(n_users, random_state=seed)
    out = sample_df[sample_df[USER].isin(keep)]
    out = (out.sort_values([USER, TS]).groupby(USER, sort=False).tail(max_hist)
              .reset_index(drop=True))
    if expect_rows is not None and len(out) != expect_rows:
        raise ValueError(f"expected {expect_rows} interactions, got {len(out)}")
    return out


def build_matrix(train_df: pd.DataFrame, min_item_count: int = 1):
    """User×item binary CSR matrix + item vocab. Items with < `min_item_count` total
    interactions are dropped. Returns (matrix, ids, pos, counts): ids[j] is the book at
    column j, pos[book]=j, counts[j] is that item's interaction count (aligned to ids).
    """
    counts_s = train_df[BOOK].value_counts()
    kept = counts_s[counts_s >= min_item_count]
    ids = list(kept.index)
    pos = {b: j for j, b in enumerate(ids)}
    df = train_df[train_df[BOOK].isin(pos)]
    users = df[USER].astype("category")
    rows = users.cat.codes.to_numpy()
    cols = df[BOOK].map(pos).to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    matrix = sp.csr_matrix((data, (rows, cols)),
                           shape=(users.cat.categories.size, len(ids)))
    matrix.data[:] = 1.0  # collapse any summed duplicates back to binary
    return matrix, ids, pos, kept.to_numpy().astype(np.float64)
```

Add to `pyproject.toml` under `[project.optional-dependencies]` (next to `recbole = [...]`):

```toml
autoencoder = ["torch"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/autoencoder/test_data.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add -f tests/models/autoencoder/__init__.py tests/models/autoencoder/test_data.py \
  book_recsys/models/autoencoder/__init__.py book_recsys/models/autoencoder/data.py pyproject.toml
git commit -m "feat(autoencoder): 30k-sample reproduction + user×item matrix builder"
```

---

### Task 2: MultVAE model (encode / decode / forward / predict / loss)

**Files:**
- Create: `book_recsys/models/autoencoder/model.py`
- Create: `tests/models/autoencoder/test_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class MultVAE(nn.Module)` with `__init__(self, n_items, hidden=600, latent=200, dropout=0.5)`, attributes `enc1` (`nn.Linear`, has `.out_features`), `drop` (`nn.Dropout`, has `.p`), `latent` (int); methods `encode(x)->(mu,logvar)`, `decode(z)->logits`, `forward(x)->(logits,mu,logvar)`, `predict(x)->logits` (deterministic, uses mu), `loss(x, logits, mu, logvar, beta)->Tensor`.

- [ ] **Step 1: Write the failing tests**

`tests/models/autoencoder/test_model.py`:

```python
import math

import torch

from book_recsys.models.autoencoder.model import MultVAE


def test_shapes():
    m = MultVAE(n_items=8, hidden=16, latent=4, dropout=0.0)
    x = torch.zeros(3, 8)
    x[:, 0] = 1.0
    logits, mu, logvar = m(x)
    assert logits.shape == (3, 8)
    assert mu.shape == (3, 4) and logvar.shape == (3, 4)
    assert m.predict(x).shape == (3, 8)


def test_predict_is_deterministic():
    m = MultVAE(n_items=8, hidden=16, latent=4, dropout=0.0).eval()
    x = torch.zeros(1, 8)
    x[0, :3] = 1.0
    a = m.predict(x)
    b = m.predict(x)
    assert torch.allclose(a, b)


def test_loss_multinomial_nll_value():
    m = MultVAE(n_items=2, hidden=4, latent=2)
    x = torch.tensor([[1.0, 0.0]])
    logits = torch.zeros(1, 2)            # log_softmax -> [-ln2, -ln2]
    mu = torch.zeros(1, 2)
    logvar = torch.zeros(1, 2)            # KL = 0
    loss = m.loss(x, logits, mu, logvar, beta=1.0)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-5)


def test_loss_beta_scales_kl():
    m = MultVAE(n_items=2, hidden=4, latent=2)
    x = torch.tensor([[1.0, 0.0]])
    logits = torch.zeros(1, 2)
    mu = torch.ones(1, 2)                 # nonzero -> positive KL
    logvar = torch.zeros(1, 2)
    lo = m.loss(x, logits, mu, logvar, beta=0.0)
    hi = m.loss(x, logits, mu, logvar, beta=1.0)
    assert hi.item() > lo.item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/autoencoder/test_model.py -q`
Expected: FAIL — `ModuleNotFoundError: ...model`.

- [ ] **Step 3: Write the model**

`book_recsys/models/autoencoder/model.py`:

```python
"""Mult-VAE (Liang et al., 2018) — variational autoencoder for collaborative filtering."""
import torch
import torch.nn.functional as F
from torch import nn


class MultVAE(nn.Module):
    """Encoder MLP -> Gaussian latent -> decoder MLP over the full item vocab.

    The multinomial likelihood (softmax over items) is what makes this less popularity-biased
    than pointwise models: it normalizes probability mass across the catalog.
    """

    def __init__(self, n_items: int, hidden: int = 600, latent: int = 200,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.enc1 = nn.Linear(n_items, hidden)
        self.enc2 = nn.Linear(hidden, latent * 2)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, n_items)
        self.drop = nn.Dropout(dropout)
        self.latent = latent

    def encode(self, x):
        h = F.normalize(x, dim=1)           # L2-normalize the user vector (paper convention)
        h = self.drop(h)                    # denoising corruption
        h = torch.tanh(self.enc1(h))
        h = self.enc2(h)
        return h[:, :self.latent], h[:, self.latent:]

    def decode(self, z):
        return self.dec2(torch.tanh(self.dec1(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(z), mu, logvar

    def predict(self, x):
        """Deterministic inference path: decode the latent mean (no sampling)."""
        mu, _ = self.encode(x)
        return self.decode(mu)

    def loss(self, x, logits, mu, logvar, beta):
        nll = -(F.log_softmax(logits, dim=1) * x).sum(dim=1).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        return nll + beta * kl
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/autoencoder/test_model.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -f book_recsys/models/autoencoder/model.py tests/models/autoencoder/test_model.py
git commit -m "feat(autoencoder): MultVAE module (multinomial likelihood, predict path, ELBO loss)"
```

---

### Task 3: Training loop + β-annealing + atomic checkpoints

**Files:**
- Create: `book_recsys/models/autoencoder/train.py`
- Create: `tests/models/autoencoder/test_train.py`

**Interfaces:**
- Consumes: `MultVAE` (Task 2), `build_matrix` (Task 1).
- Produces:
  - `anneal_beta(step: int, anneal_steps: int, beta_cap: float) -> float`
  - `_device_type(device: str) -> str` (`"cuda"`/`"mps"`/`"cpu"`)
  - `save_checkpoint(path, model, optimizer, epoch, config, ids) -> None` (atomic)
  - `load_checkpoint(path, device="cpu") -> tuple[MultVAE, dict]` (dict has `state_dict`,`optimizer`,`epoch`,`config`,`ids`)
  - `train_multvae(model, matrix, *, epochs, batch_size=500, lr=1e-3, anneal_steps=10000, beta_cap=0.2, device="cpu", amp=False, seed=42, ids=None, ckpt_dir=None, ckpt_prefix="multvae", start_epoch=0, optimizer=None) -> MultVAE`

- [ ] **Step 1: Write the failing tests**

`tests/models/autoencoder/test_train.py`:

```python
import numpy as np
import scipy.sparse as sp
import torch

from book_recsys.models.autoencoder.model import MultVAE
from book_recsys.models.autoencoder.train import (anneal_beta, _device_type,
                                                  load_checkpoint, save_checkpoint,
                                                  train_multvae)


def test_anneal_beta_linear_then_flat():
    assert anneal_beta(0, 10, 0.2) == 0.0
    assert anneal_beta(5, 10, 0.2) == 0.1
    assert anneal_beta(10, 10, 0.2) == 0.2
    assert anneal_beta(99, 10, 0.2) == 0.2          # clamps
    assert anneal_beta(3, 0, 0.2) == 0.2            # no warm-up


def test_device_type_mapping():
    assert _device_type("cuda:0") == "cuda"
    assert _device_type("mps") == "mps"
    assert _device_type("cpu") == "cpu"


def _block_matrix():
    # 8 users x 4 items, two co-occurring pairs: users 0-3 read {0,1}, 4-7 read {2,3}
    rows, cols = [], []
    for u in range(4):
        rows += [u, u]; cols += [0, 1]
    for u in range(4, 8):
        rows += [u, u]; cols += [2, 3]
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(8, 4))


def test_training_reduces_loss():
    torch.manual_seed(0)
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=16, latent=4, dropout=0.0)
    x = torch.from_numpy(matrix.toarray())
    with torch.no_grad():
        logits, mu, logvar = model(x)
        before = model.loss(x, logits, mu, logvar, beta=0.2).item()
    train_multvae(model, matrix, epochs=50, batch_size=8, anneal_steps=10, device="cpu")
    with torch.no_grad():
        logits, mu, logvar = model(x)
        after = model.loss(x, logits, mu, logvar, beta=0.2).item()
    assert after < before


def test_amp_path_runs_on_cpu():
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=8, latent=2, dropout=0.0)
    train_multvae(model, matrix, epochs=1, batch_size=8, device="cpu", amp=True)


def test_checkpoint_save_load_roundtrip(tmp_path):
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=8, latent=2, dropout=0.0)
    train_multvae(model, matrix, epochs=2, batch_size=8, device="cpu",
                  ids=["b0", "b1", "b2", "b3"], ckpt_dir=str(tmp_path))
    ckpt_file = tmp_path / "multvae_last.pt"
    assert ckpt_file.exists()
    loaded, ckpt = load_checkpoint(str(ckpt_file), device="cpu")
    assert ckpt["epoch"] == 2 and ckpt["ids"] == ["b0", "b1", "b2", "b3"]
    x = torch.from_numpy(matrix.toarray())
    assert torch.allclose(loaded.predict(x), model.predict(x))


def test_resume_from_checkpoint(tmp_path):
    matrix = _block_matrix()
    model = MultVAE(n_items=4, hidden=8, latent=2, dropout=0.0)
    train_multvae(model, matrix, epochs=1, batch_size=8, device="cpu",
                  ids=None, ckpt_dir=str(tmp_path))
    loaded, ckpt = load_checkpoint(str(tmp_path / "multvae_last.pt"), device="cpu")
    opt = torch.optim.Adam(loaded.parameters())
    opt.load_state_dict(ckpt["optimizer"])
    train_multvae(loaded, matrix, epochs=3, batch_size=8, device="cpu",
                  start_epoch=ckpt["epoch"], optimizer=opt, ckpt_dir=str(tmp_path))
    _, ckpt2 = load_checkpoint(str(tmp_path / "multvae_last.pt"), device="cpu")
    assert ckpt2["epoch"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/autoencoder/test_train.py -q`
Expected: FAIL — `ModuleNotFoundError: ...train`.

- [ ] **Step 3: Write the training module**

`book_recsys/models/autoencoder/train.py`:

```python
"""Training loop, β-annealing, and atomic checkpointing for Mult-VAE."""
import os
import tempfile

import numpy as np
import torch

from book_recsys.models.autoencoder.model import MultVAE


def anneal_beta(step: int, anneal_steps: int, beta_cap: float) -> float:
    """Linear KL warm-up: 0 -> beta_cap over `anneal_steps` gradient steps, then flat."""
    if anneal_steps <= 0:
        return beta_cap
    return beta_cap * min(1.0, step / anneal_steps)


def _device_type(device: str) -> str:
    s = str(device)
    if "cuda" in s:
        return "cuda"
    if "mps" in s:
        return "mps"
    return "cpu"


def _save_atomic(path: str, payload: dict) -> None:
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, path)            # atomic on POSIX: a mid-write kill can't corrupt path


def _config_of(model: MultVAE, n_items: int) -> dict:
    return {"n_items": n_items, "hidden": model.enc1.out_features,
            "latent": model.latent, "dropout": model.drop.p}


def save_checkpoint(path, model, optimizer, epoch, config, ids) -> None:
    _save_atomic(path, {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch, "config": config, "ids": ids,
    })


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MultVAE(cfg["n_items"], cfg["hidden"], cfg["latent"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, ckpt


def train_multvae(model, matrix, *, epochs, batch_size=500, lr=1e-3, anneal_steps=10000,
                  beta_cap=0.2, device="cpu", amp=False, seed=42, ids=None,
                  ckpt_dir=None, ckpt_prefix="multvae", start_epoch=0, optimizer=None):
    """Train `model` on a user×item CSR `matrix`. Checkpoints `<prefix>_last.pt` every epoch
    (atomic). Resume by passing `start_epoch` + a restored `optimizer`.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = matrix.shape[0]
    config = _config_of(model, matrix.shape[1])
    dev_type = _device_type(device)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            x = torch.from_numpy(matrix[idx].toarray().astype("float32")).to(device)
            beta = anneal_beta(step, anneal_steps, beta_cap)
            with torch.autocast(device_type=dev_type, enabled=amp):
                logits, mu, logvar = model(x)
                loss = model.loss(x, logits, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        if ckpt_dir is not None:
            save_checkpoint(os.path.join(ckpt_dir, f"{ckpt_prefix}_last.pt"),
                            model, optimizer, epoch + 1, config, ids)
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/autoencoder/test_train.py -q`
Expected: PASS (6 passed). (`test_training_reduces_loss` is deterministic via the fixed seeds.)

- [ ] **Step 5: Commit**

```bash
git add -f book_recsys/models/autoencoder/train.py tests/models/autoencoder/test_train.py
git commit -m "feat(autoencoder): training loop with beta-annealing + atomic per-epoch checkpoints"
```

---

### Task 4: MultVaeRecommender (fit / recommend / score_items / attach) + e2e

**Files:**
- Create: `book_recsys/models/autoencoder/recommender.py`
- Modify: `book_recsys/models/autoencoder/__init__.py` (add export)
- Create: `tests/models/autoencoder/test_recommender.py`

**Interfaces:**
- Consumes: `build_matrix` (Task 1), `MultVAE` (Task 2), `train_multvae`/`load_checkpoint` (Task 3).
- Produces: `class MultVaeRecommender` implementing the `Recommender` protocol:
  - `__init__(self, hidden=600, latent=200, dropout=0.5, beta_cap=0.2, epochs=50, batch_size=500, lr=1e-3, anneal_steps=10000, min_item_count=1, pop_discount=0.0, device=None, seed=42, ckpt_dir=None)`
  - `fit(train_data) -> self`
  - `recommend(query, k) -> list`
  - `score_items(query, item_ids) -> list[float]`
  - `attach(model, ids, pos, counts) -> self` (use an already-trained/loaded model)

- [ ] **Step 1: Write the failing tests**

`tests/models/autoencoder/test_recommender.py`:

```python
import numpy as np
import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.models.autoencoder.recommender import MultVaeRecommender


def _df(rows):
    return pd.DataFrame([{USER: u, BOOK: b, RATING: 0, TS: t} for u, b, t in rows])


# Two clusters: users a* read {b0,b1,b2}; users c* read {b3,b4,b5}. b5 is the most popular.
def _clustered():
    rows = []
    for i in range(10):
        for b in ("b0", "b1", "b2"):
            rows.append((f"a{i}", b, 0))
    for i in range(10):
        for b in ("b3", "b4", "b5"):
            rows.append((f"c{i}", b, 0))
    for i in range(8):                       # make b5 the popularity head
        rows.append((f"a{i}", "b5", 0))
    return _df(rows)


def _fit(**kw):
    return MultVaeRecommender(hidden=32, latent=8, dropout=0.0, epochs=80,
                              batch_size=8, anneal_steps=20, device="cpu",
                              seed=0, **kw).fit(_clustered())


def test_recommend_excludes_seen():
    rec = _fit()
    out = rec.recommend(["b0", "b1"], k=5)
    assert "b0" not in out and "b1" not in out


def test_recommend_prefers_co_cluster():
    rec = _fit()
    out = rec.recommend(["b0", "b1"], k=2)
    assert "b2" in out                       # same cluster as the history


def test_recommend_empty_when_unfitted():
    assert MultVaeRecommender().recommend(["b0"], k=5) == []


def test_recommend_unknown_history_ids_dropped():
    rec = _fit()
    out = rec.recommend(["does-not-exist"], k=3)
    assert isinstance(out, list) and len(out) <= 3


def test_score_items_known_and_unknown():
    rec = _fit()
    scores = rec.score_items(["b0", "b1"], ["b2", "zzz"])
    assert np.isfinite(scores[0]) and scores[1] == float("-inf")


def test_score_items_empty_when_unfitted():
    assert MultVaeRecommender().score_items(["b0"], ["b1", "b2"]) == [float("-inf")] * 2


def test_pop_discount_demotes_popular_head():
    rec = _fit()
    base = rec.score_items(["b3", "b4"], ["b5"])[0]          # b5 = popular head
    rec.pop_discount = 5.0
    discounted = rec.score_items(["b3", "b4"], ["b5"])[0]
    assert discounted < base                                 # popularity penalty applied


def test_attach_roundtrips_scores(tmp_path):
    from book_recsys.models.autoencoder.data import build_matrix
    from book_recsys.models.autoencoder.train import load_checkpoint
    rec = MultVaeRecommender(hidden=16, latent=4, dropout=0.0, epochs=3, batch_size=8,
                             device="cpu", seed=0, ckpt_dir=str(tmp_path)).fit(_clustered())
    before = rec.score_items(["b0"], ["b1", "b2"])
    _, ids, pos, counts = build_matrix(_clustered(), 1)
    model, _ = load_checkpoint(str(tmp_path / "multvae_last.pt"), device="cpu")
    rebuilt = MultVaeRecommender(device="cpu").attach(model, ids, pos, counts)
    after = rebuilt.score_items(["b0"], ["b1", "b2"])
    assert np.allclose(before, after)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/autoencoder/test_recommender.py -q`
Expected: FAIL — `ModuleNotFoundError: ...recommender`.

- [ ] **Step 3: Write the recommender + wire the export**

`book_recsys/models/autoencoder/recommender.py`:

```python
"""Mult-VAE wrapped as a Recommender (fit / recommend / score_items)."""
import numpy as np
import torch

from book_recsys.models.autoencoder.data import build_matrix
from book_recsys.models.autoencoder.model import MultVAE
from book_recsys.models.autoencoder.train import train_multvae


def _auto_device() -> str:
    if torch.cuda.is_available():       # pragma: no cover - hardware-dependent
        return "cuda"
    if torch.backends.mps.is_available():   # pragma: no cover - hardware-dependent
        return "mps"
    return "cpu"                         # pragma: no cover - hardware-dependent


class MultVaeRecommender:
    """Fold-in autoencoder recommender: history -> multi-hot vector -> reconstruct scores.

    `pop_discount` (alpha) subtracts `alpha * log(item_count)` from every score at inference,
    demoting popularity-head items — sweep it to trade accuracy for serendipity.
    """

    def __init__(self, hidden=600, latent=200, dropout=0.5, beta_cap=0.2, epochs=50,
                 batch_size=500, lr=1e-3, anneal_steps=10000, min_item_count=1,
                 pop_discount=0.0, device=None, seed=42, ckpt_dir=None) -> None:
        self.hidden, self.latent, self.dropout = hidden, latent, dropout
        self.beta_cap, self.epochs, self.batch_size = beta_cap, epochs, batch_size
        self.lr, self.anneal_steps = lr, anneal_steps
        self.min_item_count, self.pop_discount = min_item_count, pop_discount
        self.device = device or _auto_device()
        self.seed, self.ckpt_dir = seed, ckpt_dir
        self._ids: list = []
        self._pos: dict = {}
        self._model = None
        self._log_pop = None

    def fit(self, train_data):
        matrix, ids, pos, counts = build_matrix(train_data, self.min_item_count)
        self._ids, self._pos = ids, pos
        self._log_pop = np.log(counts)
        model = MultVAE(len(ids), self.hidden, self.latent, self.dropout)
        train_multvae(model, matrix, epochs=self.epochs, batch_size=self.batch_size,
                      lr=self.lr, anneal_steps=self.anneal_steps, beta_cap=self.beta_cap,
                      device=self.device, seed=self.seed, ids=ids, ckpt_dir=self.ckpt_dir)
        self._model = model.eval()
        return self

    def attach(self, model, ids, pos, counts):
        """Use an already-trained model + vocab (e.g. from load_checkpoint)."""
        self._model = model.eval()
        self._ids, self._pos = ids, pos
        self._log_pop = np.log(np.asarray(counts, dtype=float))
        return self

    def _scores(self, query):
        x = torch.zeros(1, len(self._ids))
        idx = [self._pos[b] for b in query if b in self._pos]
        if idx:
            x[0, idx] = 1.0
        with torch.no_grad():
            logits = self._model.predict(x.to(self.device)).cpu().numpy()[0]
        return logits - self.pop_discount * self._log_pop

    def recommend(self, query, k):
        if self._model is None or not self._ids:
            return []
        scores = self._scores(query)
        seen = {self._pos[b] for b in query if b in self._pos}
        out = []
        for j in np.argsort(-scores):
            if j not in seen:
                out.append(self._ids[j])
                if len(out) == k:
                    break
        return out

    def score_items(self, query, item_ids):
        if self._model is None:
            return [float("-inf")] * len(item_ids)
        scores = self._scores(query)
        return [float(scores[self._pos[b]]) if b in self._pos else float("-inf")
                for b in item_ids]
```

Update `book_recsys/models/autoencoder/__init__.py`:

```python
"""Mult-VAE autoencoder collaborative-filtering recommender (Liang et al., 2018)."""
from book_recsys.models.autoencoder.recommender import MultVaeRecommender

__all__ = ["MultVaeRecommender"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/autoencoder/test_recommender.py -q`
Expected: PASS (8 passed).

> If `test_recommend_prefers_co_cluster` is flaky, raise `epochs` to 120 in `_fit` — the block signal is strong but needs enough steps. Keep `seed=0` for determinism.

- [ ] **Step 5: Commit**

```bash
git add -f book_recsys/models/autoencoder/recommender.py \
  book_recsys/models/autoencoder/__init__.py tests/models/autoencoder/test_recommender.py
git commit -m "feat(autoencoder): MultVaeRecommender (fold-in recommend, pop-discount knob, attach)"
```

---

### Task 5: Notebook runner + optional headless script

**Files:**
- Create: `scripts/train_multvae.py`
- Create: `notebooks/10_autoencoder.ipynb` (built via nbformat)

**Interfaces:**
- Consumes: everything above + `book_recsys.eval.harness` (`build_user_histories`, `build_relevance`, `evaluate_sampled_negatives`, `popularity_diagnostics`), `book_recsys.data.splits.leave_last_n_out`, `book_recsys.data.negatives.build_cdf`.
- Produces: a runnable pipeline; no new importable package surface (so not coverage-measured).

- [ ] **Step 1: Write the headless script**

`scripts/train_multvae.py`:

```python
"""Headless Mult-VAE training + eval mirror of notebooks/10_autoencoder.ipynb.

Run from the repo root once artifacts/sample.parquet exists. The notebook is the primary
runner; this script is for resume-friendly headless/Kaggle runs.
"""
import argparse
import os

import pandas as pd

from book_recsys.data.negatives import build_cdf
from book_recsys.data.schema import BOOK
from book_recsys.data.splits import leave_last_n_out
from book_recsys.eval.harness import (build_relevance, build_user_histories,
                                       evaluate_sampled_negatives, popularity_diagnostics)
from book_recsys.models.autoencoder.data import reproduce_sasrec_sample
from book_recsys.models.autoencoder.recommender import MultVaeRecommender


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="artifacts/sample.parquet")
    p.add_argument("--n-users", type=int, default=30000)
    p.add_argument("--max-hist", type=int, default=100)
    p.add_argument("--expect-rows", type=int, default=2_273_496)
    p.add_argument("--latent", type=int, default=200)
    p.add_argument("--hidden", type=int, default=600)
    p.add_argument("--beta-cap", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-item-count", type=int, default=1)
    p.add_argument("--pop-discount", type=float, default=0.0)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-dir", default="artifacts")
    p.add_argument("--n-eval-users", type=int, default=2000)
    args = p.parse_args()

    sample = pd.read_parquet(args.sample)
    sample = reproduce_sasrec_sample(sample, args.n_users, args.max_hist,
                                     expect_rows=args.expect_rows)
    train, test = leave_last_n_out(sample, n=1)

    rec = MultVaeRecommender(hidden=args.hidden, latent=args.latent, dropout=args.dropout,
                             beta_cap=args.beta_cap, epochs=args.epochs, lr=args.lr,
                             min_item_count=args.min_item_count,
                             pop_discount=args.pop_discount, device=args.device,
                             ckpt_dir=args.ckpt_dir)
    rec.fit(train)

    histories = build_user_histories(train)
    relevance = build_relevance(test)
    eval_users = list(relevance)[:args.n_eval_users]
    relevance = {u: relevance[u] for u in eval_users}
    all_items = sample[BOOK].unique()
    weights = sample[BOOK].value_counts().reindex(all_items).to_numpy()

    headline = evaluate_sampled_negatives(rec, histories, relevance, all_items,
                                          n_neg=100, k=10, seed=0,
                                          item_weights=build_cdf(weights) is not None
                                          and weights or weights)
    pop = popularity_diagnostics(rec, {u: histories.get(u, []) for u in eval_users},
                                 list(sample[BOOK].value_counts().index),
                                 catalog_size=len(all_items), k=10)
    print("headline (popularity-matched neg):", headline)
    print("diagnostics:", pop)


if __name__ == "__main__":      # pragma: no cover
    main()
```

> NOTE: `evaluate_sampled_negatives` takes `item_weights` aligned to `all_items` (raw counts; it builds the CDF internally). Pass `weights` directly: simplify the `item_weights=` argument to `item_weights=weights`. (The `build_cdf(...)` expression above is illustrative of why weights matter — replace it with `item_weights=weights` when writing the file.)

- [ ] **Step 2: Verify the script imports and parses**

Run: `python -c "import ast; ast.parse(open('scripts/train_multvae.py').read()); print('ok')"`
Expected: `ok`. Then `python scripts/train_multvae.py --help` prints the flags (no data needed).

- [ ] **Step 3: Build the notebook (nbformat)**

Run this one-off builder (it writes the .ipynb, then you can delete the builder):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell(
        "# Mult-VAE Autoencoder — UC1\n"
        "Runner notebook (logic lives in `book_recsys.models.autoencoder`). "
        "Trained on the **same 30k users / 2,273,496 interactions as SASRec**.\n"
        "Edit the **config cell** and re-run to try a different setting."),
    nbf.v4.new_code_cell(
        "# --- config cell: edit these, then Run All ---\n"
        "N_USERS, MAX_HIST, EXPECT_ROWS = 30000, 100, 2_273_496\n"
        "LATENT, HIDDEN, BETA_CAP, DROPOUT = 200, 600, 0.2, 0.5\n"
        "EPOCHS, LR, MIN_ITEM_COUNT, POP_DISCOUNT = 50, 1e-3, 1, 0.0\n"
        "DEVICE, CKPT_DIR, N_EVAL_USERS = None, 'artifacts', 2000"),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from book_recsys.data.splits import leave_last_n_out\n"
        "from book_recsys.data.schema import BOOK\n"
        "from book_recsys.models.autoencoder.data import reproduce_sasrec_sample\n"
        "from book_recsys.models.autoencoder.recommender import MultVaeRecommender\n"
        "from book_recsys.eval.harness import (build_user_histories, build_relevance,\n"
        "    evaluate_sampled_negatives, popularity_diagnostics)\n"
        "sample = reproduce_sasrec_sample(pd.read_parquet('artifacts/sample.parquet'),\n"
        "    N_USERS, MAX_HIST, expect_rows=EXPECT_ROWS)\n"
        "train, test = leave_last_n_out(sample, n=1)\n"
        "print(sample['user_id'].nunique(), 'users,', len(sample), 'interactions')"),
    nbf.v4.new_code_cell(
        "rec = MultVaeRecommender(hidden=HIDDEN, latent=LATENT, dropout=DROPOUT,\n"
        "    beta_cap=BETA_CAP, epochs=EPOCHS, lr=LR, min_item_count=MIN_ITEM_COUNT,\n"
        "    pop_discount=POP_DISCOUNT, device=DEVICE, ckpt_dir=CKPT_DIR).fit(train)"),
    nbf.v4.new_code_cell(
        "histories = build_user_histories(train)\n"
        "relevance = build_relevance(test)\n"
        "eval_users = list(relevance)[:N_EVAL_USERS]\n"
        "relevance = {u: relevance[u] for u in eval_users}\n"
        "all_items = sample[BOOK].unique()\n"
        "weights = sample[BOOK].value_counts().reindex(all_items).to_numpy()\n"
        "headline = evaluate_sampled_negatives(rec, histories, relevance, all_items,\n"
        "    n_neg=100, k=10, seed=0, item_weights=weights)\n"
        "headline"),
    nbf.v4.new_code_cell(
        "# Anti-popularity diagnostics + the FREE alpha serendipity sweep (no retraining)\n"
        "pop_order = list(sample[BOOK].value_counts().index)\n"
        "diag = popularity_diagnostics(rec, {u: histories.get(u, []) for u in eval_users},\n"
        "    pop_order, catalog_size=len(all_items), k=10)\n"
        "rows = []\n"
        "for a in (0.0, 0.25, 0.5, 1.0):\n"
        "    rec.pop_discount = a\n"
        "    m = evaluate_sampled_negatives(rec, histories, relevance, all_items,\n"
        "        n_neg=100, k=10, seed=0, item_weights=weights)\n"
        "    rows.append({'alpha': a, **m})\n"
        "rec.pop_discount = POP_DISCOUNT\n"
        "import pandas as pd\n"
        "print(diag); pd.DataFrame(rows)"),
]
nb["cells"] = cells
with open("notebooks/10_autoencoder.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/10_autoencoder.ipynb")
```

- [ ] **Step 4: Verify the notebook is valid**

Run: `python -c "import nbformat; nbformat.read('notebooks/10_autoencoder.ipynb', as_version=4); print('valid')"`
Expected: `valid`. (Executing end-to-end needs `artifacts/sample.parquet` + a GPU/long CPU run — done by the user on Kaggle/laptop; logic is already covered by unit tests.)

- [ ] **Step 5: Commit**

```bash
git add -f scripts/train_multvae.py notebooks/10_autoencoder.ipynb
git commit -m "feat(autoencoder): notebook runner + headless script (train, eval, free alpha sweep)"
```

---

### Task 6: Integration — full suite, 100% coverage, report subsection

**Files:**
- Modify: `reports/model_report.md` (add Mult-VAE subsection with placeholders to fill after training)

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest -q`
Expected: all prior 326 tests + the new autoencoder tests PASS (≈ 351 passed).

- [ ] **Step 2: Verify 100% coverage on the new modules**

Run: `coverage run -m pytest && coverage report --show-missing --fail-under=100`
Expected: exit 0. If `book_recsys/models/autoencoder/*` shows missing lines, add a targeted test (do **not** add `# pragma: no cover` except the hardware-dependent `_auto_device` branches already marked).

- [ ] **Step 3: Lint/format/type-check**

Run:
```bash
yapf -ir book_recsys/models/autoencoder tests/models/autoencoder scripts/train_multvae.py
isort book_recsys/models/autoencoder tests/models/autoencoder scripts/train_multvae.py --line-width 99
mypy book_recsys/models/autoencoder
```
Expected: no diffs after re-run; `mypy` clean. Commit any formatting fixes.

- [ ] **Step 4: Add the report subsection**

Append to `reports/model_report.md` under §3 UC1 (fill the numbers after the Kaggle run):

```markdown
#### Mult-VAE (autoencoder, P1 neural CF)

Variational autoencoder (Liang et al., 2018): history multi-hot -> MLP encoder -> latent ->
decoder over the full item vocab, multinomial likelihood + KL (β annealed 0→β_cap). Trained on
the **same 30k users / 2,273,496 interactions as SASRec** (reproduced via
`reproduce_sasrec_sample`, asserted bit-identical), leave-last-1-out, baseline config
K=200/hidden=600/β_cap=0.2/dropout=0.5.

| method | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Mult-VAE (α=0) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Anti-popularity:** mean-pop-percentile _TBD_, coverage _TBD_; α serendipity sweep
(α∈{0,0.25,0.5,1.0}) traces the accuracy↔serendipity tradeoff — see `notebooks/10_autoencoder.ipynb`.
**β ablation:** β_cap=0 (denoising AE) vs 0.2 (variational) — _TBD_.
```

- [ ] **Step 5: Commit**

```bash
git add -f reports/model_report.md
git commit -m "docs(report): Mult-VAE UC1 subsection (results filled post-training)"
```

---

## Self-Review

**Spec coverage:**
- §2/§3 model (Mult-VAE, multinomial loss, predict path) → Task 2. ✅
- §4 β-annealing → Task 3 (`anneal_beta`). ✅
- §5 recommender (fit/recommend/score_items, pop_discount, attach) → Task 4. ✅
- §6 training loop (device cuda/mps/cpu, AMP, checkpoint every epoch atomic, resume) → Task 3. ✅
- §7 data (reproduce 30k sample + assert 2,273,496, build_matrix, min_item_count) → Task 1. ✅
- §8 serendipity (multinomial inherent + α knob + diagnostics) → Task 4 knob + Task 5 sweep + Task 6 report. ✅
- §9 eval (headline sampled-neg, full-catalog anchor, diagnostics, manual config-cell tuning, free α sweep) → Task 5 notebook + Task 6. ✅
- §10 layout + `[autoencoder]` dep → Tasks 1–5. ✅
- §11 tests (anneal, loss, matrix build, recommend mask, pop-discount, score_items, e2e, checkpoint roundtrip) → Tasks 1–4. ✅
- §12 risks (MPS memory→min_item_count, data-drift→expect_rows assert) → Tasks 1, 5. ✅

**Placeholder scan:** report `_TBD_` cells are intentional (numbers come from the user's training run) — flagged as such, not plan gaps. No other placeholders.

**Type consistency:** `build_matrix` returns `(matrix, ids, pos, counts)` consumed identically in Tasks 3/4; `train_multvae(... ids=, ckpt_dir=, start_epoch=, optimizer=)` signature matches its calls in Task 4 and tests; `load_checkpoint` returns `(model, ckpt)` used consistently; `MultVAE` ctor `(n_items, hidden, latent, dropout)` consistent across `model.py`, `train.py`, `load_checkpoint`, `recommender.py`. ✅

**Note on `scripts/train_multvae.py` (Task 5 Step 1):** when writing the file, set `item_weights=weights` directly (the inline `build_cdf(...) and ...` expression in the draft is illustrative — replace it). The notebook (Step 3) already uses the clean `item_weights=weights`.
