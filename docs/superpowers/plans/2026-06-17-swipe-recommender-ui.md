# Swipe Recommender UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Bookist-style swipe feed over the existing hybrid recommender — seed liked books, then swipe (left=read&liked, right=want-to-read, down=dislike); the feed adapts each swipe and steers *away* from books similar to disliked ones.

**Architecture:** A new tested `FeedService` (candidate gen via the hybrid → exclude seen → penalize by similarity to disliked) and an in-memory `SessionStore`, exposed by a thin FastAPI app, with a static vanilla-JS swipe SPA. All logic in the package; FastAPI + JS are thin shells.

**Tech Stack:** Python 3.12, FastAPI + uvicorn (new `[api]` extra), numpy, pytest + FastAPI `TestClient` (httpx). Reuses `LearnedHybridRecommender` and `RecommenderService`.

**Spec:** `docs/superpowers/specs/2026-06-17-swipe-recommender-ui-design.md`

---

## File Structure

- `book_recsys/ui/feed.py` (new) — `FeedService`: ranking + negative penalization. Pure (recommender + embeddings); no FastAPI/torch.
- `book_recsys/api/__init__.py` (new) — empty package marker.
- `book_recsys/api/sessions.py` (new) — `Session` dataclass + `SessionStore` (in-memory, ephemeral).
- `book_recsys/api/app.py` (new) — `create_app(rec_service, feed_service, session_store)` (testable with fakes) + `get_app()` production factory (loads artifacts, `# pragma: no cover`).
- `book_recsys/ui/web/index.html`, `app.js`, `style.css` (new) — static swipe SPA.
- `tests/ui/test_feed.py` (new) — `FeedService` unit tests.
- `tests/api/__init__.py`, `tests/api/test_sessions.py`, `tests/api/test_app.py` (new) — `SessionStore` + endpoint tests with fakes.
- `pyproject.toml` (modify) — add `[api]` extra (`fastapi`, `uvicorn`) and `httpx` for tests.

**Note on `lambda`:** the API field is named **`lam`** (not `lambda`) because `lambda` is a Python keyword. The spec's `λ` knob = this `lam`.

---

## Task 1: Project setup (`[api]` extra + package dirs)

**Files:**
- Modify: `pyproject.toml`
- Create: `book_recsys/api/__init__.py`, `tests/api/__init__.py`

- [ ] **Step 1: Add the `[api]` extra.** In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
api = ["fastapi", "uvicorn", "httpx"]
```

(`httpx` is needed by FastAPI's `TestClient`.)

- [ ] **Step 2: Install it**

Run: `pip install -e ".[api]"`
Expected: installs fastapi, uvicorn, httpx.

- [ ] **Step 3: Create empty package markers**

Create `book_recsys/api/__init__.py` (empty) and `tests/api/__init__.py` (empty).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml book_recsys/api/__init__.py tests/api/__init__.py
git commit -m "Add [api] extra and api package for the swipe UI"
```

---

## Task 2: `FeedService` — candidate generation, filtering, ranking (no penalty yet)

**Files:**
- Create: `book_recsys/ui/feed.py`
- Test: `tests/ui/test_feed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ui/test_feed.py
import numpy as np
import pytest

from book_recsys.ui.feed import FeedService


class FakeRec:
    """Stand-in for LearnedHybridRecommender: recommend() + score_items()."""

    def __init__(self, rec_order, scores):
        self._order = rec_order
        self._scores = scores

    def recommend(self, history, k):
        return self._order[:k]

    def score_items(self, history, candidates):
        return [self._scores[c] for c in candidates]


def test_next_excludes_seen_liked_disliked_and_ranks_by_score():
    book_ids = ["a", "b", "c", "d", "e"]
    emb = np.eye(5, dtype="float32")
    rec = FakeRec(rec_order=["b", "c", "d", "e"],
                  scores={"b": 0.2, "c": 0.9, "d": 0.5, "e": 0.1})
    fs = FeedService(rec, emb, book_ids, pool=10)
    # a liked, e disliked, d seen -> all excluded; remaining b,c ranked by score desc
    out = fs.next(liked=["a"], disliked=["e"], seen=["d"], k=10, lam=0.0)
    assert out == ["c", "b"]


def test_next_empty_liked_returns_empty():
    fs = FeedService(FakeRec([], {}), np.eye(2, dtype="float32"), ["a", "b"])
    assert fs.next(liked=[], disliked=[], seen=[], k=10) == []


def test_next_respects_k():
    book_ids = ["a", "b", "c", "d"]
    rec = FakeRec(["b", "c", "d"], {"b": 0.3, "c": 0.9, "d": 0.6})
    fs = FeedService(rec, np.eye(4, dtype="float32"), book_ids, pool=10)
    assert fs.next(liked=["a"], disliked=[], seen=[], k=1, lam=0.0) == ["c"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/ui/test_feed.py -v`
Expected: FAIL with `ModuleNotFoundError: book_recsys.ui.feed`.

- [ ] **Step 3: Write minimal implementation**

```python
# book_recsys/ui/feed.py
"""Swipe feed: hybrid candidate generation, exclusion, and negative penalization.

Pure logic — takes any recommender exposing recommend(history, k) and
score_items(history, candidates), plus the book embeddings for the penalty term.
"""
import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class FeedService:
    """Rank the next swipe cards: hybrid score, minus a penalty for similarity to disliked."""

    def __init__(self, recommender, embeddings, book_ids, pool: int = 200) -> None:
        self._rec = recommender
        self._emb = _l2_normalize(np.asarray(embeddings, dtype="float32"))
        self._row = {b: i for i, b in enumerate(book_ids)}
        self._pool = pool

    def next(self, liked, disliked, seen, k: int = 10, lam: float = 1.0) -> list:
        liked = list(liked)
        if not liked:
            return []
        candidates = self._rec.recommend(liked, self._pool)
        exclude = set(seen) | set(liked) | set(disliked)
        candidates = [c for c in candidates if c not in exclude]
        if not candidates:
            return []
        base = _minmax(np.asarray(self._rec.score_items(liked, candidates), dtype="float64"))
        order = np.argsort(-base, kind="stable")[:k]
        return [candidates[i] for i in order]
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/ui/test_feed.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add book_recsys/ui/feed.py tests/ui/test_feed.py
git commit -m "Add FeedService candidate generation, exclusion, and ranking"
```

---

## Task 3: `FeedService` — negative penalization

**Files:**
- Modify: `book_recsys/ui/feed.py`
- Modify: `tests/ui/test_feed.py`

- [ ] **Step 1: Append the failing tests**

```python
# tests/ui/test_feed.py  (append)
def test_penalizes_candidate_similar_to_disliked():
    # b points the same direction as disliked x; c is orthogonal. Equal base scores,
    # so with lam>0 the penalty pushes b below c.
    book_ids = ["a", "b", "c", "x"]
    emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype="float32")
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.5, "c": 0.5})
    fs = FeedService(rec, emb, book_ids, pool=10)
    # lam=0 -> no penalty, stable order keeps recommend order
    assert fs.next(liked=["a"], disliked=["x"], seen=[], k=10, lam=0.0) == ["b", "c"]
    # lam=1 -> b (cos=1 to x) penalized below c (cos=0)
    assert fs.next(liked=["a"], disliked=["x"], seen=[], k=10, lam=1.0) == ["c", "b"]


def test_no_disliked_means_no_penalty():
    book_ids = ["a", "b", "c"]
    emb = np.eye(3, dtype="float32")
    rec = FakeRec(["b", "c"], {"b": 0.9, "c": 0.1})
    fs = FeedService(rec, emb, book_ids, pool=10)
    assert fs.next(liked=["a"], disliked=[], seen=[], k=10, lam=1.0) == ["b", "c"]
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `pytest tests/ui/test_feed.py::test_penalizes_candidate_similar_to_disliked -v`
Expected: FAIL (`lam` currently ignored → returns `["b", "c"]` for the lam=1 case).

- [ ] **Step 3: Add the penalty term to `FeedService.next`**

Replace the body from `base = ...` onward in `next()` with:

```python
        base = _minmax(np.asarray(self._rec.score_items(liked, candidates), dtype="float64"))
        if disliked and lam:
            base = base - lam * self._max_sim_to_disliked(candidates, disliked)
        order = np.argsort(-base, kind="stable")[:k]
        return [candidates[i] for i in order]

    def _max_sim_to_disliked(self, candidates, disliked) -> np.ndarray:
        """For each candidate, cosine similarity to its NEAREST disliked book (0 if none)."""
        d_rows = [self._row[d] for d in disliked if d in self._row]
        if not d_rows:
            return np.zeros(len(candidates))
        c_rows = [self._row[c] for c in candidates]
        sims = self._emb[c_rows] @ self._emb[d_rows].T   # normalized -> cosine
        return sims.max(axis=1)
```

- [ ] **Step 4: Run to verify all FeedService tests pass**

Run: `pytest tests/ui/test_feed.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add book_recsys/ui/feed.py tests/ui/test_feed.py
git commit -m "Add negative penalization to FeedService (steer away from disliked-similar)"
```

---

## Task 4: `SessionStore`

**Files:**
- Create: `book_recsys/api/sessions.py`
- Test: `tests/api/test_sessions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_sessions.py
import pytest

from book_recsys.api.sessions import SessionStore


def test_create_seeds_liked_and_seen():
    st = SessionStore()
    sid = st.create(["a", "b"], lam=0.5, k=5)
    s = st.get(sid)
    assert s.liked == ["a", "b"]
    assert s.seen == {"a", "b"}      # seeds are not re-recommended
    assert s.lam == 0.5 and s.k == 5
    assert s.disliked == [] and s.reading_list == []


def test_apply_actions_update_the_right_sets():
    st = SessionStore()
    sid = st.create(["a"])
    st.apply(sid, "c", "like")
    st.apply(sid, "d", "want")
    st.apply(sid, "e", "dislike")
    st.apply(sid, "f", "skip")
    s = st.get(sid)
    assert s.liked == ["a", "c", "d"]          # like + want
    assert s.reading_list == ["d"]             # want only
    assert s.disliked == ["e"]
    assert {"a", "c", "d", "e", "f"} == s.seen  # every swipe marks seen


def test_unknown_action_raises_valueerror():
    st = SessionStore()
    sid = st.create([])
    with pytest.raises(ValueError):
        st.apply(sid, "x", "love")


def test_unknown_session_raises_keyerror():
    st = SessionStore()
    with pytest.raises(KeyError):
        st.get("nope")
    with pytest.raises(KeyError):
        st.apply("nope", "x", "like")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/api/test_sessions.py -v`
Expected: FAIL with `ModuleNotFoundError: book_recsys.api.sessions`.

- [ ] **Step 3: Write minimal implementation**

```python
# book_recsys/api/sessions.py
"""In-memory, ephemeral swipe sessions (no DB; lost on restart)."""
import uuid
from dataclasses import dataclass, field

_ACTIONS = {"like", "want", "dislike", "skip"}


@dataclass
class Session:
    lam: float = 1.0
    k: int = 10
    liked: list = field(default_factory=list)
    disliked: list = field(default_factory=list)
    reading_list: list = field(default_factory=list)
    seen: set = field(default_factory=set)


class SessionStore:
    """Maps session_id -> Session and applies swipe actions to the stored state."""

    def __init__(self) -> None:
        self._sessions: dict = {}

    def create(self, liked, lam: float = 1.0, k: int = 10) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = Session(lam=lam, k=k, liked=list(liked), seen=set(liked))
        return sid

    def get(self, session_id: str) -> Session:
        return self._sessions[session_id]   # KeyError if unknown

    def apply(self, session_id: str, book_id, action: str) -> Session:
        if action not in _ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        session = self._sessions[session_id]   # KeyError if unknown
        session.seen.add(book_id)
        if action in ("like", "want"):
            session.liked.append(book_id)
        if action == "want":
            session.reading_list.append(book_id)
        if action == "dislike":
            session.disliked.append(book_id)
        return session
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/api/test_sessions.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add book_recsys/api/sessions.py tests/api/test_sessions.py
git commit -m "Add in-memory SessionStore for swipe sessions"
```

---

## Task 5: FastAPI app — `create_app` with `/search`, `/session`, `/swipe`

**Files:**
- Create: `book_recsys/api/app.py`
- Test: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_app.py
from fastapi.testclient import TestClient

from book_recsys.api.app import create_app
from book_recsys.api.sessions import SessionStore


class FakeRecService:
    def search(self, q, limit=20):
        return ["a", "b"][:limit]

    def label(self, book_id):
        return f"Title-{book_id}"


class FakeFeed:
    """Returns x,y,z minus anything already seen/disliked (ignores scores)."""

    def next(self, liked, disliked, seen, k=10, lam=1.0):
        pool = [b for b in ["x", "y", "z"] if b not in set(seen) | set(disliked)]
        return pool[:k]


def make_client():
    return TestClient(create_app(FakeRecService(), FakeFeed(), SessionStore()))


def test_search_returns_labeled_books():
    r = make_client().get("/search", params={"q": "foo"})
    assert r.status_code == 200
    assert r.json() == [{"book_id": "a", "label": "Title-a"},
                        {"book_id": "b", "label": "Title-b"}]


def test_session_then_swipe_adapts_and_collects_reading_list():
    c = make_client()
    r = c.post("/session", json={"liked": ["a"], "lam": 1.0, "k": 10})
    body = r.json()
    sid = body["session_id"]
    assert body["cards"][0]["book_id"] == "x"      # x,y,z (a is liked->seen, not in pool anyway)

    r2 = c.post("/swipe", json={"session_id": sid, "book_id": "x", "action": "want"})
    body2 = r2.json()
    assert body2["reading_list"] == [{"book_id": "x", "label": "Title-x"}]
    assert body2["cards"][0]["book_id"] == "y"     # x now seen -> next card is y


def test_swipe_unknown_session_returns_404():
    r = make_client().post("/swipe",
                           json={"session_id": "nope", "book_id": "x", "action": "like"})
    assert r.status_code == 404


def test_swipe_bad_action_returns_400():
    c = make_client()
    sid = c.post("/session", json={"liked": []}).json()["session_id"]
    r = c.post("/swipe", json={"session_id": sid, "book_id": "x", "action": "love"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/api/test_app.py -v`
Expected: FAIL with `ImportError: cannot import name 'create_app'`.

- [ ] **Step 3: Write minimal implementation**

```python
# book_recsys/api/app.py
"""FastAPI swipe API. `create_app` is injectable (tested with fakes); `get_app`
wires the real artifact-backed services for uvicorn."""
from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from book_recsys.api.sessions import SessionStore


class SessionReq(BaseModel):
    liked: list = []
    lam: float = 1.0
    k: int = 10


class SwipeReq(BaseModel):
    session_id: str
    book_id: Union[int, str]
    action: str


def create_app(rec_service, feed_service, session_store) -> FastAPI:
    app = FastAPI(title="Book Swipe")

    def cards(book_ids):
        return [{"book_id": b, "label": rec_service.label(b)} for b in book_ids]

    def feed_for(session):
        return cards(feed_service.next(session.liked, session.disliked, session.seen,
                                       k=session.k, lam=session.lam))

    @app.get("/search")
    def search(q: str, limit: int = 20):
        return cards(rec_service.search(q, limit))

    @app.post("/session")
    def session(req: SessionReq):
        sid = session_store.create(req.liked, req.lam, req.k)
        return {"session_id": sid, "cards": feed_for(session_store.get(sid))}

    @app.post("/swipe")
    def swipe(req: SwipeReq):
        try:
            s = session_store.apply(req.session_id, req.book_id, req.action)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown session")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"cards": feed_for(s), "reading_list": cards(s.reading_list)}

    return app
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/api/test_app.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add book_recsys/api/app.py tests/api/test_app.py
git commit -m "Add FastAPI swipe endpoints (search/session/swipe) with injectable services"
```

---

## Task 6: Production wiring — `get_app()` factory (artifact-backed)

**Files:**
- Modify: `book_recsys/api/app.py`

This loads `models.joblib` / `catalog.parquet` / `embeddings.npy`, so it is excluded from
coverage with `# pragma: no cover` (the same convention the project uses for artifact/network
code). No unit test — verified by running the server (Task 8).

- [ ] **Step 1: Append `get_app` and its loader to `book_recsys/api/app.py`**

```python
def _find(name: str) -> str:  # pragma: no cover
    import glob
    for pat in (f"artifacts/{name}", f"../artifacts/{name}", f"../../artifacts/{name}"):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under artifacts/ (run notebooks/07_models.ipynb)")


def get_app() -> FastAPI:  # pragma: no cover
    """Production app for `uvicorn book_recsys.api.app:get_app --factory`."""
    import os

    import joblib
    import numpy as np
    import pandas as pd
    from fastapi.staticfiles import StaticFiles

    from book_recsys.ui.feed import FeedService
    from book_recsys.ui.service import RecommenderService

    models = joblib.load(_find("models.joblib"))
    catalog = pd.read_parquet(_find("catalog.parquet"))
    emb = np.load(_find("embeddings.npy"))

    hybrid = models["hybrid_cf_content"]
    rec_service = RecommenderService(catalog, {"hybrid": hybrid}, models["similar"])
    feed_service = FeedService(hybrid, emb, catalog["book_id"].tolist())

    app = create_app(rec_service, feed_service, SessionStore())
    web = os.path.join(os.path.dirname(__file__), "..", "ui", "web")
    app.mount("/", StaticFiles(directory=web, html=True), name="web")  # serves the SPA
    return app
```

- [ ] **Step 2: Verify existing tests still pass (no regression)**

Run: `pytest tests/api/ -v`
Expected: PASS (unchanged — `get_app` is not imported by tests).

- [ ] **Step 3: Commit**

```bash
git add book_recsys/api/app.py
git commit -m "Add artifact-backed get_app production factory + static mount"
```

---

## Task 7: Frontend — vanilla-JS swipe SPA

**Files:**
- Create: `book_recsys/ui/web/index.html`, `book_recsys/ui/web/style.css`, `book_recsys/ui/web/app.js`

No unit tests (thin shell, no logic). Verified by the manual smoke checklist in Task 8.

- [ ] **Step 1: Create `book_recsys/ui/web/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>📚 Book Swipe</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <header><h1>📚 Book Swipe</h1></header>

  <section id="onboarding">
    <p>Add a few books you love, then start swiping.</p>
    <input id="search" type="text" placeholder="Search a book…" autocomplete="off" />
    <ul id="results"></ul>
    <div id="seeds"></div>
    <button id="start" disabled>Start swiping →</button>
  </section>

  <section id="swipe" hidden>
    <div id="card-area"><div id="card"></div></div>
    <div id="buttons">
      <button data-action="dislike" title="Don't like (←? no: ↓)">👎 Not for me</button>
      <button data-action="want" title="Want to read (→)">🔖 Want to read</button>
      <button data-action="like" title="Read &amp; liked (←)">👍 Read &amp; liked</button>
      <button data-action="skip" title="Skip">⏭ Skip</button>
    </div>
    <aside id="reading-list"><h3>🔖 Reading list</h3><ul></ul></aside>
  </section>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `book_recsys/ui/web/style.css`**

```css
* { box-sizing: border-box; font-family: system-ui, sans-serif; }
body { margin: 0; background: #f5f5f7; color: #1d1d1f; }
header { padding: 1rem; text-align: center; }
#onboarding, #swipe { max-width: 480px; margin: 0 auto; padding: 1rem; }
#search { width: 100%; padding: .6rem; font-size: 1rem; border: 1px solid #ccc; border-radius: 8px; }
#results { list-style: none; padding: 0; margin: .3rem 0; }
#results li { padding: .5rem; cursor: pointer; border-radius: 6px; }
#results li:hover { background: #e8e8ed; }
#seeds .seed { display: inline-block; background: #d1e7ff; border-radius: 12px;
  padding: .2rem .6rem; margin: .2rem; font-size: .85rem; }
button { cursor: pointer; border: none; border-radius: 10px; padding: .6rem 1rem;
  font-size: 1rem; background: #0071e3; color: #fff; }
button:disabled { background: #aaa; cursor: default; }
#card-area { perspective: 1000px; height: 360px; display: flex;
  align-items: center; justify-content: center; }
#card { width: 300px; min-height: 300px; background: #fff; border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,.15); padding: 1.5rem; transition: transform .25s ease, opacity .25s ease; }
#card h2 { font-size: 1.2rem; margin: 0 0 .5rem; }
#card .author { color: #666; margin: 0 0 1rem; }
#card .desc { font-size: .9rem; color: #444; }
#buttons { display: flex; gap: .5rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0; }
#buttons button[data-action="dislike"] { background: #ff3b30; }
#buttons button[data-action="like"] { background: #34c759; }
#buttons button[data-action="want"] { background: #ff9500; }
#buttons button[data-action="skip"] { background: #8e8e93; }
#reading-list ul { list-style: none; padding: 0; }
#reading-list li { padding: .3rem 0; border-bottom: 1px solid #eee; font-size: .9rem; }
.swipe-left { transform: translateX(-400px) rotate(-20deg); opacity: 0; }
.swipe-right { transform: translateX(400px) rotate(20deg); opacity: 0; }
.swipe-down { transform: translateY(400px); opacity: 0; }
```

- [ ] **Step 3: Create `book_recsys/ui/web/app.js`**

```javascript
const seeds = [];
let sessionId = null;
let queue = [];

const $ = (id) => document.getElementById(id);

// --- onboarding: search + pick seeds ---
$("search").addEventListener("input", async (e) => {
  const q = e.target.value.trim();
  if (!q) { $("results").innerHTML = ""; return; }
  const res = await fetch(`/search?q=${encodeURIComponent(q)}`);
  const books = await res.json();
  $("results").innerHTML = "";
  for (const b of books) {
    const li = document.createElement("li");
    li.textContent = b.label;
    li.onclick = () => addSeed(b);
    $("results").appendChild(li);
  }
});

function addSeed(b) {
  if (seeds.some((s) => s.book_id === b.book_id)) return;
  seeds.push(b);
  const span = document.createElement("span");
  span.className = "seed";
  span.textContent = b.label;
  $("seeds").appendChild(span);
  $("search").value = "";
  $("results").innerHTML = "";
  $("start").disabled = seeds.length === 0;
}

// --- start session ---
$("start").onclick = async () => {
  const res = await fetch("/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ liked: seeds.map((s) => s.book_id), lam: 1.0, k: 10 }),
  });
  const body = await res.json();
  sessionId = body.session_id;
  queue = body.cards;
  $("onboarding").hidden = true;
  $("swipe").hidden = false;
  renderCard();
};

// --- swipe loop ---
function renderCard() {
  const card = $("card");
  if (!queue.length) { card.innerHTML = "<p>No more recommendations — swipe more or refine your taste!</p>"; return; }
  const b = queue[0];
  const [title, rest] = splitLabel(b.label);
  card.className = "";
  card.innerHTML = `<h2>${escapeHtml(title)}</h2><p class="desc">${escapeHtml(rest)}</p>`;
  card.dataset.bookId = b.book_id;
}

function splitLabel(label) {
  const i = label.indexOf(" — ");
  return i === -1 ? [label, ""] : [label.slice(0, i), label.slice(i + 3)];
}

async function swipe(action) {
  const b = queue[0];
  if (!b) return;
  const dir = { like: "swipe-left", want: "swipe-right", dislike: "swipe-down", skip: "swipe-down" }[action];
  $("card").classList.add(dir);
  const res = await fetch("/swipe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, book_id: b.book_id, action }),
  });
  const body = await res.json();
  queue = body.cards;
  renderReadingList(body.reading_list);
  setTimeout(renderCard, 250);
}

function renderReadingList(list) {
  const ul = $("reading-list").querySelector("ul");
  ul.innerHTML = "";
  for (const b of list) {
    const li = document.createElement("li");
    li.textContent = b.label;
    ul.appendChild(li);
  }
}

document.querySelectorAll("#buttons button").forEach((btn) =>
  (btn.onclick = () => swipe(btn.dataset.action)));

document.addEventListener("keydown", (e) => {
  const map = { ArrowLeft: "like", ArrowRight: "want", ArrowDown: "dislike" };
  if ($("swipe").hidden) return;
  if (map[e.key]) swipe(map[e.key]);
});

// drag-to-swipe (pointer events)
(() => {
  let startX = 0, startY = 0, dragging = false;
  const card = $("card");
  card.addEventListener("pointerdown", (e) => { dragging = true; startX = e.clientX; startY = e.clientY; });
  card.addEventListener("pointerup", (e) => {
    if (!dragging) return;
    dragging = false;
    const dx = e.clientX - startX, dy = e.clientY - startY;
    if (Math.abs(dx) < 60 && Math.abs(dy) < 60) return;
    if (dy > 60 && Math.abs(dy) > Math.abs(dx)) swipe("dislike");
    else if (dx < -60) swipe("like");
    else if (dx > 60) swipe("want");
  });
})();

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
```

- [ ] **Step 4: Commit**

```bash
git add book_recsys/ui/web/
git commit -m "Add vanilla-JS swipe SPA (onboarding, drag/buttons/keyboard, reading list)"
```

---

## Task 8: Verify end-to-end + coverage

**Files:** none (verification only)

- [ ] **Step 1: Full test suite + coverage of the new logic**

Run: `coverage run -m pytest tests/ui/test_feed.py tests/api/ && coverage report --show-missing --include="book_recsys/ui/feed.py,book_recsys/api/*"`
Expected: PASS; 100% for `feed.py`, `sessions.py`, `app.py` (the `get_app`/`_find` lines are `# pragma: no cover`).

- [ ] **Step 2: Run the server (manual smoke)**

Run: `uvicorn book_recsys.api.app:get_app --factory --reload`
Then open `http://127.0.0.1:8000/` and check:
- Search returns books; clicking adds a seed chip; "Start swiping" enables.
- A card appears; 👍/🔖/👎 buttons, arrow keys, and drag all advance the card.
- 🔖 (want) adds the book to the reading list.
- After a few 👎 on similar books, the feed visibly shifts away from that style.

- [ ] **Step 3: Style/type checks (match project conventions)**

Run: `yapf -ir book_recsys/ui/feed.py book_recsys/api/ --exclude '*ipynb*' && isort book_recsys/ui/feed.py book_recsys/api/ --line-width 99 && mypy book_recsys/ui/feed.py book_recsys/api/`
Expected: clean (fix any reported issues).

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -A book_recsys/ tests/
git commit -m "Format and type-clean swipe UI"
```

---

## Self-Review Notes

- **Spec coverage:** FeedService (§4) → Tasks 2–3; SessionStore (§3 state model) → Task 4;
  API `/search` `/session` `/swipe` (§5) → Task 5; production wiring + static mount (§3, §5)
  → Task 6; frontend (§6) → Task 7; testing (§7) → Tasks 2–5 + 8. Penalization, flat
  positives, reading list, exclusion-via-`lam`=0 all covered.
- **`lambda` → `lam`:** the API/JSON field is `lam` (Python keyword), noted at the top.
- **Out of scope (per spec):** persistence, real swipe library, auth — none added.
