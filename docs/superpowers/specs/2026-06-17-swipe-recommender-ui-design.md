# Swipe Recommender UI — Design

**Author:** Maya Deneva · **Date:** 2026-06-17
**Status:** Approved design, ready for implementation planning
**Branch:** `feature/swipe-ui`

## 1. Goal

A Bookist-style **swipe interface** over the existing recommenders. The user seeds a few
liked books, then swipes through an adapting feed:

- **swipe left** — *read & liked* → positive taste signal
- **swipe right** — *want to read* → positive taste signal **and** saved to a reading list
- **swipe down** — *don't like* → negative signal; the feed **moves away from books similar
  to it** (penalization, not just hiding that one book)

Each swipe re-queries the **learned hybrid** recommender (CF + content, the best UC1 method)
on the updated profile. This showcases UC1 as a live, interactive experience and reuses the
whole recommendation stack. It is a **value-add demo**, not a replacement for the required
Streamlit UI.

## 2. Scope

**In scope**
- FastAPI backend exposing search + feed endpoints that call into the package.
- A static single-page frontend (HTML/CSS/JS) with drag-to-swipe cards (buttons as fallback).
- A new, fully-tested `FeedService` holding all feed logic (candidate gen → negative
  penalization → exclude seen).
- Negative **penalization** (`hybrid_score − λ·similarity_to_disliked`) with plain exclusion
  as a baseline (λ knob).

**Out of scope (non-goals)**
- Persistence / user accounts / databases — sessions are **ephemeral, client-held**.
- A real swipe-gesture library — gestures are hand-rolled with pointer events + CSS.
- Auth, deployment, multi-user concurrency.
- Changes to the recommenders themselves (we consume `recommend` / `score_items` as-is).

## 3. Architecture

```
book_recsys/
  models/hybrid/learned.py     # existing LearnedHybridRecommender (consumed, unchanged)
  ui/service.py                # existing RecommenderService (search + labels, reused)
  ui/feed.py        (new)      # FeedService — feed logic, penalization, fully tested
  api/app.py        (new)      # FastAPI app: /search, /feed; thin, calls FeedService
  ui/web/           (new)      # static SPA: index.html, app.js, style.css
tests/
  ui/test_feed.py   (new)      # FeedService unit tests (TDD)
  api/test_app.py   (new)      # endpoint tests via FastAPI TestClient + fake service
```

**Principle (unchanged):** all logic lives in the package and is unit-tested; the FastAPI
file and the JS are thin shells. `FeedService` is independently testable with fakes — no
network, no torch in the test path.

**State model — client-driven (stateless server).** The browser holds the session
(`liked`, `disliked`, `reading_list`, `seen`) in memory/`sessionStorage` and sends the
relevant lists with each request. The server computes the next feed and returns it. No
session store, no cleanup, trivial to reason about — right for an ephemeral demo.

## 4. The feed logic (`FeedService`)

Constructed with: the hybrid recommender, the book embeddings (`np.ndarray`), a
`book_id → row` index map, and a `label(book_id)` formatter (from `RecommenderService`).

```
FeedService.next(liked, disliked, seen, k=10, lam=1.0, pool=200) -> list[card]
  1. candidates = hybrid.recommend(liked, pool)          # candidate generation (CF ∪ content)
  2. drop candidates in (seen ∪ liked ∪ disliked)
  3. base = hybrid.score_items(liked, candidates)        # hybrid score per candidate
     base = minmax_normalize(base)                        # -> [0,1] so λ is interpretable
  4. if disliked:
        pen[c] = max over d in disliked of cosine(emb[c], emb[d])   # similarity to nearest disliked
     else pen = 0
  5. final[c] = base[c] − lam · pen[c]
  6. return top-k by final, as cards {book_id, label}
```

- **`lam` (λ)** controls how hard the feed avoids books like the disliked ones.
  `lam = 0` → pure hybrid (no negative steering). Large `lam` → strong avoidance ≈ exclusion
  of look-alikes. This single knob spans "exclude-only baseline" → "full penalization", which
  also gives a clean before/after demo and an ablation for the report.
- **Why normalize `base`:** hybrid scores and cosine live on different scales; min-max over
  the candidate pool puts both in `[0,1]` so `λ` is meaningful and tunable.
- **Empty `disliked`** → penalty is zero → identical to the plain hybrid feed (clean default).
- Cosine uses L2-normalized embeddings (dot product), matching the existing FAISS convention.

## 5. API (FastAPI)

- `GET /search?q=<str>&limit=20` → `[{book_id, label}]` — onboarding seed search
  (delegates to `RecommenderService.search` + `label`).
- `POST /feed` → next cards. Body:
  ```json
  {"liked": [id...], "disliked": [id...], "seen": [id...], "k": 10, "lambda": 1.0}
  ```
  Response: `{"cards": [{"book_id": id, "label": "Title by Author — …"}]}`.
- Static frontend served from `book_recsys/ui/web/` (FastAPI `StaticFiles`).
- Run: `uvicorn book_recsys.api.app:app --reload`. Loads `models.joblib` + `catalog.parquet`
  + `embeddings.npy` once at startup (same artifacts the Streamlit UI uses).

## 6. Frontend (vanilla JS SPA)

- **Onboarding screen:** search box → pick a few seed books → "Start swiping".
- **Swipe screen:** a card stack showing the current book (title, author, description
  snippet). Drag gestures via pointer events + CSS transforms:
  left → liked, right → want-to-read, down → dislike; on-screen buttons mirror each
  (keyboard: ←/→/↓). After each swipe, update the client lists and `POST /feed` for the next
  batch (prefetch a small queue so it feels instant).
- **Reading list panel:** the right-swiped books accumulate in a visible list (the takeaway).
- Tasteful styling — card shadows, swipe animations, color-coded directions. No build step.

## 7. Testing

- **`FeedService` (TDD, 100% of the logic):** with a fake recommender + tiny embedding
  matrix — exclusion of `seen ∪ liked ∪ disliked`; a candidate near a disliked book ranks
  **below** an otherwise-equal candidate that isn't; `lam=0` reproduces the plain hybrid
  order; empty `disliked` == plain hybrid; fewer than `k` candidates handled.
- **API (`TestClient`):** `/search` and `/feed` shapes and wiring, against a **fake**
  `FeedService` (no torch/artifacts in the test path).
- **Frontend:** not unit-tested (thin shell, no logic by design); a manual smoke checklist
  in the PR description.

## 8. Risks & mitigations

- **λ scale / score normalization:** mitigated by min-max normalizing the candidate hybrid
  scores so `λ` is interpretable; expose `λ` in the UI for live tuning.
- **`score_items` cost over a 200-candidate pool per swipe:** small (vectorized); prefetch a
  queue client-side so latency is hidden.
- **Cold start (first 1–2 swipes):** the hybrid leans on content/embedding similarity early,
  which is exactly its strength; acceptable.
- **Time (deadline 2026-06-18):** the required Streamlit UI already ships; this is additive.
  Backend + `FeedService` + a functional (if simple) frontend is the MVP; gesture polish is
  the first thing to cut.
