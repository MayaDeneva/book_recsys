# Steered-Recsys "Why" Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show, per recommended card in the steered chat, a signal-based "why" label derived from which retrieval signal(s) surfaced the book — with zero extra LLM calls.

**Architecture:** `SteeredRanker` gains `rank_with_reasons(...)` that runs the existing fusion pipeline while tracking each candidate's signal provenance, then composes a combined reason string. `rank(...)` becomes a thin delegate (behavior unchanged). The `/steer` endpoint calls the new method and puts the reason on each card; the SPA already renders a card `reason`.

**Tech Stack:** Python 3.12, FastAPI, numpy, vanilla-JS SPA.

## Global Constraints

- Python 3.12; `yapf` (column_limit 99, pep8 base), `isort` (line width 99), `mypy` clean for the feature files.
- Test-first (TDD); 100% coverage bar (`coverage report --fail-under=100`); network/IO excluded via `# pragma: no cover`.
- Branch: `feature/llm-chat`, in the worktree `/Users/mayadeneva/Documents/uni/book_recsys/.claude/worktrees/llm-steered`. Run all git/pytest from there; commit only the exact files each task names (never `git add -A`).
- Reason clauses, fixed order, joined with ` · `, sentence-cased: history → `similar to your reading history`; topic → `matches your topic: {topic}`; anchor → `like {anchor_book}`. The two history signals (CF, content-history) collapse to ONE "reading history" clause. No clause for genre/avoid. No matching signal → empty reason.
- Do not change the steering state, prompt, or fusion math. `rank(...)` must keep its exact current id-ordering behavior (existing `test_rank.py` stays green).

---

## File Structure

- **Modify** `book_recsys/llm/rank.py` — add `rank_with_reasons` + `_reason`; make `rank` delegate.
- **Modify** `tests/llm/test_rank.py` — tests for `rank_with_reasons` (existing `rank` tests unchanged).
- **Modify** `book_recsys/api/app.py` — `/steer` calls `rank_with_reasons`, puts `reason` on each card.
- **Modify** `tests/api/test_app.py` — `/steer` fakes expose `rank_with_reasons`; assert card `reason`.
- **Modify** `book_recsys/ui/web/app.js` — render the server-provided card `reason`.

---

### Task 1: `rank_with_reasons` in SteeredRanker

**Files:**
- Modify: `book_recsys/llm/rank.py`
- Test: `tests/llm/test_rank.py`

**Interfaces:**
- Consumes: existing `SteeredRanker` internals (`self._cf`, `self._retriever`, `self._similar`, `self._genre`, `self._pool`, `self._lam`, `self._avoid_penalty`), `weighted_reciprocal_rank_fusion`, `minmax`.
- Produces:
  - `SteeredRanker.rank_with_reasons(state, history_ids, seen, k=10, anchor_id=None) -> list[tuple]` returning `[(book_id, reason_str), ...]` (top-k, same order `rank` would return).
  - `SteeredRanker.rank(...)` now delegates: `[b for b, _ in self.rank_with_reasons(...)]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/llm/test_rank.py`:

```python
def test_rank_with_reasons_topic_only():
    pairs = _ranker().rank_with_reasons(SteeringState(history_weight=0.0, topic="WWII subs"),
                                        history_ids=[], seen=set())
    reasons = dict(pairs)
    assert pairs[0][0] == "d"
    assert reasons["d"] == "Matches your topic: WWII subs"


def test_rank_with_reasons_history_only_single_clause():
    # history surfaces from BOTH cf and by_history -> collapses to ONE clause.
    pairs = _ranker().rank_with_reasons(SteeringState(history_weight=1.0), history_ids=["x"],
                                        seen=set())
    assert pairs, "expected some history-based picks"
    assert all(r == "Similar to your reading history" for _, r in pairs)


def test_rank_with_reasons_anchor_clause():
    pairs = _ranker().rank_with_reasons(
        SteeringState(history_weight=1.0, anchor_book="Dune"), history_ids=[], seen=set(),
        anchor_id="z")
    assert dict(pairs)["e"] == "Like Dune"


def test_rank_with_reasons_combines_signals_in_order():
    # 'c' appears in by_text ("d","e","c") AND similar ("e","c") -> topic + anchor clauses.
    pairs = _ranker().rank_with_reasons(
        SteeringState(history_weight=0.0, topic="cozy", anchor_book="Dune"),
        history_ids=[], seen=set(), anchor_id="z")
    reasons = dict(pairs)
    assert reasons["c"] == "Matches your topic: cozy · like Dune"


def test_rank_delegates_to_rank_with_reasons():
    state = SteeringState(history_weight=0.5, topic="x")
    ids = _ranker().rank(state, history_ids=[], seen=set())
    pairs = _ranker().rank_with_reasons(state, history_ids=[], seen=set())
    assert ids == [b for b, _ in pairs]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/llm/test_rank.py -v -k reasons`
Expected: FAIL — `AttributeError: 'SteeredRanker' object has no attribute 'rank_with_reasons'`.

- [ ] **Step 3: Implement**

In `book_recsys/llm/rank.py`, replace the existing `rank` method with the following three methods (the pipeline body is unchanged except it now records provenance and returns `(id, reason)` pairs; `rank` delegates):

```python
    def rank_with_reasons(self, state, history_ids, seen, k: int = 10, anchor_id=None) -> list:
        history_ids = list(history_ids)
        w = state.history_weight
        weighted_lists: list = []
        sources: dict = {}  # book_id -> set of signal labels it appeared in

        def add(ranked, weight, label):
            weighted_lists.append((ranked, weight))
            for book_id in ranked:
                sources.setdefault(book_id, set()).add(label)

        if history_ids and w > 0:
            add(self._cf.recommend(history_ids, self._pool), w / 2, "history")
            add(self._retriever.by_history(history_ids, self._pool), w / 2, "history")
        if anchor_id is not None:
            add(self._similar.recommend(anchor_id, self._pool), 0.5, "anchor")
        if state.topic and (1 - w) > 0:
            add(self._retriever.by_text(state.topic, self._pool), 1 - w, "topic")
        if not weighted_lists:
            return []

        fused = weighted_reciprocal_rank_fusion(weighted_lists)
        exclude = set(history_ids) | set(seen)
        candidates = [b for b in fused if b not in exclude]
        if state.genre:
            g = state.genre.lower()
            candidates = [b for b in candidates if g in str(self._genre.get(b, "")).lower()]
        if not candidates:
            return []

        # base score = inverse fused rank (earlier = higher), min-max normalized.
        base = minmax(np.array([-i for i in range(len(candidates))], dtype="float64"))
        if state.avoid:
            # base is min-max'd to [0,1]; the penalty is a raw cosine — different scales
            # by design (mirrors FeedService's disliked penalty). lam tunes how hard an
            # avoided theme is pushed down; do NOT "normalize" the penalty to match base.
            base = base - self._lam * self._avoid_penalty(candidates, state.avoid)
        order = np.argsort(-base, kind="stable")[:k]
        return [(candidates[i], self._reason(sources.get(candidates[i], set()), state))
                for i in order]

    def rank(self, state, history_ids, seen, k: int = 10, anchor_id=None) -> list:
        return [b for b, _ in self.rank_with_reasons(state, history_ids, seen, k, anchor_id)]

    @staticmethod
    def _reason(signals, state) -> str:
        clauses = []
        if "history" in signals:
            clauses.append("similar to your reading history")
        if "topic" in signals:
            clauses.append(f"matches your topic: {state.topic}")
        if "anchor" in signals:
            clauses.append(f"like {state.anchor_book}")
        text = " · ".join(clauses)
        return text[:1].upper() + text[1:]  # sentence-case; empty text -> "" cleanly
```

Note: the `_avoid_penalty` method stays exactly as it is (do not touch it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/llm/test_rank.py -v`
Expected: PASS (the new `reasons` tests AND all pre-existing `rank` tests via delegation).

- [ ] **Step 5: mypy + commit**

Run: `python -m mypy book_recsys/llm/rank.py` (clean).
```bash
git add book_recsys/llm/rank.py tests/llm/test_rank.py
git commit -m "rank: add rank_with_reasons (signal provenance -> per-book why); rank delegates"
```

---

### Task 2: Surface reasons through /steer and the UI

**Files:**
- Modify: `book_recsys/api/app.py`
- Modify: `tests/api/test_app.py`
- Modify: `book_recsys/ui/web/app.js`

**Interfaces:**
- Consumes: `SteeredRanker.rank_with_reasons(state, history_ids, seen, k, anchor_id) -> list[tuple]` (Task 1); the `_LazySteer` wrapper must forward it.
- Produces: `/steer` cards each carry a `reason` string; the SPA renders it.

- [ ] **Step 1: Update the /steer e2e fakes + assertion (failing test)**

In `tests/api/test_app.py`, change the `/steer` fakes from `rank(...)` to `rank_with_reasons(...)` returning `(id, reason)` pairs, and assert the reason on a card.

Replace the `_Ranker` class:
```python
class _Ranker:

    def rank_with_reasons(self, state, history_ids, seen, k=10, anchor_id=None):
        return [("x1", "Matches your topic: WWII"), ("x2", "")]
```

In `test_steer_returns_reply_state_and_cards`, after the existing assertions, add:
```python
    assert body["cards"][0]["reason"] == "Matches your topic: WWII"
    assert body["cards"][1]["reason"] == ""
```

In `test_steer_resolves_anchor_book_to_search_hit` and
`test_steer_anchor_book_with_no_search_hit_is_none`, rename each `_RecordingRanker.rank`
method to `rank_with_reasons` and have it return pairs:
```python
        def rank_with_reasons(self, state, history_ids, seen, k=10, anchor_id=None):
            self.anchor_id = anchor_id
            return [("x1", "")]
```
(for the no-hit test return `[]`, matching its current `return []`).

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/api/test_app.py -v -k steer`
Expected: FAIL — the handler still calls `ranker.rank(...)`, so the fakes (now only defining `rank_with_reasons`) raise `AttributeError`, or the `reason` assertion fails.

- [ ] **Step 3: Update the /steer handler and the lazy wrapper**

In `book_recsys/api/app.py`, in the `steer` handler, replace the `book_ids = ranker.rank(...)` line and the return's `cards` list with:

```python
        pairs = ranker.rank_with_reasons(state, session.liked, session.seen, k=req.k,
                                         anchor_id=anchor_id)
        return {
            "session_id": sid,
            "reply": state.reply,
            "state": asdict(state),
            "cards": [{**rec_service.card(b), "reason": reason} for b, reason in pairs],
        }
```

In the `_LazySteer` class (`# pragma: no cover`), add a `rank_with_reasons` delegator next to the existing `rank`:
```python
    def rank_with_reasons(self, *args, **kwargs):
        return self._ensure()[1].rank_with_reasons(*args, **kwargs)
```

- [ ] **Step 4: Run the API tests**

Run: `python -m pytest tests/api/test_app.py -v`
Expected: PASS (all `/steer` tests, including the new `reason` assertions).

- [ ] **Step 5: Render the reason in the SPA**

In `book_recsys/ui/web/app.js`, in `askChat`, the cards from `/steer` now already carry a
`reason`, so stop overwriting it with `""`. Replace:
```javascript
    renderOverview({ intro: data.reply, categories: [{ header: "Picks", items:
      data.cards.map((c) => ({ ...c, reason: "" })) }] });
```
with:
```javascript
    renderOverview({ intro: data.reply, categories: [{ header: "Picks", items: data.cards }] });
```

- [ ] **Step 6: Full suite + coverage + lint + commit**

Run:
```bash
python -m pytest -q
coverage run -m pytest >/dev/null 2>&1 && coverage report --fail-under=100 | tail -2
python -m mypy book_recsys/api/app.py
yapf --diff book_recsys/api/app.py tests/api/test_app.py
```
Expected: all tests pass; coverage 100%; mypy shows no NEW errors (only the pre-existing
pandas/joblib stub errors inside `# pragma: no cover` `get_app`); yapf diff empty.

```bash
git add book_recsys/api/app.py tests/api/test_app.py book_recsys/ui/web/app.js
git commit -m "steer: surface per-book why labels through /steer + render in the SPA"
```

---

## Self-Review

**Spec coverage:**
- Signal-based reasons, no LLM → Task 1 (`rank_with_reasons` + `_reason`, pure). ✓
- Combine all signals, fixed order, ` · `, sentence-cased → Task 1 `_reason`. ✓
- History CF+content collapse to one clause → Task 1 (set of labels; both add `"history"`). ✓
- Genre/avoid produce no clause → Task 1 (`_reason` only checks history/topic/anchor). ✓
- Empty reason when no signal → Task 1 (`_reason` returns `""`). ✓
- `rank` behavior unchanged → Task 1 delegation + `test_rank_delegates_to_rank_with_reasons`. ✓
- `/steer` cards carry reason → Task 2 handler. ✓
- SPA renders reason → Task 2 Step 5. ✓
- Tests, 100% coverage, no LLM → Tasks 1–2. ✓

**Placeholder scan:** none; every code step has complete code.

**Type consistency:** `rank_with_reasons(state, history_ids, seen, k=10, anchor_id=None) -> list[tuple]` identical in Task 1 def, Task 2 handler call, the `_LazySteer` delegator, and the test fakes. `_reason(signals, state) -> str` consistent. `rank` delegation signature matches the original.
