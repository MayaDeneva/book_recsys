# Steered-Recsys "Why" Labels — Design

**Date:** 2026-06-18
**Status:** Approved (design); pending implementation plan
**Branch:** `feature/llm-chat`
**Builds on:** `2026-06-18-llm-steered-recsys-design.md` (the LLM-steered chat feature)

## Idea

Show, per recommended card in the steered chat, **why** the recommender surfaced it —
without adding any LLM latency. The reason is derived from *which retrieval signal(s)*
found the book during fusion (signal provenance), not from a second LLM call. This is
"Approach #2 (signal-based reasons)" from the steered-recsys discussion; the LLM-prose
alternative (#1) was rejected because it would roughly double per-turn latency on local
qwen.

## Decision

- **Signal-based, zero extra LLM calls.** Reasons come from the ranker's fusion
  provenance, computed in pure Python.
- **Combine all matched signals** into one reason string (the user's choice), in a
  fixed order, joined with ` · ` and sentence-cased.

## Reason strings (fixed order)

For each surfaced book, include a clause for every positive signal whose ranked list
contained it:

| Signal (list)                              | Clause                              |
|--------------------------------------------|-------------------------------------|
| CF `recommend(history)` OR `by_history`    | `similar to your reading history`   |
| topic `by_text(topic)`                     | `matches your topic: {topic}`       |
| anchor `similar.recommend(anchor_id)`      | `like {anchor_book}`                |

- Clauses are emitted in the order above, joined with ` · `, and the whole string is
  sentence-cased (first character upper). Example:
  `"Similar to your reading history · matches your topic: WWII submarines"`.
- The two history signals (CF and content-history) collapse to a **single** "reading
  history" clause (never duplicated).
- Genre is a filter and avoid is a penalty — neither produces a positive clause.
- A book with no matching positive signal (should not occur — every returned book came
  from some list) gets an empty reason and renders as just the card, as today.

## Components

- **`book_recsys/llm/rank.py`** — add
  `rank_with_reasons(state, history_ids, seen, k=10, anchor_id=None) -> list[tuple]`
  returning `[(book_id, reason_str), ...]`. It performs the same fusion → exclusion →
  genre filter → avoid penalty → top-k pipeline as `rank`, but tracks, per candidate,
  the set of signals (`"history"`, `"topic"`, `"anchor"`) whose list contained it, and
  composes the reason for the final top-k. The existing
  `rank(...)` becomes a thin delegate: `return [b for b, _ in self.rank_with_reasons(...)]`
  — its behavior and tests are unchanged.
- **`book_recsys/api/app.py`** — the `/steer` handler calls `rank_with_reasons` and
  returns cards carrying the reason:
  `[{**rec_service.card(b), "reason": reason} for b, reason in pairs]`. Response shape
  is otherwise unchanged (`{session_id, reply, state, cards}`); each card now has a
  non-empty `reason` when a signal matched.
- **`book_recsys/ui/web/app.js`** — in `askChat`, stop hardcoding `reason: ""`; use each
  card's server-provided `reason`. `renderOverview` already renders the `reason` line, so
  no other UI/CSS change is needed.

## Data flow (one turn, unchanged except the ranker call)

```
/steer → steerer.update(...) → state
       → ranker.rank_with_reasons(state, liked, seen, k, anchor_id)
            fuse signals (tracking provenance per candidate)
            → exclude history∪seen → genre filter → avoid penalty → top-k
            → [(book_id, "<combined reason>"), ...]
       → cards = [{**card(b), "reason": r} for b, r in pairs]
```

## Error handling

- No positive signal for a book → empty reason (renders as the bare card).
- All existing `/steer` error paths (503 on steerer failure / unconfigured) are
  unchanged.

## Testing (pure logic; 100% coverage; no LLM)

- `rank_with_reasons` with the existing `test_rank.py` fakes:
  - topic-only (history_weight 0, empty history) → reason contains `matches your topic: x`.
  - history-only → reason is `Similar to your reading history` (CF + by_history collapse
    to one clause, not duplicated).
  - anchor present → reason contains `like {anchor_book}`.
  - multi-signal book → combined string with clauses in the fixed order, ` · `-joined.
  - top-k length and ids match `rank(...)` for the same inputs (delegation parity).
- `rank(...)` existing tests stay green (delegation preserves id ordering).
- `/steer` e2e in `test_app.py`: a fake ranker exposing `rank_with_reasons` → assert the
  returned card includes the expected `reason`.

## Out of scope

- LLM-written prose reasons (Approach #1) and any second LLM call.
- Explaining *absence* (why avoided/filtered books are missing).
- Any change to the steering state, prompt, or fusion math.
