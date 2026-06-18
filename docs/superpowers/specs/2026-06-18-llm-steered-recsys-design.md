# LLM-Steered Recsys — Design

**Date:** 2026-06-18
**Status:** Approved (design); pending implementation plan
**Scope:** Demo-only feature for the chat UI. **Not** wired into the benchmarking
study or the eval harness — no NDCG/Recall metrics, no model report row.

## Idea

Let the LLM **drive** the existing recommender instead of merely re-describing a
fixed candidate set. Each turn of a multi-turn conversation, the LLM reads the chat
and sets the recsys's knobs — how much to lean on the reader's past taste vs. a new
topic, what that topic is, what to avoid, and (on explicit request) a genre or a
"more like *X*" anchor. The recsys models do the recommending; the LLM picks the
parameters. The reader watches the recommendations re-steer as the conversation
evolves.

This is a conversational, agentic front-end over the recommenders we already have.
It reuses **trained models and cached artifacts only — zero new training.**

## Decisions (locked during brainstorming)

- **Demo-only.** No formal evaluation. Success = convincing, responsive behavior on
  curated scenarios in a live demo.
- **Multi-turn refinement.** A running `SteeringState` per session is updated each
  turn; recommendations refresh as the conversation progresses.
- **All four levers** are LLM-controlled: taste↔topic blend, topic extraction,
  negative ("avoid") constraints, and genre/anchor steer.
- **Mechanism: one lean LLM call per turn** (Approach A). The call is small (no
  40-book candidate list in the prompt), so it is fast once the model is warm. No
  iterative tool-loop (too slow/unreliable on local qwen), no second
  grounded-overview call (~2× latency).
- **Three fusion signals** for "like my reads": the trained collaborative model
  **and** the content/embedding retriever, blended with the topic signal.
- **Genre filtering is opt-in.** Applied only when the user explicitly names a genre;
  otherwise no genre filter.

## What the LLM actually controls — `SteeringState`

A dataclass stored in the session, returned in full by the LLM each turn:

| Field            | Type           | Meaning                                                        |
|------------------|----------------|----------------------------------------------------------------|
| `history_weight` | `float [0,1]`  | Blend: 1 = purely "like my reads", 0 = purely the topic         |
| `topic`          | `str \| None`  | Clean topic/theme string to embed (`by_text`)                   |
| `avoid`          | `list[str]`    | Themes to penalize ("too dark", "no romance")                   |
| `genre`          | `str \| None`  | Hard include-filter; set **only** on explicit user request      |
| `anchor_book`    | `str \| None`  | A named book → `similar.recommend`; resolved via catalog search |
| `reply`          | `str`          | One-line narration of what changed, shown in the chat           |

The state is the **running memory**: each turn the LLM is given the *prior* state plus
the last few messages, and returns the updated full state (full, not delta — simpler
and more robust). `parse_steering` falls back to the prior value for any field the LLM
omits or mangles, so recommendations never crash or reset unexpectedly. Users can undo
("forget the WWII thing") and the LLM clears the field.

## Architecture — one turn end to end

```
user message
   │
   ▼
[1] Steerer.update(recent_messages, prev_state, anchor_titles)
       → ONE qwen call → SteeringState{history_weight, topic, avoid[], genre, anchor_book, reply}
   │
   ▼
[2] SteeredRanker.rank(state, history_ids, seen, k)
       gather ranked lists (each capped at pool ≈ 200):
         L_cf     = hybrid_cf_content.recommend(history)      ┐ history signal
         L_hist   = retriever.by_history(history)             ┘
         L_topic  = retriever.by_text(state.topic)              topic signal
         L_anchor = similar.recommend(anchor_id)   (only if anchor_book resolved; joins history side)
       → weighted RRF (below)
       → exclude history ∪ seen
       → − avoid penalty
       → genre filter (only if state.genre set)
       → top-K book_ids
   │
   ▼
[3] render: state.reply + cards (RecommenderService.card) + the live SteeringState
```

### Fusion math (`SteeredRanker.rank`)

Weighted Reciprocal Rank Fusion, extending the existing `fusion.py`:

```
score(b) = Σ_list  w_list · 1 / (60 + rank_b_in_list)
```

The LLM's `history_weight = w` splits across the history-side signals; the topic gets
the remainder:

```
w_cf = w/2,   w_hist = w/2,   w_topic = (1 − w)
```

- `w = 1` → purely "like my reads"; `w = 0` → purely the topic.
- `L_anchor`, when present, joins the history side (shares its weight).
- A book missing from a list contributes 0 for that list — robust to short lists and to
  `topic = None` (then `w_topic` list is empty and the blend is pure history).

Then, in order:

1. **Exclude** `history ∪ seen` from the candidate union.
2. **Avoid penalty:** `encode(avoid phrase)` → for each candidate subtract
   `λ · max cosine(candidate_emb, avoid_vec)` over the avoid phrases. This is exactly
   `FeedService._max_sim_to_disliked`, but with a *text* vector instead of disliked
   book vectors. `λ` fixed at 1.0.
3. **Genre filter** only if `state.genre` is set (include-filter on the catalog
   `genre` column).
4. Sort by final score, take top-K.

## Special case — gift / "for someone else" (UC5)

In a gift query the session's reading history belongs to the **asker, not the
recipient**, so it is noise. Handled entirely through the existing levers, driven by
one instruction in the steering prompt:

> *If the request is a gift or for someone else, set `history_weight` near 0 — the
> session's reading history is the asker's, not the recipient's — and build `topic`
> from the recipient's described tastes. If the asker names a book the recipient loved,
> set `anchor_book`.*

So a gift query runs almost entirely on the **topic (`by_text`)** signal (plus the
`similar` anchor if a recipient-favorite is named), with the CF + content-history
signals suppressed. No new code — the weight simply slides to "topic."

## Limitations (state plainly; do not oversell)

- **No user/demographic model.** The dataset has no user age/demographics, and the
  book-side mood/age labels are **eval-only, not system input** (per the project
  spec). When the LLM "models an older reader," it is using its own **commonsense
  priors** to translate the recipient description into age-appropriate *theme/genre/
  reading-level terms*, packed into the `topic` string; retrieval then matches by
  content-embedding similarity. This is fuzzy, LLM-prior-driven theme inference — not
  a learned age→taste mapping. The spec/report must say so.
- **Content-blend caveat.** The topic blend lives in the `bge` embedding space; the
  trained CF model contributes only via history (it cannot ingest free text).

## Components (new — all small, isolated, testable)

- **`book_recsys/llm/steer.py`** — pure prompt/parse + the `Steerer`:
  - `build_steer_prompt(messages, prev_state, anchor_titles) -> str` — carries prior
    state + recent messages + anchor titles; instructs genre-null-by-default and the
    gift rule.
  - `parse_steering(raw, prev_state) -> SteeringState` — merge LLM output onto prior
    state; clamp `history_weight` to [0,1]; malformed → prior state (robust, like the
    existing `parse_overview`).
  - `Steerer(client)` with `.update(...) -> SteeringState` — the one LLM call.
  - `SteeringState` dataclass.
- **`book_recsys/llm/rank.py`** — `SteeredRanker(cf_model, retriever, similar,
  embeddings, book_ids, catalog_genre)` with `.rank(state, history_ids, seen, k)`.
  **Pure ranking logic, no LLM** — unit-testable with fakes.
- **Weighted RRF** — add a weighted variant to `book_recsys/llm/fusion.py`.

## API

- New `POST /steer {session_id, message} -> {reply, state, cards}`:
  pulls the session's `liked` (anchor) + stored steering state + recent messages, runs
  `Steerer` (1 LLM call) then `SteeredRanker`, persists the new state, appends the
  message to the session transcript, returns reply + live state + rendered cards.
- The existing `/chat` (grounded overview) endpoint is **left untouched** — it can
  become an optional "rich mode" (Approach B) later.
- `SessionStore` gains a `steering: SteeringState` field and a short rolling
  `messages: list` per session, with get/set methods.
- Heavy stack (encoder, FAISS, models) is built lazily and reused, as today; the faiss
  single-thread OpenMP fix already committed covers this path.

## UI

- The chat panel calls `/steer` instead of `/chat`.
- Each turn renders three things:
  1. the LLM **reply** (one-line narration),
  2. the recommended **cards** (`RecommenderService.card`),
  3. a **live steering panel** — the demo centerpiece — showing current knobs:
     a `history ▮▮▮▮▯ topic` bar, a topic chip, `avoid:` chips, a genre chip — visibly
     updating turn to turn.
- Messages accumulate so the audience sees the conversation steer the system.

## Error handling

- LLM/Ollama down → 503 (existing graceful path + the `log.exception` already added).
- Malformed steering JSON → fall back to **prior state**; recs stay stable; neutral
  default reply.
- No history **and** no topic yet → cannot recommend from nothing → reply asks for a
  favorite book or a topic.
- Avoid-encode failure → skip the penalty (degrade, don't fail).
- faiss OpenMP segfault → already fixed (`faiss.omp_set_num_threads(1)`).

## Testing

Matches project conventions — pure logic fully tested; network/IO excluded
(`# pragma: no cover`).

- `parse_steering`: full state; partial-merge-onto-prior; malformed → prior;
  `history_weight` clamp to [0,1]; `genre` stays `None` when absent.
- weighted RRF: weighting shifts order; `w=1`/`w=0` extremes; missing items contribute 0.
- `SteeredRanker.rank` with **fake** cf/retriever/similar (the `create_app` injection
  pattern): history/seen exclusion; avoid penalty demotes offenders; genre filter;
  anchor inclusion; top-K; `topic=None` → pure-history path.
- `build_steer_prompt`: carries prior state + recent messages + anchor titles;
  contains the genre-null-by-default and gift instructions.
- e2e: `create_app` with a fake client returning canned steering JSON → `/steer`
  returns the expected cards and state.
- Live `Steerer` LLM call + `_build_*` wiring → `# pragma: no cover`.

## Out of scope

- Eval harness wiring / offline metrics / model-report row.
- The 2-call grounded-overview "rich mode" (Approach B) and tool-calling loop
  (Approach C).
- Book-side age-label filtering (would cross the "labels are eval-only" line).
