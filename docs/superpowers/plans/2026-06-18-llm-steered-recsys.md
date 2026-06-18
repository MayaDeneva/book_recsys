# LLM-Steered Recsys Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-turn chat where the LLM reads the conversation each turn and sets the recommender's knobs (taste↔topic blend, topic, avoid, opt-in genre/anchor); the existing trained models produce the recommendations.

**Architecture:** One lean LLM call per turn returns a `SteeringState` (a small JSON of knobs). A pure `SteeredRanker` then fuses three ranked lists — collaborative `recommend(history)`, content `by_history(history)`, content `by_text(topic)` — with weighted Reciprocal Rank Fusion, applies a text-vector "avoid" penalty and an opt-in genre filter, and returns top-K book ids. A new `POST /steer` endpoint orchestrates this over per-session state; the static SPA renders the reply, the cards, and a live knob panel.

**Tech Stack:** Python 3.12, FastAPI, faiss, sentence-transformers (bge-small, 384-d), LiteLLM→Ollama (qwen2.5:7b), numpy, vanilla JS SPA.

## Global Constraints

- Python 3.12; format with `yapf` (column_limit 99, pep8 base), `isort` line width 99, `mypy` clean.
- Test-first (TDD). Pure logic must have unit tests; **100% coverage bar** (`coverage report --fail-under=100`). Network/IO boundaries are excluded with `# pragma: no cover`, matching existing code (`book_recsys/llm/clients.py`, the `_build_*`/`get_app` functions in `book_recsys/api/app.py`).
- Recommender artifacts are cached on disk under `artifacts/`; **never recompute embeddings** — load `embeddings.npy`. The query encoder must stay `BAAI/bge-small-en-v1.5` (384-d) to match the cached catalog.
- This is a **demo-only** feature: no eval-harness wiring, no metrics, no model-report row.
- The existing `/chat` (grounded overview) endpoint must remain untouched.
- faiss must run single-threaded in any process that also loads torch + `models.joblib` (`faiss.omp_set_num_threads(1)`) — already done in `_build_overview`; replicate in the new steer builder.

---

## File Structure

- **Modify** `book_recsys/llm/fusion.py` — add `weighted_reciprocal_rank_fusion`.
- **Create** `book_recsys/llm/steer.py` — `SteeringState` dataclass, `parse_steering`, `build_steer_prompt`, `Steerer` (the one LLM call).
- **Create** `book_recsys/llm/rank.py` — `SteeredRanker` (pure fusion/penalty/filter ranking).
- **Modify** `book_recsys/api/sessions.py` — add `steering` + `messages` to `Session`; `ensure`/`append_message`/`set_steering` on `SessionStore`.
- **Modify** `book_recsys/api/app.py` — `create_app` gains `steerer`/`ranker` params; add `POST /steer`; add lazy `_build_steer` + wire in `get_app`.
- **Modify** `book_recsys/ui/web/app.js`, `index.html`, `style.css` — point chat at `/steer`, render the live steering panel.
- **Create** tests: `tests/llm/test_steer.py`, `tests/llm/test_rank.py`; **modify** `tests/llm/test_fusion.py`, `tests/api/test_sessions.py`, `tests/api/test_app.py` (match existing names if different — see each task).

---

### Task 1: Weighted Reciprocal Rank Fusion

**Files:**
- Modify: `book_recsys/llm/fusion.py`
- Test: `tests/llm/test_fusion.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `weighted_reciprocal_rank_fusion(weighted_lists, k: int = 60) -> list` where `weighted_lists` is a list of `(ranked_list, weight)` tuples. `score(item) = Σ weight · 1/(k + rank + 1)` (rank 0-indexed); returns items sorted by descending score, ties keep first-seen order.

- [ ] **Step 1: Write the failing tests**

Append to `tests/llm/test_fusion.py`:

```python
from book_recsys.llm.fusion import weighted_reciprocal_rank_fusion


def test_weighted_rrf_weight_one_list_dominates():
    # 'a' leads list A (weight 10), 'b' leads list B (weight 1) -> 'a' first.
    out = weighted_reciprocal_rank_fusion([(["a", "b"], 10.0), (["b", "a"], 1.0)])
    assert out[0] == "a"


def test_weighted_rrf_zero_weight_list_ignored():
    # Topic list has weight 0 -> only the history list decides order.
    out = weighted_reciprocal_rank_fusion([(["h1", "h2"], 1.0), (["t1", "t2"], 0.0)])
    assert out[:2] == ["h1", "h2"]
    assert set(out) == {"h1", "h2"}  # zero-weight items still contribute 0, never rank above


def test_weighted_rrf_missing_items_contribute_zero():
    out = weighted_reciprocal_rank_fusion([(["a"], 1.0), (["b"], 1.0)])
    assert set(out) == {"a", "b"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/llm/test_fusion.py -v -k weighted`
Expected: FAIL — `ImportError: cannot import name 'weighted_reciprocal_rank_fusion'`.

- [ ] **Step 3: Implement**

Append to `book_recsys/llm/fusion.py`:

```python
def weighted_reciprocal_rank_fusion(weighted_lists, k: int = 60) -> list:
    """Weighted RRF. score(item) = sum weight * 1/(k + rank + 1), rank 0-indexed.

    weighted_lists: iterable of (ranked_list, weight). Higher fused score ranks
    first; ties keep first-seen order. Items absent from a list contribute 0 there.
    """
    scores: dict = {}
    for ranked, weight in weighted_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=lambda item: -scores[item])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/llm/test_fusion.py -v`
Expected: PASS (all, including the pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add book_recsys/llm/fusion.py tests/llm/test_fusion.py
git commit -m "Add weighted reciprocal rank fusion"
```

---

### Task 2: SteeringState + parse_steering

**Files:**
- Create: `book_recsys/llm/steer.py`
- Test: `tests/llm/test_steer.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SteeringState` dataclass: `history_weight: float = 1.0`, `topic: str | None = None`, `avoid: list = []`, `genre: str | None = None`, `anchor_book: str | None = None`, `reply: str = ""`.
  - `parse_steering(raw: str, prev: SteeringState) -> SteeringState`. Contract: if no JSON object is found or JSON is invalid → return a copy of `prev`. Otherwise, per field: **key present (including JSON `null`)** → use the parsed value (`null`/`""` → `None`/`[]`); **key absent** → keep `prev`'s value. `history_weight` is clamped to `[0,1]`; a present-but-unparseable `history_weight` keeps `prev`. `reply` defaults to `""` when absent (it is not persistent memory).

- [ ] **Step 1: Write the failing tests**

Create `tests/llm/test_steer.py`:

```python
from dataclasses import replace

from book_recsys.llm.steer import SteeringState, parse_steering


def test_parse_full_state_overrides_prev():
    prev = SteeringState(history_weight=1.0)
    raw = ('{"history_weight": 0.6, "topic": "WWII submarines", "avoid": ["too dark"], '
           '"genre": "history", "anchor_book": "Das Boot", "reply": "Shifting toward WWII."}')
    out = parse_steering(raw, prev)
    assert out == SteeringState(history_weight=0.6, topic="WWII submarines",
                                avoid=["too dark"], genre="history",
                                anchor_book="Das Boot", reply="Shifting toward WWII.")


def test_parse_absent_key_keeps_prev_present_null_clears():
    prev = SteeringState(history_weight=0.5, topic="sailing", genre="history")
    # topic absent -> kept; genre explicit null -> cleared.
    out = parse_steering('{"genre": null, "reply": "ok"}', prev)
    assert out.topic == "sailing"
    assert out.genre is None
    assert out.history_weight == 0.5
    assert out.reply == "ok"


def test_parse_clamps_history_weight():
    out = parse_steering('{"history_weight": 1.7}', SteeringState())
    assert out.history_weight == 1.0
    out = parse_steering('{"history_weight": -3}', SteeringState())
    assert out.history_weight == 0.0


def test_parse_bad_history_weight_keeps_prev():
    out = parse_steering('{"history_weight": "lots"}', SteeringState(history_weight=0.4))
    assert out.history_weight == 0.4


def test_parse_avoid_sanitized_to_str_list():
    out = parse_steering('{"avoid": ["dark", 5, "", "  romance "]}', SteeringState())
    assert out.avoid == ["dark", "romance"]


def test_parse_empty_topic_string_becomes_none():
    out = parse_steering('{"topic": "   "}', SteeringState(topic="old"))
    assert out.topic is None


def test_parse_no_json_returns_copy_of_prev():
    prev = SteeringState(history_weight=0.3, topic="x")
    out = parse_steering("just prose, no json", prev)
    assert out == prev and out is not prev


def test_parse_invalid_json_returns_copy_of_prev():
    prev = SteeringState(topic="x")
    out = parse_steering("{not valid}", prev)
    assert out == prev and out is not prev


def test_reply_defaults_empty_when_absent():
    out = parse_steering('{"topic": "y"}', SteeringState(reply="old reply"))
    assert out.reply == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/llm/test_steer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'book_recsys.llm.steer'`.

- [ ] **Step 3: Implement `SteeringState` + `parse_steering`**

Create `book_recsys/llm/steer.py`:

```python
"""LLM-driven steering: the LLM reads the chat and emits the recsys's knobs.

`SteeringState` is the running, per-session memory. `parse_steering` turns one LLM
JSON reply into the next state (robust to malformed output); `build_steer_prompt`
builds the request; `Steerer` makes the single LLM call per turn.
"""
import json
import re
from dataclasses import dataclass, field, replace


@dataclass
class SteeringState:
    history_weight: float = 1.0  # 1 = purely "like my reads", 0 = purely the topic
    topic: str | None = None
    avoid: list = field(default_factory=list)
    genre: str | None = None  # hard include-filter; set only on explicit request
    anchor_book: str | None = None  # a named book -> similar.recommend
    reply: str = ""  # one-line narration for the chat (not persistent memory)


def _clean_str(value):
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def parse_steering(raw: str, prev: SteeringState) -> SteeringState:
    """Merge one LLM JSON reply onto `prev`. Absent key -> keep prev; present (incl.
    null) -> use parsed; whole-parse failure -> a copy of prev. See module/spec."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return replace(prev)
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return replace(prev)
    if not isinstance(obj, dict):
        return replace(prev)

    history_weight = prev.history_weight
    if "history_weight" in obj:
        try:
            history_weight = min(1.0, max(0.0, float(obj["history_weight"])))
        except (TypeError, ValueError):
            history_weight = prev.history_weight

    topic = _clean_str(obj["topic"]) if "topic" in obj else prev.topic
    genre = _clean_str(obj["genre"]) if "genre" in obj else prev.genre
    anchor_book = _clean_str(obj["anchor_book"]) if "anchor_book" in obj else prev.anchor_book

    if "avoid" in obj:
        raw_avoid = obj["avoid"] if isinstance(obj["avoid"], list) else []
        avoid = [s.strip() for s in raw_avoid if isinstance(s, str) and s.strip()]
    else:
        avoid = list(prev.avoid)

    reply = _clean_str(obj.get("reply")) or "" if "reply" in obj else ""

    return SteeringState(history_weight=history_weight, topic=topic, avoid=avoid,
                         genre=genre, anchor_book=anchor_book, reply=reply)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/llm/test_steer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add book_recsys/llm/steer.py tests/llm/test_steer.py
git commit -m "Add SteeringState + parse_steering (robust LLM-knob parsing)"
```

---

### Task 3: build_steer_prompt + Steerer

**Files:**
- Modify: `book_recsys/llm/steer.py`
- Test: `tests/llm/test_steer.py`

**Interfaces:**
- Consumes: `SteeringState`, `parse_steering` (Task 2); a client with `.complete(prompt: str) -> str` (the existing `LiteLLMClient`).
- Produces:
  - `build_steer_prompt(messages: list, prev: SteeringState, anchor_titles: list) -> str`. `messages` is a list of `{"role", "text"}`. The prompt embeds the prior state, the recent messages, and the anchor titles, and instructs: leave `genre` null unless the user explicitly names one; on a gift / "for someone else" request set `history_weight` near 0 and build `topic` from the recipient's tastes.
  - `Steerer(client)` with `.update(messages, prev, anchor_titles) -> SteeringState` = `parse_steering(client.complete(build_steer_prompt(...)), prev)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/llm/test_steer.py`:

```python
from book_recsys.llm.steer import Steerer, build_steer_prompt


def test_build_steer_prompt_includes_state_messages_titles_and_rules():
    prev = SteeringState(history_weight=0.7, topic="sailing", avoid=["gore"])
    prompt = build_steer_prompt([{"role": "user", "text": "but about WWII"}], prev,
                                ["Moby-Dick", "The Old Man and the Sea"])
    assert "0.7" in prompt and "sailing" in prompt and "gore" in prompt
    assert "but about WWII" in prompt
    assert "Moby-Dick" in prompt
    assert "genre" in prompt.lower()  # the null-unless-explicit rule
    assert "gift" in prompt.lower()   # the gift / for-someone-else rule


class _FakeClient:
    def __init__(self, raw):
        self.raw = raw
        self.prompt = None

    def complete(self, prompt):
        self.prompt = prompt
        return self.raw


def test_steerer_update_parses_client_reply_onto_prev():
    client = _FakeClient('{"history_weight": 0.4, "topic": "WWII", "reply": "ok"}')
    out = Steerer(client).update([{"role": "user", "text": "WWII please"}],
                                 SteeringState(topic="sailing"), [])
    assert out.history_weight == 0.4
    assert out.topic == "WWII"
    assert out.reply == "ok"
    assert "WWII please" in client.prompt  # the message reached the prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/llm/test_steer.py -v -k "prompt or steerer"`
Expected: FAIL — `ImportError: cannot import name 'Steerer'`.

- [ ] **Step 3: Implement**

Append to `book_recsys/llm/steer.py`:

```python
def build_steer_prompt(messages, prev: SteeringState, anchor_titles) -> str:
    lines = [
        "You steer a book recommender by choosing its settings. Read the conversation "
        "and return the UPDATED settings as JSON.",
        "",
        "Current settings (carry forward unless the conversation changes them):",
        f"- history_weight: {prev.history_weight}  (1.0 = recommend books like the "
        "reader's past reads; 0.0 = ignore past reads, follow the topic instead)",
        f"- topic: {prev.topic!r}  (the theme/subject to retrieve by; null if none yet)",
        f"- avoid: {prev.avoid}  (themes to steer away from)",
        f"- genre: {prev.genre!r}",
        f"- anchor_book: {prev.anchor_book!r}",
    ]
    if anchor_titles:
        lines.append("")
        lines.append("The reader's past reads: " + ", ".join(anchor_titles[:15]))
    lines.append("")
    lines.append("Conversation so far:")
    for msg in messages:
        lines.append(f"{msg['role']}: {msg['text']}")
    lines += [
        "",
        "Rules:",
        "- Set genre to null UNLESS the reader explicitly names a genre to restrict to.",
        "- If the request is a GIFT or for someone else, set history_weight near 0 (the "
        "past reads are the asker's, not the recipient's) and build topic from the "
        "recipient's described tastes; if a book the recipient loved is named, set "
        "anchor_book to it.",
        "- To clear a setting, return it as null (omitting a key keeps its current value).",
        "",
        'Reply with ONLY a JSON object: {"history_weight": <0..1>, "topic": <string|null>, '
        '"avoid": [<string>...], "genre": <string|null>, "anchor_book": <string|null>, '
        '"reply": "<one short sentence telling the reader what you changed>"}.',
    ]
    return "\n".join(lines)


class Steerer:  # the single LLM call per turn
    def __init__(self, client) -> None:
        self._client = client

    def update(self, messages, prev: SteeringState, anchor_titles) -> SteeringState:
        raw = self._client.complete(build_steer_prompt(messages, prev, anchor_titles))
        return parse_steering(raw, prev)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/llm/test_steer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add book_recsys/llm/steer.py tests/llm/test_steer.py
git commit -m "Add steering prompt builder + Steerer (one LLM call per turn)"
```

---

### Task 4: SteeredRanker

**Files:**
- Create: `book_recsys/llm/rank.py`
- Test: `tests/llm/test_rank.py`

**Interfaces:**
- Consumes: `SteeringState` (Task 2); `weighted_reciprocal_rank_fusion` (Task 1).
- Produces: `SteeredRanker(cf_model, retriever, similar, embeddings, book_ids, encoder, catalog_genre=None, pool=200, lam=1.0)` with
  `.rank(state: SteeringState, history_ids, seen, k: int = 10, anchor_id=None) -> list`.
  - `cf_model.recommend(history_ids, n) -> list`, `retriever.by_history(history_ids, n) -> list`, `retriever.by_text(text, n) -> list`, `similar.recommend(anchor_id, n) -> list`, `encoder.encode(list_of_str) -> array (m, d)`.
  - `embeddings` is an `(N, d)` array aligned to `book_ids`; `catalog_genre` is a `{book_id: genre_str}` dict (or None).
  - Behavior: build weighted lists — `(L_cf, w/2)`, `(L_hist, w/2)`, `(L_topic, 1-w)`, plus `(L_anchor, w/2)` when `anchor_id` is given and `state.topic`/history present per availability — fuse, drop `history_ids ∪ seen`, apply genre include-filter when `state.genre` set, subtract `lam · max cosine(candidate, encode(avoid))` (min-max normalized base, mirroring `FeedService`), return top-`k` book ids.

- [ ] **Step 1: Write the failing tests**

Create `tests/llm/test_rank.py`:

```python
import numpy as np

from book_recsys.llm.rank import SteeredRanker
from book_recsys.llm.steer import SteeringState

BOOK_IDS = ["a", "b", "c", "d", "e"]
# Simple 2-d embeddings so cosine is predictable.
EMB = np.array([[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1]], dtype="float32")


class _CF:
    def recommend(self, history, n):
        return ["a", "b", "c"][:n]


class _Retriever:
    def by_history(self, history, n):
        return ["b", "a"][:n]

    def by_text(self, text, n):
        return ["d", "e", "c"][:n]


class _Similar:
    def recommend(self, anchor_id, n):
        return ["e", "c"][:n]


class _Encoder:
    def __init__(self, vec):
        self._vec = vec

    def encode(self, phrases):
        return np.array([self._vec for _ in phrases], dtype="float32")


def _ranker(**kw):
    return SteeredRanker(_CF(), _Retriever(), _Similar(), EMB, BOOK_IDS,
                         _Encoder([1, 0]), **kw)


def test_rank_excludes_history_and_seen():
    out = _ranker().rank(SteeringState(history_weight=1.0), history_ids=["a"], seen={"b"})
    assert "a" not in out and "b" not in out


def test_rank_topic_only_uses_text_list():
    # history_weight 0 -> only by_text ("d","e","c") drives ranking.
    out = _ranker().rank(SteeringState(history_weight=0.0, topic="x"),
                         history_ids=[], seen=set())
    assert set(out) <= {"c", "d", "e"}
    assert out[0] == "d"


def test_rank_avoid_penalty_demotes_similar_book():
    # Avoid vector [1,0] == book 'a'. Without 'a' (history), test 'c'(=[1,1]) vs 'e'(=[0,-1]).
    # 'c' is more similar to the avoid vector, so it should rank below 'e'.
    state = SteeringState(history_weight=0.0, topic="x", avoid=["spiky"])
    out = _ranker().rank(state, history_ids=[], seen=set())
    assert out.index("e") < out.index("c")


def test_rank_genre_filter_includes_only_matching():
    genre = {"c": "fantasy", "d": "history", "e": "fantasy"}
    out = SteeredRanker(_CF(), _Retriever(), _Similar(), EMB, BOOK_IDS, _Encoder([1, 0]),
                        catalog_genre=genre).rank(
        SteeringState(history_weight=0.0, topic="x", genre="fantasy"),
        history_ids=[], seen=set())
    assert set(out) <= {"c", "e"}
    assert "d" not in out


def test_rank_anchor_adds_similar_results():
    out = _ranker().rank(SteeringState(history_weight=1.0), history_ids=[], seen=set(),
                         anchor_id="z")
    assert "e" in out  # 'e' comes only from similar.recommend


def test_rank_returns_at_most_k():
    out = _ranker().rank(SteeringState(history_weight=0.5, topic="x"),
                         history_ids=[], seen=set(), k=2)
    assert len(out) <= 2


def test_rank_no_signals_returns_empty():
    # No history and no topic -> nothing to fuse.
    out = _ranker().rank(SteeringState(history_weight=1.0), history_ids=[], seen=set())
    assert out == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/llm/test_rank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'book_recsys.llm.rank'`.

- [ ] **Step 3: Implement**

Create `book_recsys/llm/rank.py`:

```python
"""Pure ranking for LLM-steered recsys: fuse weighted signals, penalize 'avoid',
optionally filter by genre. No LLM here — the LLM only supplies the SteeringState.
"""
import numpy as np

from book_recsys.llm.fusion import weighted_reciprocal_rank_fusion


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class SteeredRanker:
    """Rank candidates from a SteeringState using the existing recsys primitives."""

    def __init__(self, cf_model, retriever, similar, embeddings, book_ids, encoder,
                 catalog_genre=None, pool: int = 200, lam: float = 1.0) -> None:
        self._cf = cf_model
        self._retriever = retriever
        self._similar = similar
        self._emb = _l2_normalize(np.asarray(embeddings, dtype="float32"))
        self._row = {b: i for i, b in enumerate(book_ids)}
        self._encoder = encoder
        self._genre = catalog_genre or {}
        self._pool = pool
        self._lam = lam

    def rank(self, state, history_ids, seen, k: int = 10, anchor_id=None) -> list:
        history_ids = list(history_ids)
        w = state.history_weight
        weighted_lists = []
        if history_ids:
            weighted_lists.append((self._cf.recommend(history_ids, self._pool), w / 2))
            weighted_lists.append((self._retriever.by_history(history_ids, self._pool), w / 2))
        if anchor_id is not None:
            weighted_lists.append((self._similar.recommend(anchor_id, self._pool), w / 2))
        if state.topic:
            weighted_lists.append((self._retriever.by_text(state.topic, self._pool), 1 - w))
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
        base = _minmax(np.array([-i for i in range(len(candidates))], dtype="float64"))
        if state.avoid:
            base = base - self._lam * self._avoid_penalty(candidates, state.avoid)
        order = np.argsort(-base, kind="stable")[:k]
        return [candidates[i] for i in order]

    def _avoid_penalty(self, candidates, avoid) -> np.ndarray:
        rows = [self._row[c] for c in candidates if c in self._row]
        if not rows or len(rows) != len(candidates):  # unknown ids -> no penalty
            return np.zeros(len(candidates))
        avoid_vecs = _l2_normalize(np.asarray(self._encoder.encode(list(avoid)),
                                              dtype="float32"))
        sims = self._emb[rows] @ avoid_vecs.T  # (n_cand, n_avoid) cosine
        return sims.max(axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/llm/test_rank.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add book_recsys/llm/rank.py tests/llm/test_rank.py
git commit -m "Add SteeredRanker: weighted fusion + avoid penalty + genre filter"
```

---

### Task 5: Session steering state + transcript

**Files:**
- Modify: `book_recsys/api/sessions.py`
- Test: `tests/api/test_sessions.py` (create if absent)

**Interfaces:**
- Consumes: `SteeringState` (Task 2).
- Produces, on `Session`: `steering: SteeringState` and `messages: list` fields. On `SessionStore`:
  - `ensure(session_id) -> str` — returns `session_id` if known, else creates a fresh empty session and returns its new id.
  - `append_message(session_id, role, text) -> None`.
  - `set_steering(session_id, state) -> None`.

- [ ] **Step 1: Write the failing tests**

Create/append `tests/api/test_sessions.py`:

```python
from book_recsys.api.sessions import SessionStore
from book_recsys.llm.steer import SteeringState


def test_new_session_has_default_steering_and_empty_messages():
    store = SessionStore()
    sid = store.create([])
    s = store.get(sid)
    assert s.steering == SteeringState()
    assert s.messages == []


def test_ensure_returns_same_id_when_known():
    store = SessionStore()
    sid = store.create([])
    assert store.ensure(sid) == sid


def test_ensure_creates_session_when_unknown_or_none():
    store = SessionStore()
    sid = store.ensure(None)
    assert store.get(sid).liked == []
    assert store.ensure("nope") != "nope"  # unknown id -> a fresh session


def test_append_message_and_set_steering():
    store = SessionStore()
    sid = store.create([])
    store.append_message(sid, "user", "hi")
    store.set_steering(sid, SteeringState(topic="WWII"))
    s = store.get(sid)
    assert s.messages == [{"role": "user", "text": "hi"}]
    assert s.steering.topic == "WWII"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/api/test_sessions.py -v`
Expected: FAIL — `AttributeError: 'Session' object has no attribute 'steering'` (and missing methods).

- [ ] **Step 3: Implement**

In `book_recsys/api/sessions.py`, add the import and fields/methods:

```python
from book_recsys.llm.steer import SteeringState
```

Add to the `Session` dataclass (after `seen`):

```python
    steering: SteeringState = field(default_factory=SteeringState)
    messages: list = field(default_factory=list)
```

Add to `SessionStore`:

```python
    def ensure(self, session_id) -> str:
        if session_id is not None and session_id in self._sessions:
            return session_id
        return self.create([])

    def append_message(self, session_id: str, role: str, text: str) -> None:
        self._sessions[session_id].messages.append({"role": role, "text": text})

    def set_steering(self, session_id: str, state: SteeringState) -> None:
        self._sessions[session_id].steering = state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/api/test_sessions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add book_recsys/api/sessions.py tests/api/test_sessions.py
git commit -m "Add per-session steering state + chat transcript"
```

---

### Task 6: POST /steer endpoint + wiring

**Files:**
- Modify: `book_recsys/api/app.py`
- Test: `tests/api/test_app.py` (match the existing file that tests `create_app`; create if absent)

**Interfaces:**
- Consumes: `Steerer.update` (Task 3), `SteeredRanker.rank` (Task 4), `SessionStore.ensure/append_message/set_steering` (Task 5), `RecommenderService.card`/`.search` (existing).
- Produces:
  - `create_app(rec_service, feed_service, session_store, overview=None, steerer=None, ranker=None)` — two new optional params.
  - `POST /steer` with body `{message: str, session_id: str | None = None, k: int = 10}` → `{session_id, reply, state, cards}`. Returns 503 if `steerer`/`ranker` is None. `state` is the `SteeringState` as a dict.
  - `_build_steer(models, catalog, emb, book_ids)` (pragma no cover) and lazy wiring in `get_app`, mirroring `_build_overview`/`_LazyOverview`.

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_app.py` (reuse existing fakes/fixtures there; this shows the full setup):

```python
from fastapi.testclient import TestClient

from book_recsys.api.app import create_app
from book_recsys.api.sessions import SessionStore
from book_recsys.llm.steer import SteeringState


class _RecSvc:
    def card(self, book_id):
        return {"book_id": book_id, "title": f"T{book_id}", "author": "", "description": "",
                "image_url": ""}

    def search(self, q, limit=20):
        return ["anchor1"]

    def label(self, book_id):
        return f"T{book_id}"


class _Steerer:
    def update(self, messages, prev, anchor_titles):
        return SteeringState(history_weight=0.5, topic="WWII", reply="Toward WWII.")


class _Ranker:
    def rank(self, state, history_ids, seen, k=10, anchor_id=None):
        return ["x1", "x2"]


def test_steer_returns_reply_state_and_cards():
    app = create_app(_RecSvc(), None, SessionStore(), steerer=_Steerer(), ranker=_Ranker())
    client = TestClient(app)
    resp = client.post("/steer", json={"message": "books about WWII"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Toward WWII."
    assert body["state"]["topic"] == "WWII"
    assert [c["book_id"] for c in body["cards"]] == ["x1", "x2"]
    assert body["session_id"]  # a session was created


def test_steer_503_when_not_configured():
    app = create_app(_RecSvc(), None, SessionStore())  # no steerer/ranker
    resp = TestClient(app).post("/steer", json={"message": "hi"})
    assert resp.status_code == 503
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_app.py -v -k steer`
Expected: FAIL — `create_app() got an unexpected keyword argument 'steerer'` (or 404 on `/steer`).

- [ ] **Step 3: Implement the endpoint and wiring**

In `book_recsys/api/app.py`:

(a) Add request model near the other `BaseModel`s:

```python
class SteerReq(BaseModel):
    message: str
    session_id: Union[str, None] = None
    k: int = 10
```

(b) Change `create_app` signature:

```python
def create_app(rec_service, feed_service, session_store, overview=None,
               steerer=None, ranker=None) -> FastAPI:
```

(c) Add the route inside `create_app` (after the `/chat` handler), using `dataclasses.asdict`:

```python
    @app.post("/steer")
    def steer(req: SteerReq):
        if steerer is None or ranker is None:
            raise HTTPException(status_code=503,
                                detail="LLM steering unavailable (is Ollama running?)")
        from dataclasses import asdict
        sid = session_store.ensure(req.session_id)
        session = session_store.get(sid)
        session_store.append_message(sid, "user", req.message)
        anchor_titles = [rec_service.card(b)["title"] for b in session.liked][:15]
        try:
            state = steerer.update(session.messages[-6:], session.steering, anchor_titles)
        except Exception:  # noqa: BLE001 — Ollama down / model load -> graceful 503
            log.exception("steer failed -> 503")
            raise HTTPException(status_code=503,
                                detail="LLM steering unavailable (is Ollama running?)")
        session_store.set_steering(sid, state)
        session_store.append_message(sid, "assistant", state.reply)
        anchor_id = None
        if state.anchor_book:
            hits = rec_service.search(state.anchor_book, 1)
            anchor_id = hits[0] if hits else None
        book_ids = ranker.rank(state, session.liked, session.seen, k=req.k, anchor_id=anchor_id)
        return {"session_id": sid, "reply": state.reply, "state": asdict(state),
                "cards": [rec_service.card(b) for b in book_ids]}
```

(d) Add the lazy builder (pragma no cover) near `_build_overview`:

```python
def _build_steer(models, catalog, emb, book_ids):  # pragma: no cover
    import faiss
    from sentence_transformers import SentenceTransformer

    from book_recsys import config
    from book_recsys.llm.clients import LiteLLMClient
    from book_recsys.llm.rank import SteeredRanker
    from book_recsys.llm.retrieve import Retriever
    from book_recsys.llm.steer import Steerer

    faiss.omp_set_num_threads(1)  # avoid the sklearn+torch+faiss OpenMP segfault
    encoder = SentenceTransformer(config.EMBED_MODEL)
    retriever = Retriever(book_ids, emb, encoder=encoder)
    genre = (dict(zip(catalog["book_id"], catalog["genre"]))
             if "genre" in catalog.columns else None)
    ranker = SteeredRanker(models["hybrid_cf_content"], retriever, models["similar"], emb,
                           book_ids, encoder, catalog_genre=genre)
    steerer = Steerer(LiteLLMClient(config.LLM_MODEL, api_base=config.LLM_API_BASE))
    return steerer, ranker
```

(e) In `get_app` (pragma no cover), build lazily and pass to `create_app`. Add a lazy wrapper so the heavy stack loads on first `/steer`, mirroring `_LazyOverview`:

```python
class _LazySteer:  # pragma: no cover
    """Builds the steer stack (encoder + FAISS + ranker) on first use, then reuses it."""

    def __init__(self, build):
        self._build = build
        self._pair = None

    def _ensure(self):
        if self._pair is None:
            self._pair = self._build()
        return self._pair

    def update(self, *args, **kwargs):
        return self._ensure()[0].update(*args, **kwargs)

    def rank(self, *args, **kwargs):
        return self._ensure()[1].rank(*args, **kwargs)
```

Then in `get_app`, after `overview = _LazyOverview(...)`:

```python
    steer = _LazySteer(lambda: _build_steer(models, catalog, emb, book_ids))
    app = create_app(rec_service, feed_service, SessionStore(), overview=overview,
                     steerer=steer, ranker=steer)
```

(Replace the existing `app = create_app(...)` line with the one above; keep the static-mount lines that follow.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/api/test_app.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add book_recsys/api/app.py tests/api/test_app.py
git commit -m "Add POST /steer endpoint + lazy steer-stack wiring"
```

---

### Task 7: UI — chat steers the recsys (live knob panel)

**Files:**
- Modify: `book_recsys/ui/web/app.js`
- Modify: `book_recsys/ui/web/index.html`
- Modify: `book_recsys/ui/web/style.css`

**Interfaces:**
- Consumes: `POST /steer` (Task 6) → `{session_id, reply, state, cards}`.
- Produces: chat panel posts to `/steer`, threads `session_id`, renders reply + cards + a live steering panel.

There is no unit-test harness for the static SPA; this task is verified by running the app (Step 4). Keep all logic thin and server-driven.

- [ ] **Step 1: Add the steering-panel container**

In `index.html`, inside the chat panel (`<div id="chat">`), add a panel above `chat-output`:

```html
      <div id="steer-state" class="steer-state" hidden></div>
```

- [ ] **Step 2: Point the chat at /steer and render reply + state + cards**

In `app.js`, replace the body of `askChat` (the `try { ... } catch` posting to `/chat`) with a `/steer` call that threads `sessionId` and renders the knob panel. Note: `sessionId` is the module-level variable already used by the swipe flow — reuse it.

```javascript
async function askChat() {
  const message = $("chat-input").value.trim();
  if (!message) return;
  const out = $("chat-output");
  out.innerHTML = `<p class="thinking">📚 Steering…</p>`;
  try {
    const res = await fetch("/steer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    if (res.status === 503) {
      out.innerHTML = `<p class="thinking">⚠️ AI steering is offline (start Ollama to enable it).</p>`;
      return;
    }
    const data = await res.json();
    sessionId = data.session_id;          // thread the session across turns
    renderSteerState(data.state);
    renderOverview({ intro: data.reply, categories: [{ header: "Picks", items:
      data.cards.map((c) => ({ ...c, reason: "" })) }] });
  } catch (e) {
    out.innerHTML = `<p class="thinking">⚠️ Something went wrong — try again.</p>`;
  }
}

function renderSteerState(state) {
  const panel = $("steer-state");
  if (!state) { panel.hidden = true; return; }
  const w = Math.round((state.history_weight ?? 1) * 100);
  const chips = [];
  if (state.topic) chips.push(`topic: ${state.topic}`);
  if (state.genre) chips.push(`genre: ${state.genre}`);
  if (state.anchor_book) chips.push(`like: ${state.anchor_book}`);
  (state.avoid || []).forEach((a) => chips.push(`avoid: ${a}`));
  panel.hidden = false;
  panel.innerHTML =
    `<div class="blend">past reads <b>${w}%</b> ▮ <b>${100 - w}%</b> topic</div>` +
    chips.map((c) => `<span class="chip">${c}</span>`).join("");
}
```

If `renderOverview` is not already shaped to accept `{intro, categories:[{header, items:[card+reason]}]}`, reuse the existing overview renderer from the `/chat` path (it already renders exactly that shape). Confirm by reading the current `renderOverview` in `app.js`.

- [ ] **Step 3: Style the panel**

Append to `style.css`:

```css
.steer-state { margin: 0 0 12px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.steer-state .blend { font-size: 13px; color: #444; margin-right: 8px; }
.steer-state .chip { background: #eef; border-radius: 12px; padding: 2px 10px; font-size: 12px; }
```

- [ ] **Step 4: Verify by running the app**

```bash
ollama serve   # in another terminal, if not already running
uvicorn book_recsys.api.app:get_app --factory --port 8001
```

Open http://127.0.0.1:8001/, switch to **Ask-AI**, and run a multi-turn check:
1. "books like the ones I've read" → panel shows `past reads ~100%`.
2. "but about WWII submarines" → topic chip appears, blend shifts toward topic, cards change.
3. "nothing too gory" → an `avoid: gory` chip appears.
4. "actually it's a gift for my dad who loves sailing" → blend snaps toward topic (`past reads ~0–20%`).

Expected: the steering panel updates each turn and the cards re-rank accordingly. (First turn ~60–80s cold; subsequent turns faster.)

- [ ] **Step 5: Commit**

```bash
git add book_recsys/ui/web/app.js book_recsys/ui/web/index.html book_recsys/ui/web/style.css
git commit -m "UI: chat steers the recsys with a live knob panel"
```

---

### Task 8: Full suite + coverage + lint gate

**Files:** none (verification only).

- [ ] **Step 1: Run the whole suite with coverage**

Run: `coverage run -m pytest && coverage report --show-missing --fail-under=100`
Expected: all tests PASS; coverage 100% (network/`_build_*`/`get_app`/`Steerer` real-call paths are `# pragma: no cover`). If any new pure-logic line is uncovered, add a focused test for it.

- [ ] **Step 2: Lint/format/type gate**

Run:
```bash
yapf -ir book_recsys tests --exclude '*ipynb*' --exclude '*venv*'
isort book_recsys tests --line-width 99
mypy book_recsys
```
Expected: no diffs left by yapf/isort after a second run; mypy clean.

- [ ] **Step 3: Commit any formatting fixes**

```bash
git add -A
git commit -m "Format + type-clean LLM-steered recsys"
```

---

## Self-Review

**Spec coverage:**
- Multi-turn running state → Task 5 (`Session.steering`/`messages`) + Task 6 (persisted each turn). ✓
- All four levers (blend, topic, avoid, genre, anchor) → Task 2 (`SteeringState`), Task 4 (`rank` applies each). ✓
- One LLM call/turn (Approach A) → Task 3 (`Steerer.update`), Task 6 (single `update` call). ✓
- Three fusion signals incl. CF → Task 4 (`L_cf`, `L_hist`, `L_topic`, `+L_anchor`). ✓
- Weighted blend `w/2, w/2, 1-w` → Task 4. ✓
- Avoid penalty reusing FeedService math → Task 4 (`_minmax`/`_l2_normalize`, `_avoid_penalty`). ✓
- Opt-in genre filter → Task 2 (null-by-default rule in prompt), Task 4 (filter only when set). ✓
- Gift/UC5 (history_weight→0) → Task 3 prompt rule; Task 7 demo step. ✓
- `/steer` alongside untouched `/chat` → Task 6 (new route; `/chat` unchanged). ✓
- Live knob UI → Task 7. ✓
- Error handling (503, malformed→prior, no-signal→empty, avoid-fail→skip, faiss single-thread) → Task 2 (parse fallback), Task 4 (`no weighted_lists`→[], unknown-id→no penalty), Task 6 (503 + `_build_steer` faiss thread). ✓
- No new training / cached artifacts / bge-small → Task 6 builder loads existing `models`/`emb`, encoder `config.EMBED_MODEL`. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `SteeringState` fields identical across Tasks 2/3/4/5/6. `rank(state, history_ids, seen, k, anchor_id)` signature identical in Task 4 def and Task 6 call. `Steerer.update(messages, prev, anchor_titles)` identical in Tasks 3/6. `weighted_reciprocal_rank_fusion(weighted_lists, k)` identical in Tasks 1/4. ✓
