"""Swipe feed: candidate generation, exclusion, near-duplicate suppression, and penalization.

Holds one or more recommenders (each exposing recommend(history, k) and
score_items(history, candidates) -> scores) so the UI can toggle which one drives the feed.
The book embeddings (one shared normalized copy) power the disliked-similarity penalty and the
near-duplicate filter that drops other editions/omnibuses of a book you've already seen.
"""
import numpy as np

from book_recsys.vecmath import l2_normalize, minmax


class FeedService:
    """Rank the next swipe cards from a chosen recommender, drop near-duplicates of seen books,
    and penalize similarity to disliked books. `recommenders` is a {name: recommender} dict (or a
    single recommender). `dedup` is the cosine above which two books count as the same work
    (different editions of one title embed at ~0.91-0.93; genuinely different books are < ~0.90)."""

    def __init__(self,
                 recommenders,
                 embeddings,
                 book_ids,
                 pool: int = 200,
                 default: str = "",
                 dedup: float = 0.9,
                 avoid: float = 0.86,
                 diversity: float = 0.0,
                 language=None) -> None:
        if not isinstance(recommenders, dict):
            recommenders = {"default": recommenders}
        self._recs = dict(recommenders)
        self._default = default or next(iter(self._recs))
        self._emb = l2_normalize(np.asarray(embeddings, dtype="float32"))
        self._row = {b: i for i, b in enumerate(book_ids)}
        self._pool = pool
        self._dedup = dedup
        self._avoid = avoid
        self._diversity = diversity  # MMR weight: 0 = pure relevance, higher = more varied
        self._language = language or {}  # book_id -> language_code; restricts recs to liked langs

    def methods(self) -> list:
        """Recommender names available for the UI toggle (first is the default)."""
        return list(self._recs)

    def next(self,
             liked,
             disliked,
             seen,
             k: int = 10,
             lam: float = 1.0,
             method=None,
             diversity=None) -> list:
        liked = list(liked)
        if not liked:
            return []
        div = self._diversity if diversity is None else diversity
        rec = self._recs.get(method) or self._recs[self._default]
        candidates = rec.recommend(liked, self._pool)
        exclude = set(seen) | set(liked) | set(disliked)
        candidates = [c for c in candidates if c not in exclude]
        if not candidates:
            return []
        candidates = self._same_language(candidates, liked)  # no random-language recs
        if disliked and lam:
            candidates = self._avoid_disliked(candidates, disliked)  # escape the disliked region
        base = minmax(np.asarray(rec.score_items(liked, candidates), dtype="float64"))
        if disliked and lam:
            base = base - lam * self._max_sim_to_disliked(candidates, disliked)
        return self._select(candidates, base, liked, k, div)

    def _same_language(self, candidates, liked) -> list:
        """Restrict recs to the languages of the liked books: drop candidates in a *known, different*
        language, but keep matching-language and blank/unknown-language books (so an untagged book
        isn't wrongly excluded). No-op without a language map or when no liked book has a known
        language; falls back to the full set if the filter would empty the feed."""
        if not self._language:
            return candidates
        liked_langs = {self._language.get(b) for b in liked} - {"", None}
        if not liked_langs:
            return candidates
        kept = [
            c for c in candidates
            if self._language.get(c, "") in liked_langs or not self._language.get(c)
        ]
        return kept or candidates

    def _avoid_disliked(self, candidates, disliked) -> list:
        """Hard-drop candidates within `avoid` cosine of any disliked book, so a dislike steers
        the feed *out* of that neighbourhood instead of merely re-ranking inside it (the soft
        penalty alone couldn't escape a tight cluster). Falls back to the full set if the filter
        would empty the feed (e.g. every candidate sits near a disliked book)."""
        d_rows = [self._row[d] for d in disliked if d in self._row]
        if not d_rows:
            return candidates
        c_rows = [self._row[c] for c in candidates]
        near = (self._emb[c_rows] @ self._emb[d_rows].T).max(axis=1)
        kept = [c for c, s in zip(candidates, near) if s < self._avoid]
        return kept or candidates

    def _select(self, candidates, base, liked, k: int, diversity: float) -> list:
        """Greedy MMR selection: each slot maximizes relevance − `diversity`·(max cosine to the
        already-selected books), so the feed spreads across clusters instead of filling with one
        (a single tight cluster — e.g. one non-English book — otherwise sweeps the list). Also skips
        near-duplicates (cosine >= `dedup`) of a liked or already-selected book. `diversity=0`
        reduces to pure relevance order with the dedup filter."""
        crows = np.array([self._row[c] for c in candidates])
        rel = np.asarray(base, dtype="float64")
        ref_rows = [self._row[b] for b in liked if b in self._row]
        sel_sim = np.zeros(
            len(candidates))  # running max cosine of each candidate to the chosen set
        avail = np.ones(len(candidates), dtype=bool)
        out: list = []
        while len(out) < k and avail.any():
            score = np.where(avail, rel - diversity * sel_sim, -np.inf)
            j = int(np.argmax(score))
            avail[j] = False
            crow = crows[j]
            if ref_rows and float((self._emb[ref_rows] @ self._emb[crow]).max()) >= self._dedup:
                continue
            out.append(candidates[j])
            ref_rows.append(crow)
            sel_sim = np.maximum(sel_sim, self._emb[crows] @ self._emb[crow])
        return out

    def _max_sim_to_disliked(self, candidates, disliked) -> np.ndarray:
        """For each candidate, cosine similarity to its NEAREST disliked book (0 if none)."""
        d_rows = [self._row[d] for d in disliked if d in self._row]
        if not d_rows:
            return np.zeros(len(candidates))
        c_rows = [self._row[c] for c in candidates]
        sims = self._emb[c_rows] @ self._emb[d_rows].T  # normalized -> cosine
        return sims.max(axis=1)
