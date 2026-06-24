import numpy as np

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
    rec = FakeRec(rec_order=["b", "c", "d", "e"], scores={"b": 0.2, "c": 0.9, "d": 0.5, "e": 0.1})
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


def test_next_returns_empty_when_all_candidates_excluded():
    book_ids = ["a", "b", "c"]
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.5, "c": 0.5})
    fs = FeedService(rec, np.eye(3, dtype="float32"), book_ids, pool=10)
    # both candidates already seen -> nothing left to recommend
    assert fs.next(liked=["a"], disliked=[], seen=["b", "c"], k=10, lam=0.0) == []


def test_next_equal_scores_keep_recommend_order():
    book_ids = ["a", "b", "c"]
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.5, "c": 0.5})
    fs = FeedService(rec, np.eye(3, dtype="float32"), book_ids, pool=10)
    # equal scores -> _minmax returns zeros -> stable sort preserves recommend order
    assert fs.next(liked=["a"], disliked=[], seen=[], k=10, lam=0.0) == ["b", "c"]


def test_dislike_drops_candidates_in_the_disliked_neighbourhood():
    # b points the same direction as disliked x (cos 1.0 >= avoid); c is orthogonal.
    book_ids = ["a", "b", "c", "x"]
    emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype="float32")
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.5, "c": 0.5})
    fs = FeedService(rec, emb, book_ids, pool=10)
    # lam=0 -> dislike ignored, both kept in recommend order
    assert fs.next(liked=["a"], disliked=["x"], seen=[], k=10, lam=0.0) == ["b", "c"]
    # lam=1 -> b is in x's neighbourhood -> dropped entirely (escape, not just demote)
    assert fs.next(liked=["a"], disliked=["x"], seen=[], k=10, lam=1.0) == ["c"]


def test_dislike_avoid_falls_back_when_all_candidates_are_near_disliked():
    # both b and c sit in disliked x's neighbourhood; filtering would empty the feed, so the
    # avoid filter falls back to the full set rather than returning nothing.
    book_ids = ["a", "b", "c", "x"]
    emb = np.array([[1, 0, 0], [0, 0.9, 0.4359], [0, 0.9, -0.4359], [0, 1, 0]], dtype="float32")
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.9, "c": 0.1})
    fs = FeedService(rec, emb, book_ids, pool=10)
    assert fs.next(liked=["a"], disliked=["x"], seen=[], k=10, lam=1.0) == ["b", "c"]


def test_no_disliked_means_no_penalty():
    book_ids = ["a", "b", "c"]
    emb = np.eye(3, dtype="float32")
    rec = FakeRec(["b", "c"], {"b": 0.9, "c": 0.1})
    fs = FeedService(rec, emb, book_ids, pool=10)
    assert fs.next(liked=["a"], disliked=[], seen=[], k=10, lam=1.0) == ["b", "c"]


def test_unknown_disliked_id_is_ignored():
    book_ids = ["a", "b", "c"]
    rec = FakeRec(rec_order=["b", "c"], scores={"b": 0.9, "c": 0.1})
    fs = FeedService(rec, np.eye(3, dtype="float32"), book_ids, pool=10)
    # disliked id not in the catalog -> no embedding -> no penalty, ranking unchanged
    assert fs.next(liked=["a"], disliked=["zzz"], seen=[], k=10, lam=1.0) == ["b", "c"]


def _two_method_fs():
    book_ids = ["a", "b", "c"]
    recs = {
        "m1": FakeRec(["b", "c"], {
            "b": 0.9,
            "c": 0.1
        }),  # m1 prefers b
        "m2": FakeRec(["b", "c"], {
            "b": 0.1,
            "c": 0.9
        }),  # m2 prefers c
    }
    return FeedService(recs, np.eye(3, dtype="float32"), book_ids, pool=10)


def test_methods_lists_recommender_names_default_first():
    assert _two_method_fs().methods() == ["m1", "m2"]


def test_next_uses_selected_method():
    fs = _two_method_fs()
    assert fs.next(["a"], [], [], k=10, lam=0.0, method="m2")[0] == "c"  # m2 prefers c
    assert fs.next(["a"], [], [], k=10, lam=0.0, method="m1")[0] == "b"  # m1 prefers b


def test_next_unknown_or_no_method_falls_back_to_default():
    fs = _two_method_fs()
    assert fs.next(["a"], [], [], k=10, lam=0.0)[0] == "b"  # default = m1
    assert fs.next(["a"], [], [], k=10, lam=0.0, method="nope")[0] == "b"


def test_drops_near_duplicate_edition_of_a_liked_book():
    # 'dup' has the same embedding as the liked 'a' (a different edition of the same work);
    # 'diff' is orthogonal. Exclusion by id can't catch dup (different id) — the dedup must.
    book_ids = ["a", "dup", "diff"]
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype="float32")
    rec = FakeRec(["dup", "diff"], {"dup": 0.9, "diff": 0.1})
    fs = FeedService(rec, emb, book_ids, pool=10)
    assert fs.next(["a"], [], [], k=10, lam=0.0) == ["diff"]  # dup (cos 1.0 to liked) removed


def test_dedups_near_identical_recommendations():
    # x and y are the same work (identical embedding); only the higher-ranked one is shown.
    book_ids = ["a", "x", "y"]
    emb = np.array([[1, 0], [0, 1], [0, 1]], dtype="float32")
    rec = FakeRec(["x", "y"], {"x": 0.9, "y": 0.8})
    fs = FeedService(rec, emb, book_ids, pool=10)
    assert fs.next(["a"], [], [], k=10, lam=0.0) == ["x"]  # y is a near-dup of x -> dropped


def test_keeps_similar_but_not_duplicate_items():
    book_ids = ["a", "b"]
    emb = np.array([[1, 0], [0.8, 0.6]], dtype="float32")  # cos(a,b)=0.8 < dedup 0.9 -> keep
    rec = FakeRec(["b"], {"b": 0.5})
    fs = FeedService(rec, emb, book_ids, pool=10)
    assert fs.next(["a"], [], [], k=10, lam=0.0) == ["b"]


def test_diversity_spreads_across_clusters():
    # b,c are similar (cos 0.85 — below the 0.9 dedup, so both survive); d is orthogonal to b.
    book_ids = ["a", "b", "c", "d"]
    emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0.85, 0.5268], [0, 0, 1]], dtype="float32")
    rec = FakeRec(["b", "c", "d"], {"b": 1.0, "c": 0.7, "d": 0.6})
    # no diversity -> top-2 by relevance: b then c (the cluster)
    fs0 = FeedService(rec, emb, book_ids, pool=10, diversity=0.0)
    assert fs0.next(["a"], [], [], k=2, lam=0.0) == ["b", "c"]
    # diversity on -> c is penalized for hugging b; the orthogonal d wins the 2nd slot
    fs1 = FeedService(rec, emb, book_ids, pool=10, diversity=0.9)
    assert fs1.next(["a"], [], [], k=2, lam=0.0) == ["b", "d"]


def test_language_filter_keeps_liked_languages_and_blanks():
    book_ids = ["a", "b", "c", "d"]
    lang = {"a": "eng", "b": "eng", "c": "ger", "d": ""}  # liked a=eng
    rec = FakeRec(["b", "c", "d"], {"b": 0.9, "c": 0.8, "d": 0.7})
    fs = FeedService(rec, np.eye(4, dtype="float32"), book_ids, pool=10, language=lang)
    # c (german, known-different) dropped; b (eng match) + d (blank/unknown) kept
    assert fs.next(["a"], [], [], k=10, lam=0.0) == ["b", "d"]


def test_language_filter_groups_english_variants():
    book_ids = ["a", "b", "c"]
    lang = {"a": "eng", "b": "en-US", "c": "jpn"}  # liked eng; en-US is still English -> kept
    rec = FakeRec(["b", "c"], {"b": 0.9, "c": 0.1})
    fs = FeedService(rec, np.eye(3, dtype="float32"), book_ids, pool=10, language=lang)
    assert fs.next(["a"], [], [], k=10, lam=0.0) == ["b"]  # jpn dropped, en-US kept


def test_language_filter_falls_back_when_it_would_empty_feed():
    book_ids = ["a", "b"]
    lang = {"a": "eng", "b": "ger"}  # only a german candidate -> filter empties -> fall back
    rec = FakeRec(["b"], {"b": 0.5})
    fs = FeedService(rec, np.eye(2, dtype="float32"), book_ids, pool=10, language=lang)
    assert fs.next(["a"], [], [], k=10, lam=0.0) == ["b"]


def test_language_filter_skipped_when_liked_language_unknown():
    book_ids = ["a", "b", "c"]
    lang = {"a": "", "b": "ger", "c": "fre"}  # liked has no known language -> no filtering
    rec = FakeRec(["b", "c"], {"b": 0.9, "c": 0.1})
    fs = FeedService(rec, np.eye(3, dtype="float32"), book_ids, pool=10, language=lang)
    assert fs.next(["a"], [], [], k=10, lam=0.0) == ["b", "c"]


def test_dedup_skipped_when_liked_not_in_catalog():
    book_ids = ["x", "y"]
    rec = FakeRec(["x", "y"], {"x": 0.9, "y": 0.1})
    fs = FeedService(rec, np.eye(2, dtype="float32"), book_ids, pool=10)
    # liked id absent from the catalog -> no reference rows -> dedup can't run, both kept
    assert fs.next(["notincatalog"], [], [], k=10, lam=0.0) == ["x", "y"]


class WeightAwareRec(FakeRec):
    weight_aware = True

    def __init__(self, order, scores):
        super().__init__(order, scores)
        self.seen_recommend = self.seen_score = "unset"

    def recommend(self, history, k, weights=None):
        self.seen_recommend = weights
        return self._order[:k]

    def score_items(self, history, candidates, weights=None):
        self.seen_score = weights
        return [self._scores[c] for c in candidates]


def test_feed_passes_event_weights_to_weight_aware_recommender():
    rec = WeightAwareRec(["b", "c"], {"b": 0.9, "c": 0.1})
    fs = FeedService(rec, np.eye(3, dtype="float32"), ["a", "b", "c"], pool=10)
    fs.next(["a"], [], [], k=10, lam=0.0, weights={"a": 0.4})
    assert rec.seen_recommend == [0.4] and rec.seen_score == [0.4]  # aligned to liked


def test_feed_does_not_pass_weights_to_plain_recommender():
    rec = FakeRec(["b", "c"], {"b": 0.9, "c": 0.1})  # no weight_aware flag -> called plainly
    fs = FeedService(rec, np.eye(3, dtype="float32"), ["a", "b", "c"], pool=10)
    assert fs.next(["a"], [], [], k=10, lam=0.0, weights={"a": 0.4}) == ["b", "c"]


def test_sasrec_recommender_drives_feed():
    import torch

    from book_recsys.models.sequential.model import SASRec
    from book_recsys.models.sequential.recommender import SasRecRecommender

    torch.manual_seed(0)
    tokens = ["[PAD]", "a", "b", "c", "d", "e"]  # index -> book_id, 0 = PAD
    rec = SasRecRecommender(SASRec(n_items=6,
                                   hidden_size=8,
                                   n_layers=2,
                                   n_heads=2,
                                   inner_size=16,
                                   max_seq_length=4).eval(),
                            tokens,
                            device="cpu")
    book_ids = ["a", "b", "c", "d", "e"]
    fs = FeedService(rec, np.eye(5, dtype="float32"), book_ids, pool=10)
    out = fs.next(liked=["a"], disliked=[], seen=[], k=3, lam=0.0)
    assert len(out) == 3
    assert "a" not in out  # the liked (seen) item is never re-served
    assert set(out) <= {"b", "c", "d", "e"}
