import pandas as pd

from book_recsys.ui.service import RecommenderService

CATALOG = pd.DataFrame({
    "book_id": ["b0", "b1", "b2", "b3"],
    "title": ["The Hobbit", "The Fellowship of the Ring", "Dune", "Hobbit Tales"],
    "author": ["J.R.R. Tolkien", "J.R.R. Tolkien", "Frank Herbert", ""],
    "description": ["A hobbit's quest", "Frodo leaves the Shire", "Desert planet Arrakis", ""],
    "image_url": ["", "", "http://img/dune.jpg",
                  "https://s.gr-assets.com/assets/nophoto/book/x.png"],
})


class _HistRec:
    def recommend(self, history, k):
        return ["b2", "b1"][:k]   # ignores history; fixed output


class _SimRec:
    def recommend(self, anchor, k):
        return {"b0": ["b3", "b1"], "b2": ["b1"]}.get(anchor, [])[:k]


def _svc():
    return RecommenderService(CATALOG, {"svd": _HistRec()}, _SimRec())


def test_methods_lists_history_recommenders():
    assert _svc().methods() == ["svd"]


def test_search_returns_book_ids_by_title_substring():
    assert set(_svc().search("hobbit", limit=5)) == {"b0", "b3"}


def test_search_matches_author_too():
    assert set(_svc().search("tolkien", limit=5)) == {"b0", "b1"}   # by author
    assert set(_svc().search("herbert", limit=5)) == {"b2"}


def test_search_respects_limit():
    assert len(_svc().search("the", limit=1)) == 1


def test_label_includes_author_and_description():
    assert _svc().label("b2") == "Dune by Frank Herbert — Desert planet Arrakis…"


def test_label_title_only_when_no_author_or_description():
    assert _svc().label("b3") == "Hobbit Tales"


def test_recommend_by_history_returns_labels():
    out = _svc().recommend_by_history(["b0"], method="svd", k=2)
    assert out == ["Dune by Frank Herbert — Desert planet Arrakis…",
                   "The Fellowship of the Ring by J.R.R. Tolkien — Frodo leaves the Shire…"]


class _WeightAwareRec:
    def __init__(self):
        self.last_weights = "unset"

    def recommend(self, history, k, weights=None):
        self.last_weights = weights
        return ["b1"][:k]


def test_recommend_by_history_recency_weights_recent_picks_higher():
    rec = _WeightAwareRec()
    svc = RecommenderService(CATALOG, {"m": rec}, _SimRec())
    svc.recommend_by_history(["b0", "b1", "b2"], method="m", k=1, recency=True)
    w = rec.last_weights
    assert w[-1] == 1.0          # most recent pick: no decay
    assert w[0] < w[1] < w[2]    # earlier picks decay more


def test_recommend_by_history_no_recency_omits_weights():
    rec = _WeightAwareRec()
    svc = RecommenderService(CATALOG, {"m": rec}, _SimRec())
    svc.recommend_by_history(["b0", "b1"], method="m", k=1)
    assert rec.last_weights is None   # default path passes no weights


def test_similar_to_returns_labels():
    out = _svc().similar_to("b0", k=2)
    assert out == ["Hobbit Tales",
                   "The Fellowship of the Ring by J.R.R. Tolkien — Frodo leaves the Shire…"]


def test_similar_to_no_neighbors_returns_empty():
    assert _svc().similar_to("b1", k=2) == []


def test_card_returns_full_untruncated_fields_with_cover():
    assert _svc().card("b2") == {"book_id": "b2", "title": "Dune",
                                 "author": "Frank Herbert",
                                 "description": "Desert planet Arrakis",
                                 "image_url": "http://img/dune.jpg"}


def test_card_missing_fields_empty_and_nophoto_cover_dropped():
    # b3 has no author/description and a Goodreads `nophoto` placeholder -> all blank
    assert _svc().card("b3") == {"book_id": "b3", "title": "Hobbit Tales",
                                 "author": "", "description": "", "image_url": ""}


def test_works_without_author_or_description_columns():
    cat = pd.DataFrame({"book_id": ["x"], "title": ["X"]})
    svc = RecommenderService(cat, {"svd": _HistRec()}, _SimRec())
    assert svc.label("x") == "X"
