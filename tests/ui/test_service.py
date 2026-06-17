import pandas as pd

from book_recsys.ui.service import RecommenderService

CATALOG = pd.DataFrame({
    "book_id": ["b0", "b1", "b2", "b3"],
    "title": ["The Hobbit", "The Fellowship of the Ring", "Dune", "Hobbit Tales"],
    "author": ["J.R.R. Tolkien", "J.R.R. Tolkien", "Frank Herbert", ""],
    "description": ["A hobbit's quest", "Frodo leaves the Shire", "Desert planet Arrakis", ""],
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


def test_similar_to_returns_labels():
    out = _svc().similar_to("b0", k=2)
    assert out == ["Hobbit Tales",
                   "The Fellowship of the Ring by J.R.R. Tolkien — Frodo leaves the Shire…"]


def test_similar_to_no_neighbors_returns_empty():
    assert _svc().similar_to("b1", k=2) == []


def test_card_returns_full_untruncated_fields():
    assert _svc().card("b2") == {"book_id": "b2", "title": "Dune",
                                 "author": "Frank Herbert",
                                 "description": "Desert planet Arrakis"}


def test_card_missing_author_and_description_are_empty_strings():
    assert _svc().card("b3") == {"book_id": "b3", "title": "Hobbit Tales",
                                 "author": "", "description": ""}


def test_works_without_author_or_description_columns():
    cat = pd.DataFrame({"book_id": ["x"], "title": ["X"]})
    svc = RecommenderService(cat, {"svd": _HistRec()}, _SimRec())
    assert svc.label("x") == "X"
