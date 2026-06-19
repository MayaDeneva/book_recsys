import pandas as pd

from book_recsys.features.document import build_book_document, build_documents


def test_builds_document_with_all_parts():
    doc = build_book_document("Dragon Dawn", "A young mage rises.", ["fantasy", "magic"])
    assert doc == ("Title: Dragon Dawn\n"
                   "Plot: A young mage rises.\n"
                   "Themes/shelves: fantasy, magic")


def test_omits_themes_line_when_no_shelves():
    doc = build_book_document("No Shelves", "", [])
    assert doc == "Title: No Shelves\nPlot: "


def test_drops_behavioural_shelf_noise():
    # EDA: ~38% of shelf tags are status/format noise (to-read on 99% of books) -> strip them.
    doc = build_book_document("Dragon Dawn", "A mage.",
                              ["to-read", "fantasy", "currently-reading", "magic", "favorites"])
    assert doc == "Title: Dragon Dawn\nPlot: A mage.\nThemes/shelves: fantasy, magic"


def test_omits_themes_line_when_all_shelves_are_noise():
    doc = build_book_document("X", "", ["to-read", "currently-reading", "owned", "kindle"])
    assert doc == "Title: X\nPlot: "


def test_build_documents_returns_one_string_per_row():
    df = pd.DataFrame({
        "title": ["A", "B"],
        "description": ["x", "y"],
        "shelves": [["s1"], []],
    })
    docs = build_documents(df)
    assert docs == ["Title: A\nPlot: x\nThemes/shelves: s1", "Title: B\nPlot: y"]


def test_title_only_field():
    assert build_book_document("Dragon", "plot here", ["fantasy"], fields=("title",)) == "Title: Dragon"


def test_title_and_plot_fields():
    doc = build_book_document("Dragon", "plot here", ["fantasy"], fields=("title", "plot"))
    assert doc == "Title: Dragon\nPlot: plot here"


def test_shelves_field_still_skipped_when_empty():
    doc = build_book_document("Dragon", "p", [], fields=("title", "plot", "shelves"))
    assert doc == "Title: Dragon\nPlot: p"


def test_build_documents_passes_fields_through():
    import pandas as pd
    df = pd.DataFrame({"title": ["A"], "description": ["x"], "shelves": [["s1"]]})
    assert build_documents(df, fields=("title",)) == ["Title: A"]


def test_handles_numpy_array_shelves():
    import numpy as np
    doc = build_book_document("T", "d", np.array(["a", "b"]))
    assert doc == "Title: T\nPlot: d\nThemes/shelves: a, b"


def test_handles_nan_shelves():
    doc = build_book_document("T", "d", float("nan"))
    assert doc == "Title: T\nPlot: d"


def test_author_appended_to_title():
    doc = build_book_document("Dune", "desc", ["sci-fi"], author="Frank Herbert")
    assert doc.splitlines()[0] == "Title: Dune by Frank Herbert"


def test_no_author_leaves_title_unchanged():
    doc = build_book_document("Dune", "desc", ["sci-fi"])
    assert doc.splitlines()[0] == "Title: Dune"


def test_build_documents_uses_author_column():
    import pandas as pd
    df = pd.DataFrame({"title": ["Dune"], "description": ["d"], "shelves": [["sci-fi"]],
                       "author": ["Frank Herbert"]})
    assert build_documents(df)[0].splitlines()[0] == "Title: Dune by Frank Herbert"


def test_genre_field_included_when_present():
    doc = build_book_document("Dune", "desc", ["sci-fi"], author="Frank Herbert",
                              genre="science fiction, classics",
                              fields=("title", "genre", "plot", "shelves"))
    assert doc == ("Title: Dune by Frank Herbert\n"
                   "Genre: science fiction, classics\n"
                   "Plot: desc\n"
                   "Themes/shelves: sci-fi")


def test_genre_omitted_when_not_in_fields():
    doc = build_book_document("Dune", "desc", ["sci-fi"], genre="science fiction")
    assert "Genre:" not in doc   # default fields don't include genre


def test_genre_omitted_when_empty():
    doc = build_book_document("Dune", "desc", ["sci-fi"], genre="",
                              fields=("title", "genre", "plot"))
    assert doc == "Title: Dune\nPlot: desc"


def test_build_documents_uses_genre_column():
    df = pd.DataFrame({"title": ["Dune"], "description": ["d"], "shelves": [["sci-fi"]],
                       "genre": ["science fiction"]})
    assert "Genre: science fiction" in build_documents(df, fields=("title", "genre", "plot"))[0]
