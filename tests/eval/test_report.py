import pandas as pd

from book_recsys.eval.report import results_to_markdown, splice_section


def test_results_to_markdown_table():
    df = pd.DataFrame({"ndcg@10": [0.5, 0.1]}, index=["a", "b"])
    md = results_to_markdown(df)
    assert md.splitlines()[0] == "| method | ndcg@10 |"
    assert "| a | 0.5000 |" in md
    assert "| b | 0.1000 |" in md


def test_splice_section_replaces_between_markers():
    text = "pre <!--S-->old<!--E--> post"
    out = splice_section(text, "<!--S-->", "<!--E-->", "new")
    assert out == "pre <!--S-->\nnew\n<!--E--> post"
    assert "old" not in out
