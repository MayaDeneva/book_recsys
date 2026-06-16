import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.recbole_adapter.atomic import write_inter_file


def _df():
    return pd.DataFrame({USER: ["u0", "u1"], BOOK: ["b0", "b1"],
                         RATING: [5, 3], TS: [100, 200]})


def test_writes_typed_header(tmp_path):
    p = tmp_path / "d.inter"
    write_inter_file(_df(), p)
    header = p.read_text().splitlines()[0]
    assert header == "user_id:token\titem_id:token\trating:float\ttimestamp:float"


def test_writes_tab_separated_rows(tmp_path):
    p = tmp_path / "d.inter"
    write_inter_file(_df(), p)
    rows = p.read_text().splitlines()[1:]
    assert rows[0].split("\t")[:2] == ["u0", "b0"]
    assert len(rows) == 2
