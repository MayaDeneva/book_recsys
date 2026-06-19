import numpy as np
import pandas as pd

from book_recsys.data.schema import BOOK, RATING, TS, USER
from book_recsys.models.autoencoder.recommender import MultVaeRecommender


def _df(rows):
    return pd.DataFrame([{USER: u, BOOK: b, RATING: 0, TS: t} for u, b, t in rows])


# Two clusters: users a* read {b0,b1,b2}; users c* read {b3,b4,b5}. b5 is the most popular.
def _clustered():
    rows = []
    for i in range(10):
        for b in ("b0", "b1", "b2"):
            rows.append((f"a{i}", b, 0))
    for i in range(10):
        for b in ("b3", "b4", "b5"):
            rows.append((f"c{i}", b, 0))
    for i in range(8):  # make b5 the popularity head
        rows.append((f"a{i}", "b5", 0))
    return _df(rows)


def _fit(**kw):
    return MultVaeRecommender(hidden=32,
                              latent=8,
                              dropout=0.0,
                              epochs=80,
                              batch_size=8,
                              anneal_steps=20,
                              device="cpu",
                              seed=0,
                              **kw).fit(_clustered())


def test_recommend_excludes_seen():
    rec = _fit()
    out = rec.recommend(["b0", "b1"], k=5)
    assert "b0" not in out and "b1" not in out


def test_recommend_prefers_co_cluster():
    rec = _fit()
    out = rec.recommend(["b0", "b1"], k=2)
    assert "b2" in out  # same cluster as the history


def test_recommend_empty_when_unfitted():
    assert MultVaeRecommender().recommend(["b0"], k=5) == []


def test_recommend_unknown_history_ids_dropped():
    rec = _fit()
    out = rec.recommend(["does-not-exist"], k=3)
    assert isinstance(out, list) and len(out) <= 3


def test_score_items_known_and_unknown():
    rec = _fit()
    scores = rec.score_items(["b0", "b1"], ["b2", "zzz"])
    assert np.isfinite(scores[0]) and scores[1] == float("-inf")


def test_score_items_empty_when_unfitted():
    assert MultVaeRecommender().score_items(["b0"], ["b1", "b2"]) == [float("-inf")] * 2


def test_pop_discount_demotes_popular_head():
    rec = _fit()
    base = rec.score_items(["b3", "b4"], ["b5"])[0]  # b5 = popular head
    rec.pop_discount = 5.0
    discounted = rec.score_items(["b3", "b4"], ["b5"])[0]
    assert discounted < base  # popularity penalty applied


def test_attach_roundtrips_scores(tmp_path):
    from book_recsys.models.autoencoder.data import build_matrix
    from book_recsys.models.autoencoder.train import load_checkpoint
    rec = MultVaeRecommender(hidden=16,
                             latent=4,
                             dropout=0.0,
                             epochs=3,
                             batch_size=8,
                             device="cpu",
                             seed=0,
                             ckpt_dir=str(tmp_path)).fit(_clustered())
    before = rec.score_items(["b0"], ["b1", "b2"])
    _, ids, pos, counts = build_matrix(_clustered(), 1)
    model, _ = load_checkpoint(str(tmp_path / "multvae_last.pt"), device="cpu")
    rebuilt = MultVaeRecommender(device="cpu").attach(model, ids, pos, counts)
    after = rebuilt.score_items(["b0"], ["b1", "b2"])
    assert np.allclose(before, after)
