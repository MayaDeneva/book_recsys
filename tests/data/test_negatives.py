import numpy as np
import pytest

from book_recsys.data.negatives import build_cdf, sample_negatives


def test_build_cdf_normalizes_and_is_monotonic():
    cdf = build_cdf([1, 1, 2])
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cdf) >= 0)


def test_sample_negatives_uniform_excludes_seen():
    pool = np.array(["a", "b", "c"])
    out = sample_negatives(pool, {"a"}, 2, np.random.default_rng(0))
    assert set(out) == {"b", "c"}            # only non-seen, all distinct


def test_sample_negatives_weighted_picks_high_weight_item():
    pool = np.array(["a", "b", "c", "d"])
    cdf = build_cdf([1, 0, 0, 0])            # all mass on 'a'
    out = sample_negatives(pool, set(), 1, np.random.default_rng(0), cdf=cdf)
    assert out == ["a"]


def test_sample_negatives_weighted_rejects_repeats_until_filled():
    pool = np.array(["a", "b", "c", "d"])
    cdf = build_cdf([10, 10, 1, 1])          # heavy on a,b -> repeats rejected to reach c,d
    out = sample_negatives(pool, set(), 4, np.random.default_rng(0), cdf=cdf)
    assert set(out) == {"a", "b", "c", "d"}
