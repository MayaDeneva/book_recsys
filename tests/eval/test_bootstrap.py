from book_recsys.eval.bootstrap import bootstrap_ci, paired_bootstrap


def test_bootstrap_ci_constant_scores_has_zero_width():
    mean, lo, hi = bootstrap_ci([0.4] * 50)
    assert lo == hi == mean  # all resamples identical -> zero-width CI
    assert abs(mean - 0.4) < 1e-9


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([0.0, 1.0] * 100, n_resamples=500, seed=0)
    assert abs(mean - 0.5) < 1e-9
    assert lo < mean < hi


def test_paired_bootstrap_significant_when_a_beats_b():
    out = paired_bootstrap([1.0] * 100, [0.0] * 100)
    assert out["mean_diff"] == 1.0
    assert out["lo"] > 0 and out["significant"] is True


def test_paired_bootstrap_not_significant_when_equal():
    a = [0.5, 0.4, 0.6, 0.3] * 25
    out = paired_bootstrap(a, a)
    assert out["mean_diff"] == 0.0
    assert out["significant"] is False
