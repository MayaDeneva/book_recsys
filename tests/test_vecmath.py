import numpy as np

from book_recsys.vecmath import l2_normalize, minmax


def test_l2_normalize_unit_rows_and_zero_row_safe():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype="float32"))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])  # zero row stays zero, no divide error


def test_minmax_scales_to_unit_and_handles_constant():
    assert np.allclose(minmax(np.array([0.0, 5.0, 10.0])), [0.0, 0.5, 1.0])
    assert np.allclose(minmax(np.array([7.0, 7.0, 7.0])), [0.0, 0.0, 0.0])
