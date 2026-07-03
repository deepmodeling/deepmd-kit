import numpy as np
import pytest

from deepmd.dpmodel.utils.neighbor_graph import pad_and_guard_angles


def test_pad_angles_dynamic_appends_min_guard():
    ai = np.array([[0, 1], [1, 0]], dtype=np.int64)  # 2 real angles
    out_ai, out_mask = pad_and_guard_angles(ai, angle_capacity=None, min_angles=2)
    assert out_ai.shape == (2, 4)  # 2 real + 2 guard
    np.testing.assert_array_equal(out_mask, [True, True, False, False])


def test_pad_angles_static_capacity():
    ai = np.array([[0, 1], [1, 0]], dtype=np.int64)
    out_ai, out_mask = pad_and_guard_angles(ai, angle_capacity=5)
    assert out_ai.shape == (2, 5)
    assert int(out_mask.sum()) == 2


def test_pad_angles_overflow_raises():
    ai = np.zeros((2, 6), dtype=np.int64)
    with pytest.raises(ValueError):
        pad_and_guard_angles(ai, angle_capacity=4)
