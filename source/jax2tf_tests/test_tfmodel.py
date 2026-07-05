# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.jax.jax2tf.tfmodel import (
    TFModelWrapper,
)


def _make_wrapper(
    default_chg_spin: list[float] | None = None,
) -> TFModelWrapper:
    wrapper = object.__new__(TFModelWrapper)
    wrapper.dim_chg_spin = 2
    wrapper._has_default_chg_spin = default_chg_spin is not None
    wrapper.default_chg_spin = default_chg_spin
    return wrapper


def test_make_charge_spin_input_uses_default() -> None:
    wrapper = _make_wrapper([0.0, 1.0])

    charge_spin = wrapper._make_charge_spin_input(3, None)

    np.testing.assert_allclose(
        np.asarray(charge_spin),
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    )


def test_make_charge_spin_input_broadcasts_explicit_single_frame() -> None:
    wrapper = _make_wrapper([0.0, 1.0])

    charge_spin = wrapper._make_charge_spin_input(3, np.array([2.0, 1.0]))

    np.testing.assert_allclose(
        np.asarray(charge_spin),
        np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]),
    )


def test_make_charge_spin_input_requires_default_or_explicit_value() -> None:
    wrapper = _make_wrapper()

    with pytest.raises(ValueError, match="charge_spin is required"):
        wrapper._make_charge_spin_input(1, None)
