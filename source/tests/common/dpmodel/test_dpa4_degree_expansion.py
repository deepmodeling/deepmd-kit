# SPDX-License-Identifier: LGPL-3.0-or-later
"""Degree-layout expansion preserves the index-defined feature map."""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.activation import (
    GatedActivation,
)
from deepmd.dpmodel.descriptor.dpa4_nn.indexing import (
    build_m_major_l_index,
    expand_degree_axis,
    map_degree_idx,
)


@pytest.mark.parametrize(
    "lmax,mmax", [(0, None), (0, 0), (1, None), (3, None), (3, 0), (3, 1), (3, 3)]
)
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("scalar", [False, True])
def test_degree_expansion_matches_index_map(
    lmax: int, mmax: int | None, axis: int, scalar: bool
) -> None:
    """All full and reduced layouts broadcast each degree to its own orders."""
    indices = (
        map_degree_idx(lmax) if mmax is None else build_m_major_l_index(lmax, mmax)
    )
    first = 0 if scalar else 1
    indices = indices[first:] - first
    shape = [2, 3, 4]
    shape[axis] = lmax + 1 - first
    values = np.random.default_rng(42).normal(size=shape)
    expected = np.take(values, indices, axis=axis)
    actual = expand_degree_axis(values, lmax, mmax, axis=axis, scalar=scalar)
    np.testing.assert_array_equal(actual, expected)


def test_gated_activation_rejects_inconsistent_serialized_layout() -> None:
    """Stored structural indices must describe the configured coefficient order."""
    layer = GatedActivation(lmax=2, mmax=1, channels=2, precision="float64")
    data = layer.serialize()
    GatedActivation.deserialize(data)
    data["@variables"]["expand_index"] = data["@variables"]["expand_index"][::-1]
    with pytest.raises(ValueError, match=r"expand_index.*degree layout"):
        GatedActivation.deserialize(data)
