# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the DPA4 SO(3) grid utility functions.

Compares the dpmodel ports of ``resolve_so3_grid`` and ``_build_so3_frame_set``
against the reference pt implementations.
"""

import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
    _build_so3_frame_set,
    resolve_so3_grid,
)


@pytest.mark.parametrize("kmax", [0, 1, 2, 3])  # frame-index half-width
def test_build_so3_frame_set(kmax) -> None:
    from deepmd.pt.model.descriptor.sezm_nn.projection import (
        _build_so3_frame_set as pt_build_so3_frame_set,
    )

    assert _build_so3_frame_set(kmax) == pt_build_so3_frame_set(kmax)
    if kmax == 2:
        assert _build_so3_frame_set(kmax) == [0, -1, 1, -2, 2]


@pytest.mark.parametrize(
    "lmax,kmax",  # max angular momentum, frame-index half-width
    [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)],
)
def test_resolve_so3_grid(lmax, kmax) -> None:
    from deepmd.pt.model.descriptor.sezm_nn.projection import (
        resolve_so3_grid as pt_resolve_so3_grid,
    )

    dp_result = resolve_so3_grid(lmax, kmax=kmax)
    pt_result = pt_resolve_so3_grid(lmax, kmax=kmax)
    assert dp_result == pt_result
    n_gamma = dp_result[2]
    assert n_gamma == (1 if kmax == 0 else 3 * kmax + 1)


def test_resolve_so3_grid_kmax_zero() -> None:
    """kmax=0 collapses the gamma grid to a single sample (n_gamma=1)."""
    from deepmd.pt.model.descriptor.sezm_nn.projection import (
        resolve_so3_grid as pt_resolve_so3_grid,
    )

    dp_result = resolve_so3_grid(2, kmax=0)
    assert dp_result == pt_resolve_so3_grid(2, kmax=0)
    assert dp_result[2] == 1


def test_resolve_so3_grid_explicit_precision() -> None:
    """An explicitly supplied (packaged) Lebedev precision is honored."""
    from deepmd.dpmodel.utils.lebedev import (
        LEBEDEV_PRECISION_TO_NPOINTS,
    )
    from deepmd.pt.model.descriptor.sezm_nn.projection import (
        resolve_so3_grid as pt_resolve_so3_grid,
    )

    precision = sorted(LEBEDEV_PRECISION_TO_NPOINTS)[3]
    dp_result = resolve_so3_grid(2, kmax=1, lebedev_precision=precision)
    assert dp_result == pt_resolve_so3_grid(2, kmax=1, lebedev_precision=precision)
    assert dp_result[0] == precision


def test_resolve_so3_grid_unpackaged_precision_raises() -> None:
    """An unpackaged explicit precision raises ValueError."""
    with pytest.raises(ValueError, match="not packaged"):
        resolve_so3_grid(2, kmax=1, lebedev_precision=999999)


@pytest.mark.parametrize("kmax", [-1, -2])  # negative half-width is invalid
def test_negative_kmax_raises(kmax) -> None:
    """Both utilities reject negative kmax."""
    with pytest.raises(ValueError, match="non-negative"):
        _build_so3_frame_set(kmax)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_so3_grid(2, kmax=kmax)
