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
