# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the dpmodel ``SO3GridProjector`` (Wigner-D grid quadrature).

Compares the dpmodel port against the reference pt implementation
(``deepmd.pt.model.descriptor.sezm_nn.projection.SO3GridProjector``) and checks
the legal-frame round-trip identity, serialization round-trip, and the kmax=0
zonal convention. pt imports live inside the test functions to satisfy the
``source/tests/common`` import-isolation rule (ruff TID253).
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
    SO3GridProjector,
)


def _legal_so3_frame_mask(projector: SO3GridProjector) -> np.ndarray:
    """Build the boolean mask of legal ``(l, m, k)`` flat-coefficient slots."""
    mask = np.ones(projector.coeff_dim, dtype=np.bool_)
    n_frames = projector.n_frames
    for degree in range(projector.lmax + 1):
        for m_order in range(-degree, degree + 1):
            packed_idx = degree * degree + degree + m_order
            for frame_pos, frame_order in enumerate(projector.frame_set):
                flat_idx = packed_idx * n_frames + frame_pos
                if flat_idx >= projector.coeff_dim:
                    continue
                if abs(frame_order) > degree:
                    mask[flat_idx] = False
    return mask


# (lmax, kmax, mmax): max degree, frame-index half-width, retained order
_CASES = [(1, 1, 1), (2, 1, 1), (2, 2, 2), (3, 1, 1)]


@pytest.mark.parametrize("lmax,kmax,mmax", _CASES)
def test_projection_matrices_match_pt(lmax, kmax, mmax) -> None:
    """Dpmodel projection matrices match the pt buffers at fp64."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.projection import (
        SO3GridProjector as PTSO3GridProjector,
    )

    dp = SO3GridProjector(lmax=lmax, mmax=mmax, kmax=kmax, precision="float64")
    pt = PTSO3GridProjector(lmax=lmax, mmax=mmax, kmax=kmax, dtype=torch.float64)

    np.testing.assert_allclose(
        dp.to_grid_mat,
        pt.to_grid_mat.detach().cpu().numpy(),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dp.from_grid_mat,
        pt.from_grid_mat.detach().cpu().numpy(),
        atol=1e-12,
        rtol=1e-12,
    )
    assert dp.n_frames == pt.n_frames
    assert dp.coeff_dim == pt.coeff_dim
    assert dp.grid_size == pt.grid_size
    assert dp.frame_set == pt.frame_set


@pytest.mark.parametrize("lmax", [1, 2, 3, 4, 5, 6])  # max degree
def test_roundtrip_preserves_legal_frame_coeffs(lmax) -> None:
    """Project legal-frame coefficients to grid and back; recovery to 1e-11.

    The round-trip is a chain of Wigner-D / Lebedev-quadrature matrix products,
    so the float64 recovery residual grows with the coefficient count and hence
    with ``lmax``; at ``lmax=6`` it sits at ~1e-12. A tolerance of 1e-11 keeps
    an order-of-magnitude margin over that floor while still asserting recovery
    to eleven significant digits.
    """
    rng = np.random.default_rng(8100 + lmax)
    projector = SO3GridProjector(lmax=lmax, kmax=1, precision="float64")
    x = rng.standard_normal((2, projector.coeff_dim, 2)).astype(np.float64)
    mask = _legal_so3_frame_mask(projector)
    x[:, ~mask, :] = 0.0
    y = projector.from_grid(projector.to_grid(x))
    np.testing.assert_allclose(y[:, mask, :], x[:, mask, :], atol=1e-11, rtol=1e-11)
    assert float(np.max(np.abs(y[:, ~mask, :]))) < 1e-14


@pytest.mark.parametrize("lmax,kmax,mmax", _CASES)
def test_serialize_roundtrip(lmax, kmax, mmax) -> None:
    """Serialize -> deserialize reproduces the matrices and config keys."""
    projector = SO3GridProjector(lmax=lmax, mmax=mmax, kmax=kmax, precision="float64")
    data = projector.serialize()
    assert data["@class"] == "SO3GridProjector"
    assert data["@version"] == 1
    config = data["config"]
    for key in (
        "lmax",
        "mmax",
        "kmax",
        "precision",
        "lebedev_precision",
        "coefficient_layout",
    ):
        assert key in config
    # the matrices must NOT be serialized (rebuilt at deserialize)
    assert "to_grid_mat" not in config
    assert "from_grid_mat" not in config

    restored = SO3GridProjector.deserialize(data)
    np.testing.assert_array_equal(restored.to_grid_mat, projector.to_grid_mat)
    np.testing.assert_array_equal(restored.from_grid_mat, projector.from_grid_mat)
    assert restored.frame_set == projector.frame_set


def test_kmax_zero_zonal() -> None:
    """kmax=0 collapses to a single frame and matches the Wigner zonal column."""
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
        build_edge_quaternion,
    )
    from deepmd.dpmodel.utils.lebedev import (
        load_lebedev_rule,
    )

    lmax = 6
    projector = SO3GridProjector(lmax=lmax, kmax=0, precision="float64")
    assert projector.n_frames == 1

    points, _ = load_lebedev_rule(projector.lebedev_precision)
    points = np.asarray(points, dtype=np.float64)
    edge_quaternion = build_edge_quaternion(points, eps=1e-14)
    zonal = WignerDCalculator(lmax, precision="float64").forward_zonal(
        edge_quaternion, lmin=1
    )
    np.testing.assert_allclose(
        projector.to_grid_mat[:, 0], np.ones_like(points[:, 0]), atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(
        projector.to_grid_mat[:, 1:], zonal, atol=1e-14, rtol=1e-14
    )

    # the single-frame projector still round-trips legal coefficients; the
    # lmax=6 recovery residual sits at ~1e-12 (float64 re-association in the
    # Wigner-D monomial products accumulates through the round-trip), so it is
    # asserted to 1e-11 for the same reason as
    # ``test_roundtrip_preserves_legal_frame_coeffs``.
    rng = np.random.default_rng(99)
    x = rng.standard_normal((2, projector.coeff_dim, 2)).astype(np.float64)
    mask = _legal_so3_frame_mask(projector)
    x[:, ~mask, :] = 0.0
    y = projector.from_grid(projector.to_grid(x))
    np.testing.assert_allclose(y[:, mask, :], x[:, mask, :], atol=1e-11, rtol=1e-11)
