# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

import deepmd.dpmodel.utils.lebedev as lebedev_module
from deepmd.dpmodel.utils.spherical_harmonics import (
    real_spherical_harmonics,
)


class TestLebedevRules:
    @pytest.mark.parametrize("precision", [3, 9, 11, 29])  # quadrature precision order
    def test_rule_basic(self, precision):
        pts, wts = lebedev_module.load_lebedev_rule(precision)
        assert pts.shape[1] == 3 and wts.shape == (pts.shape[0],)
        assert pts.shape[0] == lebedev_module.LEBEDEV_PRECISION_TO_NPOINTS[precision]
        np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(wts.sum(), 1.0, rtol=1e-12)


class TestRealSphericalHarmonics:
    @pytest.mark.parametrize("lmax", [0, 1, 2, 3, 4, 6])  # maximum angular degree
    def test_matches_e3nn(self, lmax):
        pytest.importorskip("e3nn")
        import torch
        from e3nn import (
            o3,
        )

        rng = np.random.default_rng(0)
        v = rng.standard_normal((64, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # exact call convention used by the SeZM Lebedev projection path
        # (deepmd/pt/model/descriptor/sezm_nn/projection.py)
        ref = o3.spherical_harmonics(
            list(range(lmax + 1)),
            torch.from_numpy(v),
            normalize=True,
            normalization="norm",
        ).numpy()
        out = real_spherical_harmonics(v, lmax)
        assert out.shape == (64, (lmax + 1) ** 2)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-13)

    def test_basis_vectors_lmax2(self):
        # e3nn-free convention pin: analytic SH values at lmax=2 for the
        # Cartesian basis vectors. Cross-checked against
        # e3nn.o3.spherical_harmonics([0, 1, 2], v, normalize=True,
        # normalization="norm"): the analytic values below equal the e3nn
        # output to the last bit (max |diff| = 0.0).
        h = np.sqrt(3.0) / 2.0
        cases = [
            ((1.0, 0.0, 0.0), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, -h]),
            ((0.0, 1.0, 0.0), [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            ((0.0, 0.0, 1.0), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.5, 0.0, h]),
        ]
        for vec, ref in cases:
            out = real_spherical_harmonics(np.array([vec]), 2)
            np.testing.assert_allclose(out[0], ref, rtol=1e-12, atol=1e-15)

    def test_quadrature_orthogonality(self):
        lmax = 3
        pts, wts = lebedev_module.load_lebedev_rule(2 * lmax + 1)
        sh = real_spherical_harmonics(pts, lmax)
        gram = (sh[:, :, None] * sh[:, None, :] * wts[:, None, None]).sum(axis=0)
        # The implemented convention is e3nn normalization="norm":
        #   Y_lm = sqrt(4*pi/(2l+1)) * Y_lm^{orthonormal}
        # so int Y_lm Y_l'm' dOmega = (4*pi/(2l+1)) * delta_ll' delta_mm'.
        # Lebedev weights sum to 1 (absorbing the 1/(4*pi) surface factor),
        # hence gram = blockdiag over l of I_{2l+1} / (2l+1).
        expected = np.zeros_like(gram)
        for ll in range(lmax + 1):
            for mm in range(ll * ll, (ll + 1) ** 2):
                expected[mm, mm] = 1.0 / (2 * ll + 1)
        # products of degree <= 2*lmax are integrated exactly by the
        # precision-(2*lmax+1) Lebedev rule -> machine precision
        np.testing.assert_allclose(gram, expected, atol=1e-12, rtol=0)
