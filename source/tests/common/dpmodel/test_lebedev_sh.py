# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

import deepmd.dpmodel.utils.lebedev as lebedev_module
from deepmd.dpmodel.utils.lebedev import (
    LEBEDEV_PRECISION_TO_NPOINTS,
    load_lebedev_rule,
)
from deepmd.dpmodel.utils.spherical_harmonics import (
    real_spherical_harmonics,
)


class TestLebedevRules:
    @pytest.mark.parametrize("precision", [3, 9, 11, 29])  # quadrature precision order
    def test_rule_basic(self, precision):
        pts, wts = load_lebedev_rule(precision)
        assert pts.shape[1] == 3 and wts.shape == (pts.shape[0],)
        assert pts.shape[0] == LEBEDEV_PRECISION_TO_NPOINTS[precision]
        np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(wts.sum(), 1.0, rtol=1e-12)

    def test_unpackaged_precision_raises(self):
        with pytest.raises(ValueError, match="not packaged"):
            load_lebedev_rule(4)

    def test_missing_data_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lebedev_module, "LEBEDEV_RULES_FILE", tmp_path / "nope.npz")
        with pytest.raises(FileNotFoundError, match="missing"):
            load_lebedev_rule(11)

    def test_pt_loader_matches(self):
        torch = pytest.importorskip("torch")
        from deepmd.pt.model.descriptor.sezm_nn.lebedev import (
            load_lebedev_rule as pt_rule,
        )

        pts, wts = load_lebedev_rule(11)
        tpts, twts = pt_rule(11, dtype=torch.float64, device="cpu")
        np.testing.assert_allclose(pts, tpts.numpy(), rtol=0, atol=0)
        np.testing.assert_allclose(wts, twts.numpy(), rtol=0, atol=0)


class TestRealSphericalHarmonics:
    @pytest.mark.parametrize("lmax", [0, 1, 2, 3, 4, 6])  # maximum angular degree
    def test_matches_e3nn(self, lmax):
        pytest.importorskip("e3nn")
        import torch
        from e3nn import o3

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

    @pytest.mark.parametrize("lmax", [2, 4])  # maximum angular degree
    def test_scale_invariance(self, lmax):
        # normalize=True in the e3nn call: input vectors are normalized
        # internally, so non-unit inputs must give identical output.
        rng = np.random.default_rng(1)
        v = rng.standard_normal((32, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        scale = rng.uniform(0.1, 10.0, size=(32, 1))
        np.testing.assert_allclose(
            real_spherical_harmonics(v * scale, lmax),
            real_spherical_harmonics(v, lmax),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_batched_leading_dims(self):
        rng = np.random.default_rng(2)
        v = rng.standard_normal((4, 5, 3))
        out = real_spherical_harmonics(v, 3)
        assert out.shape == (4, 5, 16)
        flat = real_spherical_harmonics(v.reshape(-1, 3), 3)
        np.testing.assert_allclose(out.reshape(-1, 16), flat, rtol=0, atol=0)

    def test_quadrature_orthogonality(self):
        lmax = 3
        pts, wts = load_lebedev_rule(2 * lmax + 1)
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
