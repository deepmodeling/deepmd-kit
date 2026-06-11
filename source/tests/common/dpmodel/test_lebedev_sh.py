# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

import deepmd.dpmodel.utils.lebedev as lebedev_module
from deepmd.dpmodel.utils.lebedev import (
    LEBEDEV_PRECISION_TO_NPOINTS,
    load_lebedev_rule,
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
