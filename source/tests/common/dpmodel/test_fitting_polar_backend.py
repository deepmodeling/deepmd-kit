# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-boundary regressions for dpmodel polarizability fitting."""

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from deepmd.dpmodel.fitting.polarizability_fitting import (
    PolarFitting,
)


class TestPolarFittingBackendBuffers(unittest.TestCase):
    """Keep portable polar buffers compatible with backend-native inputs."""

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_scale_and_shift_follow_torch_prediction(self) -> None:
        fitting = PolarFitting(
            ntypes=2,
            dim_descrpt=3,
            embedding_width=2,
            neuron=[5, 5],
            precision="float64",
            mixed_types=True,
            fit_diag=True,
            scale=[1.5, 0.25],
            shift_diag=True,
            seed=20260717,
        )
        fitting.constant_matrix[:] = [0.75, -0.5]

        descriptor = np.array([[[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]]], dtype=np.float64)
        atype = np.array([[0, 1]], dtype=np.int64)
        rotation = np.array(
            [
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ],
            dtype=np.float64,
        )
        expected = fitting(descriptor, atype, gr=rotation)["polarizability"]

        result = fitting(
            torch.as_tensor(descriptor),
            torch.as_tensor(atype),
            gr=torch.as_tensor(rotation),
        )["polarizability"]

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device.type, "cpu")
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected)
