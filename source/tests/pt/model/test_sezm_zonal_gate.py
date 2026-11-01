# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gate tests for the standalone fused-zonal-scatter opt-in.

``DP_ZONAL_FUSED_INFER`` must enable the scatter binding without raising
the rest of the ``DP_CUDA_INFER`` level-1 suite, keep the default
(both flags unset) exactly at the previous behavior, and never bind when
the compiled operator is unavailable.
"""

import unittest

import torch

from deepmd.pt.utils.env import (
    DEVICE,
)


def _seeded_embedding():
    from deepmd.pt.model.descriptor.sezm_nn.embedding import (
        GeometricInitialEmbedding,
    )

    return GeometricInitialEmbedding(
        lmax=2,
        channels=64,
        dtype=torch.float32,
    ).to(DEVICE)


class TestSeZMZonalStandaloneGate(unittest.TestCase):
    """The scatter binding follows the standalone flag alone."""

    def setUp(self):
        import os

        self._saved = {
            k: os.environ.get(k) for k in ("DP_CUDA_INFER", "DP_ZONAL_FUSED_INFER")
        }
        os.environ.pop("DP_CUDA_INFER", None)
        os.environ.pop("DP_ZONAL_FUSED_INFER", None)

    def tearDown(self):
        import os

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_off(self):
        emb = _seeded_embedding()
        self.assertFalse(emb._cuda_scatter)

    def test_standalone_flag_binds_without_the_suite(self):
        from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
            op_available,
        )

        import os

        os.environ["DP_ZONAL_FUSED_INFER"] = "1"
        emb = _seeded_embedding()
        if op_available():
            self.assertTrue(emb._cuda_scatter)
        else:
            self.skipTest("compiled zonal scatter operator unavailable")

    def test_suite_still_binds(self):
        from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
            op_available,
        )

        import os

        os.environ["DP_CUDA_INFER"] = "1"
        emb = _seeded_embedding()
        if op_available():
            self.assertTrue(emb._cuda_scatter)
        else:
            self.skipTest("compiled zonal scatter operator unavailable")

    def test_flag_off_overrides_nothing_when_suite_off(self):
        import os

        os.environ["DP_ZONAL_FUSED_INFER"] = "0"
        os.environ["DP_CUDA_INFER"] = "0"
        emb = _seeded_embedding()
        self.assertFalse(emb._cuda_scatter)


if __name__ == "__main__":
    unittest.main()
