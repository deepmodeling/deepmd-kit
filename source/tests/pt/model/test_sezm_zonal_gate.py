# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gate tests for the fused-zonal-scatter level binding.

The scatter serves the fused-GIE contract of the inference levels: the
CUDA level-1 suite selects it, and so does the Triton level -- where a
compiled CUDA operator, when available, takes precedence over the Triton
implementation of the same contract. There is no standalone per-operator
opt-in; the cumulative levels are the single control, and with every
level off the binding stays exactly at the previous default behavior.
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


class TestSeZMZonalLevelGate(unittest.TestCase):
    """The scatter binding follows the cumulative inference levels alone."""

    def setUp(self):
        import os

        self._saved = {
            k: os.environ.get(k) for k in ("DP_CUDA_INFER", "DP_TRITON_INFER")
        }
        os.environ.pop("DP_CUDA_INFER", None)
        os.environ.pop("DP_TRITON_INFER", None)

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

    def test_cuda_suite_binds(self):
        import os

        from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
            op_available,
        )

        os.environ["DP_CUDA_INFER"] = "1"
        emb = _seeded_embedding()
        if op_available():
            self.assertTrue(emb._cuda_scatter)
        else:
            self.skipTest("compiled zonal scatter operator unavailable")

    def test_triton_level_binds_without_the_cuda_suite(self):
        import os

        from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
            op_available,
        )

        os.environ["DP_TRITON_INFER"] = "1"
        os.environ["DP_CUDA_INFER"] = "0"
        emb = _seeded_embedding()
        if op_available():
            self.assertTrue(
                emb._cuda_scatter,
                "the Triton level must reach the compiled CUDA operator "
                "without raising the rest of the CUDA suite",
            )
        else:
            self.skipTest("compiled zonal scatter operator unavailable")

    def test_all_levels_off_keeps_the_previous_default(self):
        import os

        os.environ["DP_CUDA_INFER"] = "0"
        os.environ["DP_TRITON_INFER"] = "0"
        emb = _seeded_embedding()
        self.assertFalse(emb._cuda_scatter)


if __name__ == "__main__":
    unittest.main()
