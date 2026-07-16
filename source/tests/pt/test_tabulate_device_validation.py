# SPDX-License-Identifier: LGPL-3.0-or-later
"""Device-contract regression tests for PyTorch tabulation custom ops."""

import unittest

import torch

from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)


@unittest.skipIf(not ENABLE_CUSTOMIZED_OP, "PyTorch customized OPs are not built")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTabulateDeviceValidation(unittest.TestCase):
    """Reject mixed raw-pointer inputs before native kernel dispatch."""

    def setUp(self) -> None:
        self.table_info = torch.zeros(6, dtype=torch.float64, device="cpu")

    def test_se_r_rejects_embedding_on_different_device(self) -> None:
        table = torch.zeros((1, 6), dtype=torch.float64, device="cpu")
        em = torch.zeros((1, 1), dtype=torch.float64, device="cuda")

        with self.assertRaisesRegex(
            RuntimeError, r"em must be on the same device as table"
        ):
            torch.ops.deepmd.tabulate_fusion_se_r(table, self.table_info, em, 1)

    def test_se_r_rejects_cpu_embedding_for_cuda_table(self) -> None:
        table = torch.zeros((1, 6), dtype=torch.float64, device="cuda")
        em = torch.zeros((1, 1), dtype=torch.float64, device="cpu")

        with self.assertRaisesRegex(
            RuntimeError, r"em must be on the same device as table"
        ):
            torch.ops.deepmd.tabulate_fusion_se_r(table, self.table_info, em, 1)

    def test_se_atten_rejects_optional_embedding_on_different_device(self) -> None:
        table = torch.zeros((1, 6), dtype=torch.float64, device="cuda")
        em_x = torch.zeros((1, 1), dtype=torch.float64, device="cuda")
        em = torch.zeros((1, 1, 1), dtype=torch.float64, device="cuda")
        two_embed = torch.zeros((1, 1), dtype=torch.float64, device="cpu")

        with self.assertRaisesRegex(
            RuntimeError, r"two_embed must be on the same device as table"
        ):
            torch.ops.deepmd.tabulate_fusion_se_atten(
                table,
                self.table_info,
                em_x,
                em,
                two_embed,
                1,
                True,
            )

    def test_table_info_remains_cpu_only(self) -> None:
        table = torch.zeros((1, 6), dtype=torch.float64, device="cuda")
        table_info = self.table_info.to("cuda")
        em = torch.zeros((1, 1), dtype=torch.float64, device="cuda")

        with self.assertRaisesRegex(RuntimeError, r"table_info must be on the CPU"):
            torch.ops.deepmd.tabulate_fusion_se_r(table, table_info, em, 1)


if __name__ == "__main__":
    unittest.main()
