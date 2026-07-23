# SPDX-License-Identifier: LGPL-3.0-or-later
"""Device-contract regression tests for PyTorch tabulation custom ops."""

import unittest
from collections.abc import (
    Callable,
)

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

    def _se_r_cuda_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build minimal valid SeR inputs for backward device checks."""
        table = torch.zeros((1, 132), dtype=torch.float64, device="cuda")
        table_info = torch.tensor(
            [0.0, 0.2, 0.4, 0.01, 0.1, -1.0],
            dtype=torch.float64,
            device="cpu",
        )
        em = torch.full(
            (1, 1),
            0.1,
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )
        return table, table_info, em

    @staticmethod
    def _offload_saved_tensor(
        target: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Pack one saved CUDA tensor on CPU to emulate a wiring mismatch."""
        target_data_ptr = target.data_ptr()

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.device.type == "cuda" and tensor.data_ptr() == target_data_ptr:
                return tensor.cpu()
            return tensor

        return pack

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

    def test_se_t_variants_reject_embedding_on_different_device(self) -> None:
        table = torch.zeros((1, 6), dtype=torch.float64, device="cuda")
        em_x = torch.zeros((1, 1), dtype=torch.float64, device="cpu")
        em = torch.zeros((1, 1, 1), dtype=torch.float64, device="cuda")

        for op_name in ("tabulate_fusion_se_t", "tabulate_fusion_se_t_tebd"):
            with self.subTest(op_name=op_name):
                op = getattr(torch.ops.deepmd, op_name)
                with self.assertRaisesRegex(
                    RuntimeError, r"em_x must be on the same device as table"
                ):
                    op(table, self.table_info, em_x, em, 1)

    def test_se_r_backward_checks_saved_tensor_devices(self) -> None:
        table, table_info, em = self._se_r_cuda_inputs()
        pack = self._offload_saved_tensor(em)

        # saved_tensors_hooks models a copy/paste wiring error without asking
        # autograd to accept a gradient tensor on the wrong device.
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            descriptor = torch.ops.deepmd.tabulate_fusion_se_r(
                table, table_info, em, 1
            )[0]

        with self.assertRaisesRegex(
            RuntimeError, r"em must be on the same device as table"
        ):
            descriptor.sum().backward()

    def test_se_r_double_backward_checks_saved_tensor_devices(self) -> None:
        table, table_info, em = self._se_r_cuda_inputs()
        descriptor = torch.ops.deepmd.tabulate_fusion_se_r(table, table_info, em, 1)[0]
        pack = self._offload_saved_tensor(em)

        # The first backward creates TabulateFusionSeRGradOp; offloading the
        # tensor it saves makes its subsequent GradGrad device check observable.
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            (gradient,) = torch.autograd.grad(descriptor.sum(), em, create_graph=True)

        with self.assertRaisesRegex(
            RuntimeError, r"em must be on the same device as table"
        ):
            gradient.sum().backward()

    def test_table_info_remains_cpu_only(self) -> None:
        table = torch.zeros((1, 6), dtype=torch.float64, device="cuda")
        table_info = self.table_info.to("cuda")
        em = torch.zeros((1, 1), dtype=torch.float64, device="cuda")

        with self.assertRaisesRegex(RuntimeError, r"table_info must be on the CPU"):
            torch.ops.deepmd.tabulate_fusion_se_r(table, table_info, em, 1)


if __name__ == "__main__":
    unittest.main()
