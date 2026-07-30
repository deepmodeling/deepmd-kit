# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reject invalid tabulation widths before any PyTorch GPU kernel launch."""

import unittest
from collections.abc import (
    Callable,
)

import torch

from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)

ERROR_MESSAGE = "last_layer_size must be between 1 and 1024"


@unittest.skipIf(not ENABLE_CUSTOMIZED_OP, "PyTorch customized OPs are not built")
@unittest.skipUnless(
    torch.cuda.is_available(), "GPU tabulation validation requires a GPU"
)
class TestTabulateGpuSizeValidation(unittest.TestCase):
    """The GPU wrappers must fail on the host, not inside a launch config."""

    dtype = torch.float64

    def _zeros(self, *shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype, device="cuda")

    def _table(self, last_layer_size: int) -> torch.Tensor:
        return self._zeros(1, max(1, 6 * last_layer_size))

    def _table_info(self) -> torch.Tensor:
        # table_info stays on the CPU for every backend.
        return torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=self.dtype)

    def _forward_calls(self, last_layer_size: int) -> dict[str, Callable[[], object]]:
        table = self._table(last_layer_size)
        table_info = self._table_info()
        em_x = self._zeros(1, 1)
        em_a = self._zeros(1, 1, 4)
        em_t = self._zeros(1, 1, 1)
        em_r = self._zeros(1, 1)
        two_embed = self._zeros(1, max(1, last_layer_size))
        return {
            "se_a": lambda: torch.ops.deepmd.tabulate_fusion_se_a(
                table, table_info, em_x, em_a, last_layer_size
            ),
            "se_atten": lambda: torch.ops.deepmd.tabulate_fusion_se_atten(
                table, table_info, em_x, em_a, two_embed, last_layer_size, True
            ),
            "se_t": lambda: torch.ops.deepmd.tabulate_fusion_se_t(
                table, table_info, em_x, em_t, last_layer_size
            ),
            "se_t_tebd": lambda: torch.ops.deepmd.tabulate_fusion_se_t_tebd(
                table, table_info, em_x, em_t, last_layer_size
            ),
            "se_r": lambda: torch.ops.deepmd.tabulate_fusion_se_r(
                table, table_info, em_r, last_layer_size
            ),
        }

    def _assert_rejected(self, last_layer_size: int) -> None:
        for name, call in self._forward_calls(last_layer_size).items():
            with self.subTest(op=name, last_layer_size=last_layer_size):
                with self.assertRaisesRegex(RuntimeError, ERROR_MESSAGE):
                    call()

    def test_rejects_oversized_width(self) -> None:
        """A width past the maximum block dimension must be refused."""
        self._assert_rejected(1025)

    def test_rejects_nonpositive_width(self) -> None:
        """Zero and negative widths are invalid launch dimensions."""
        self._assert_rejected(0)
        self._assert_rejected(-1)

    def _gradient_cases(
        self,
    ) -> dict[
        str,
        tuple[
            Callable[[], torch.Tensor],
            tuple[torch.Tensor, ...],
            tuple[int, ...],
        ],
    ]:
        """Build valid forwards whose saved descriptors can test grad guards."""
        last_layer_size = 2
        table = self._table(last_layer_size)
        table_info = self._table_info()
        em_x = self._zeros(1, 1).requires_grad_(True)
        em_a = self._zeros(1, 1, 4).requires_grad_(True)
        two_embed = self._zeros(1, last_layer_size).requires_grad_(True)
        em_t = self._zeros(1, 1, 1).requires_grad_(True)
        em_t_tebd_x = self._zeros(1, 1).requires_grad_(True)
        em_t_tebd = self._zeros(1, 1, 1).requires_grad_(True)
        em_r = self._zeros(1, 1).requires_grad_(True)
        return {
            "se_a": (
                lambda: torch.ops.deepmd.tabulate_fusion_se_a(
                    table, table_info, em_x, em_a, last_layer_size
                )[0],
                (em_x, em_a),
                (1, 4, last_layer_size),
            ),
            "se_atten": (
                lambda: torch.ops.deepmd.tabulate_fusion_se_atten(
                    table,
                    table_info,
                    em_x,
                    em_a,
                    two_embed,
                    last_layer_size,
                    True,
                )[0],
                (em_x, em_a, two_embed),
                (1, 4, last_layer_size),
            ),
            "se_t": (
                lambda: torch.ops.deepmd.tabulate_fusion_se_t(
                    table, table_info, em_x, em_t, last_layer_size
                )[0],
                (em_x, em_t),
                (1, last_layer_size),
            ),
            "se_t_tebd": (
                lambda: torch.ops.deepmd.tabulate_fusion_se_t_tebd(
                    table,
                    table_info,
                    em_t_tebd_x,
                    em_t_tebd,
                    last_layer_size,
                )[0],
                (em_t_tebd_x,),
                (1, 1, 1, last_layer_size),
            ),
            "se_r": (
                lambda: torch.ops.deepmd.tabulate_fusion_se_r(
                    table, table_info, em_r, last_layer_size
                )[0],
                (em_r,),
                (1, 1, last_layer_size),
            ),
        }

    def _oversized_saved_descriptor(
        self, descriptor_shape: tuple[int, ...]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Replace only a saved descriptor with an oversized-width sentinel."""

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            if tuple(tensor.shape) == descriptor_shape:
                return self._zeros(*descriptor_shape[:-1], 1025)
            return tensor

        return pack

    @staticmethod
    def _unpack_saved_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Return saved tensors unchanged when their autograd node reloads."""
        return tensor

    def test_first_gradient_wrappers_reject_oversized_width(self) -> None:
        """Every first-gradient wrapper validates its saved descriptor width."""
        for name, (forward, inputs, descriptor_shape) in self._gradient_cases().items():
            with self.subTest(op=name):
                pack = self._oversized_saved_descriptor(descriptor_shape)
                with torch.autograd.graph.saved_tensors_hooks(
                    pack, self._unpack_saved_tensor
                ):
                    descriptor = forward()
                with self.assertRaisesRegex(RuntimeError, ERROR_MESSAGE):
                    torch.autograd.grad(descriptor.sum(), inputs)

    def test_second_gradient_wrappers_reject_oversized_width(self) -> None:
        """Every grad-grad wrapper validates the descriptor before launching."""
        for name, (forward, inputs, descriptor_shape) in self._gradient_cases().items():
            with self.subTest(op=name):
                descriptor = forward()
                pack = self._oversized_saved_descriptor(descriptor_shape)
                with torch.autograd.graph.saved_tensors_hooks(
                    pack, self._unpack_saved_tensor
                ):
                    first_gradients = torch.autograd.grad(
                        descriptor.sum(), inputs, create_graph=True
                    )
                differentiable_sum = sum(
                    gradient.sum()
                    for gradient in first_gradients
                    if gradient.requires_grad
                )
                with self.assertRaisesRegex(RuntimeError, ERROR_MESSAGE):
                    torch.autograd.grad(differentiable_sum, inputs)


if __name__ == "__main__":
    unittest.main()
