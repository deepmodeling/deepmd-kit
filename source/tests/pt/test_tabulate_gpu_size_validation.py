# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reject invalid tabulation widths before any PyTorch GPU kernel launch."""

import unittest

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

    def _forward_calls(self, last_layer_size: int) -> dict[str, callable]:
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

    def test_rejects_zero_width(self) -> None:
        """A zero width would divide by zero while sizing the launch."""
        self._assert_rejected(0)

    def test_gradient_paths_reject_oversized_width(self) -> None:
        """The autograd wrappers derive the width from the descriptor."""
        last_layer_size = 1025
        table = self._table(last_layer_size)
        table_info = self._table_info()
        em_x = self._zeros(1, 1).requires_grad_(True)
        em_a = self._zeros(1, 1, 4).requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, ERROR_MESSAGE):
            descriptor = torch.ops.deepmd.tabulate_fusion_se_a(
                table, table_info, em_x, em_a, last_layer_size
            )[0]
            descriptor.sum().backward()


if __name__ == "__main__":
    unittest.main()
