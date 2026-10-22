# SPDX-License-Identifier: LGPL-3.0-or-later
"""A missing neighbor-list backend must say what to install.

The DPA-4 / SeZM descriptor needs ``vesin`` or ``nvalchemiops`` at runtime.
Both ship with the ``torch`` extra, so a normal install has one, but an
install that skips that extra reaches a model build and then fails with a
message naming the *imports* rather than the distributions that provide
them. These tests pin that the message names installable packages.
"""

import unittest

import torch

from deepmd.pt_expt.utils import (
    auto_neighbor_list,
)


class TestNeighborBackendMissing(unittest.TestCase):
    def _message_without_backends(self, method: str = "native") -> str:
        """The error raised when neither backend is importable."""
        nv = auto_neighbor_list.is_nv_available
        vesin = auto_neighbor_list.is_vesin_torch_available
        auto_neighbor_list.is_nv_available = lambda: False
        auto_neighbor_list.is_vesin_torch_available = lambda: False
        try:
            with self.assertRaises(RuntimeError) as caught:
                builder = auto_neighbor_list.make_neighbor_list(
                    method, torch.device("cpu"), return_mode="edges"
                )
                builder.build(
                    torch.zeros((1, 1, 3), device="cpu"),
                    torch.zeros((1, 1), dtype=torch.int64, device="cpu"),
                    None,
                    1.0,
                    [1],
                    return_mode="edges",
                )
            return str(caught.exception)
        finally:
            auto_neighbor_list.is_nv_available = nv
            auto_neighbor_list.is_vesin_torch_available = vesin

    def test_error_names_installable_packages(self) -> None:
        """The message must name pip-installable distributions, not imports."""
        message = self._message_without_backends()
        self.assertIn("pip install", message)
        # the distribution names, which differ from the import names
        self.assertIn("vesin[torch]", message)
        self.assertIn("nvalchemi-toolkit-ops", message)

    def test_error_points_at_the_extra(self) -> None:
        """A user who wants the whole descriptor should be told about it."""
        self.assertIn("deepmd-kit[torch]", self._message_without_backends("auto"))


if __name__ == "__main__":
    unittest.main()
