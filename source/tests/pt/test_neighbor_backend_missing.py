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

from deepmd.pt.model.model import (
    sezm_model,
)


class TestNeighborBackendMissing(unittest.TestCase):
    def _message_without_backends(self) -> str:
        """The error raised when neither backend is importable."""
        nv = sezm_model.is_nv_available
        vesin = sezm_model.is_vesin_torch_available
        sezm_model.is_nv_available = lambda: False
        sezm_model.is_vesin_torch_available = lambda: False
        try:
            with self.assertRaises(RuntimeError) as caught:
                sezm_model._select_neighbor_builder(1, torch.device("cpu"))
            return str(caught.exception)
        finally:
            sezm_model.is_nv_available = nv
            sezm_model.is_vesin_torch_available = vesin

    def test_error_names_installable_packages(self) -> None:
        """The message must name pip-installable distributions, not imports."""
        message = self._message_without_backends()
        self.assertIn("pip install", message)
        # the distribution names, which differ from the import names
        self.assertIn("vesin[torch]", message)
        self.assertIn("nvalchemi-toolkit-ops", message)

    def test_error_points_at_the_extra(self) -> None:
        """A user who wants the whole descriptor should be told about it."""
        self.assertIn("deepmd-kit[torch]", self._message_without_backends())


if __name__ == "__main__":
    unittest.main()
