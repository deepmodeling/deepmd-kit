# SPDX-License-Identifier: LGPL-3.0-or-later
"""Missing neighbor-list backends must say what to install.

The DPA-4 / SeZM descriptor needs a neighbor-list backend at runtime, but
neither ``vesin`` nor ``nvalchemiops`` was declared anywhere in the project
metadata, so a plain install got as far as building a model and then failed
with a message that named the imports and not the packages. These tests pin
both halves of the fix: the message is actionable, and an extra declares the
dependency.
"""

import pathlib
import tomllib
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
        self.assertIn("deepmd-kit[dpa4]", self._message_without_backends())

    def test_dpa4_extra_declares_the_runtime_dependencies(self) -> None:
        """The extra must cover what DPA-4 needs and nothing else declares."""
        root = pathlib.Path(__file__).resolve().parents[3]
        data = tomllib.loads((root / "pyproject.toml").read_text())
        extras = data["tool"]["deepmd_build_backend"]["optional-dependencies"]
        self.assertIn("dpa4", extras)
        joined = " ".join(extras["dpa4"])
        self.assertIn("e3nn", joined)
        self.assertIn("vesin", joined)
        # the pre-existing extra pulls the full toolkit for deepmd.pt.nvalchemi
        # and must not be repurposed for the neighbor list
        self.assertIn("nvalchemi-toolkit", " ".join(extras["nvalchemi"]))


if __name__ == "__main__":
    unittest.main()
