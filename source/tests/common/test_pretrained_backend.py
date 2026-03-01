# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pretrained backend registration and alias parsing."""

import importlib
import unittest

from deepmd.backend.backend import (
    Backend,
)
from deepmd.backend.pretrained import (
    PretrainedBackend,
)
from deepmd.pretrained.deep_eval import (
    parse_pretrained_alias,
)


class TestPretrainedBackend(unittest.TestCase):
    """Test pretrained backend integration points."""

    @classmethod
    def setUpClass(cls) -> None:
        # ensure backend registration side effects are loaded
        importlib.import_module("deepmd.backend")

    def test_detect_backend_by_pretrained_suffix(self) -> None:
        backend = Backend.detect_backend_by_model("DPA-3.2-5M.pretrained")
        self.assertIs(backend, PretrainedBackend)

    def test_parse_pretrained_alias(self) -> None:
        self.assertEqual(
            parse_pretrained_alias("DPA-3.2-5M.pretrained"),
            "DPA-3.2-5M",
        )

    def test_parse_pretrained_alias_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_pretrained_alias("DPA-3.2-5M.pt")

    def test_deep_eval_property(self) -> None:
        self.assertIsNotNone(PretrainedBackend().deep_eval)
