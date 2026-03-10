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

    def test_detect_backend_by_model_name(self) -> None:
        backend = Backend.detect_backend_by_model("DPA-3.2-5M")
        self.assertIs(backend, PretrainedBackend)

    def test_detect_backend_by_pretrained_suffix_not_supported(self) -> None:
        with self.assertRaises(ValueError):
            Backend.detect_backend_by_model("DPA-3.2-5M.pretrained")

    def test_parse_pretrained_alias_plain_name(self) -> None:
        self.assertEqual(parse_pretrained_alias("DPA-3.2-5M"), "DPA-3.2-5M")
        self.assertEqual(parse_pretrained_alias("dpa-3.2-5m"), "DPA-3.2-5M")

    def test_parse_pretrained_alias_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_pretrained_alias("DPA-3.2-5M.pt")
        with self.assertRaises(ValueError):
            parse_pretrained_alias("DPA-3.2-5M.pretrained")

    def test_deep_eval_property(self) -> None:
        from deepmd.pretrained.deep_eval import (
            PretrainedDeepEvalBackend,
        )

        self.assertIs(PretrainedBackend().deep_eval, PretrainedDeepEvalBackend)
