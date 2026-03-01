# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pretrained backend registration and alias parsing."""

import unittest
from unittest.mock import (
    patch,
)

import deepmd.backend  # noqa: F401
from deepmd.backend.backend import (
    Backend,
)
from deepmd.backend.pretrained import (
    PretrainedBackend,
)
from deepmd.pretrained.backend import (
    parse_pretrained_alias,
)


class TestPretrainedBackend(unittest.TestCase):
    """Test pretrained backend integration points."""

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

    def test_deep_eval_property_is_lazy(self) -> None:
        with patch(
            "deepmd.pretrained.backend.get_pretrained_deep_eval_backend",
            return_value=object,
        ):
            self.assertIs(PretrainedBackend().deep_eval, object)
