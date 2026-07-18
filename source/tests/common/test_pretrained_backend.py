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
from deepmd.backend.suffix import (
    format_model_suffix,
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

    def test_detect_backend_by_new_model_name(self) -> None:
        backend = Backend.detect_backend_by_model("DPA3-Omol-Large")
        self.assertIs(backend, PretrainedBackend)

    def test_detect_backend_by_pretrained_suffix_not_supported(self) -> None:
        with self.assertRaises(ValueError):
            Backend.detect_backend_by_model("DPA-3.2-5M.pretrained")

    def test_detect_savedmodel_suffix_split(self) -> None:
        self.assertEqual(
            Backend.detect_backend_by_model("model.savedmodel").name,
            "JAX",
        )
        self.assertEqual(
            Backend.detect_backend_by_model("model.savedmodeltf").name,
            "TensorFlow2",
        )

    def test_parse_pretrained_alias_plain_name(self) -> None:
        self.assertEqual(parse_pretrained_alias("DPA-3.2-5M"), "DPA-3.2-5M")
        self.assertEqual(parse_pretrained_alias("dpa-3.2-5m"), "DPA-3.2-5M")

    def test_parse_pretrained_alias_new_model_name(self) -> None:
        self.assertEqual(parse_pretrained_alias("DPA3-Omol-Large"), "DPA3-Omol-Large")
        self.assertEqual(parse_pretrained_alias("dpa3-omol-large"), "DPA3-Omol-Large")

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

    def test_format_model_suffix_keeps_pretrained_alias(self) -> None:
        # CLI deep-eval commands normalize `-m` through format_model_suffix
        # (strict_prefer=False). A pretrained alias must be recognized and
        # returned unchanged so PretrainedBackend can resolve it, matching the
        # endswith-based detection in Backend.detect_backend_by_model. Before
        # the fix these were mangled (e.g. "DPA-3.2-5M" -> "DPA-3.2-5M.pth")
        # because only Path(...).suffix was compared against the alias list.
        for alias in ("DPA-3.2-5M", "DPA3-Omol-Large", "dpa-3.2-5m"):
            self.assertEqual(
                format_model_suffix(
                    alias,
                    feature=Backend.Feature.DEEP_EVAL,
                    preferred_backend="pytorch",
                    strict_prefer=False,
                ),
                alias,
            )

    def test_format_model_suffix_keeps_regular_suffix(self) -> None:
        # Control: an ordinary backend suffix is still returned unchanged.
        self.assertEqual(
            format_model_suffix(
                "model.pth",
                feature=Backend.Feature.DEEP_EVAL,
                preferred_backend="pytorch",
                strict_prefer=False,
            ),
            "model.pth",
        )

    def test_format_model_suffix_appends_for_unknown(self) -> None:
        # Control: a genuinely unknown name still gets the preferred suffix.
        self.assertEqual(
            format_model_suffix(
                "model",
                feature=Backend.Feature.DEEP_EVAL,
                preferred_backend="pytorch",
                strict_prefer=False,
            ),
            "model" + PretrainedBackend.get_backend("pytorch").suffixes[0],
        )
