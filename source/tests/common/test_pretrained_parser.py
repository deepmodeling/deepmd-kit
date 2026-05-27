# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pretrained argument parsing."""

import unittest

from deepmd.main import (
    parse_args,
)
from deepmd.pretrained.registry import (
    available_model_names,
)


class TestPretrainedParser(unittest.TestCase):
    """Test `dp pretrained` parser behavior."""

    def test_pretrained_download_parser(self) -> None:
        model = available_model_names()[0]
        args = parse_args(["pretrained", "download", model])

        self.assertEqual(args.command, "pretrained")
        self.assertEqual(args.pretrained_command, "download")
        self.assertEqual(args.MODEL, model)
        self.assertIsNone(args.cache_dir)

    def test_pretrained_download_with_cache_dir(self) -> None:
        model = available_model_names()[0]
        args = parse_args(
            [
                "pretrained",
                "download",
                model,
                "--cache-dir",
                "/tmp/deepmd-pretrained",
            ]
        )

        self.assertEqual(args.cache_dir, "/tmp/deepmd-pretrained")

    def test_pretrained_download_rejects_unknown_model(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["pretrained", "download", "NOT-EXIST"])
