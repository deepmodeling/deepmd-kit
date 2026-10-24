# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for offline, version-matched workflow discovery."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepmd.main import main_parser, parse_args
from deepmd.skills import read_skill, run, skill_root


class TestSkills(unittest.TestCase):
    def test_list_and_read(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run(parse_args(["skills", "list"]))
        self.assertIn("deepmd-train", output.getvalue().splitlines())
        expected = (skill_root() / "deepmd-train" / "SKILL.md").read_bytes()
        self.assertEqual(read_skill("deepmd-train").encode(), expected)

    def test_help_and_legacy_parser(self) -> None:
        self.assertIn("skills", main_parser().format_help())
        args = parse_args(["train", "input.json"])
        self.assertEqual(args.INPUT, "input.json")

    def test_invalid_paths(self) -> None:
        for name, path in [
            ("../deepmd-train", "SKILL.md"),
            ("missing", "SKILL.md"),
            ("deepmd-train", "../README.md"),
            ("deepmd-train", "/etc/passwd"),
            ("deepmd-train", "..\\README.md"),
            ("deepmd-train", "C:/file"),
            ("deepmd-train", "SKILL.md\0"),
        ]:
            with self.subTest(name=name, path=path), self.assertRaises(ValueError):
                read_skill(name, path)

    def test_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "skills" / "guide"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("guide", encoding="utf-8")
            outside = root / "private.txt"
            outside.write_text("private", encoding="utf-8")
            (skill / "escape.md").symlink_to(outside)
            with patch("deepmd.skills.skill_root", return_value=root / "skills"):
                with self.assertRaises(ValueError):
                    read_skill("guide", "escape.md")

    def test_cli_leaves_home_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            result = subprocess.run(
                [sys.executable, "-m", "deepmd", "skills", "list"],
                env={**os.environ, "HOME": home},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("deepmd-train", result.stdout)
            self.assertEqual(list(Path(home).iterdir()), [])

    def test_no_backend_import(self) -> None:
        code = (
            "import sys; "
            "sys.modules.update({name: None for name in "
            "('tensorflow', 'torch', 'jax', 'paddle')}); "
            "from deepmd.main import main; main(['skills', 'list'])"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
