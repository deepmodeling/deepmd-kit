# SPDX-License-Identifier: LGPL-3.0-or-later
"""The fixture cache key must change when imported generator helpers change."""

import hashlib
import shlex
import subprocess
import tempfile
import unittest
from pathlib import (
    Path,
)


class TestCCFixtureCacheFingerprint(unittest.TestCase):
    def test_shared_helper_change_invalidates_fingerprint(self):
        workflow = (
            Path(__file__).resolve().parents[3]
            / ".github"
            / "workflows"
            / "test_cc.yml"
        ).read_text(encoding="utf-8")
        command = workflow.split("source_fingerprint=$(git ls-files -s --", 1)[
            1
        ].split("| sha256sum", 1)[0]
        pathspecs = shlex.split(command.replace("\\\n", " "))
        self.assertIn("source/tests/dpa4_fixtures.py", pathspecs)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            helper = repo / "source" / "tests" / "dpa4_fixtures.py"
            generator = repo / "source" / "tests" / "infer" / "gen_dpa4_spin.py"
            helper.parent.mkdir(parents=True)
            generator.parent.mkdir(parents=True)
            helper.write_text("jitter = 0.05\n", encoding="utf-8")
            generator.write_text("from dpa4_fixtures import jitter\n", encoding="utf-8")

            def git(*args):
                return subprocess.run(
                    ["git", *args],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                ).stdout

            def fingerprint():
                return hashlib.sha256(
                    git("ls-files", "-s", "--", *pathspecs)
                ).hexdigest()

            git("init", "-q")
            git("add", "--", "source/tests")
            before = fingerprint()
            # The generator is unchanged; only its imported helper changes.
            helper.write_text("jitter = 0.10\n", encoding="utf-8")
            git("add", "--", "source/tests/dpa4_fixtures.py")
            self.assertNotEqual(before, fingerprint())


if __name__ == "__main__":
    unittest.main()
