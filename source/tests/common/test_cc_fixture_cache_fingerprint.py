# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fixture dependency changes invalidate the cache without hashing C++ tests."""

import hashlib
import shlex
import subprocess
import tempfile
import unittest
from pathlib import (
    Path,
)


class TestCCFixtureCacheFingerprint(unittest.TestCase):
    def test_dependency_changes_invalidate_fingerprint(self):
        workflow = (
            Path(__file__).resolve().parents[3]
            / ".github"
            / "workflows"
            / "test_cc.yml"
        ).read_text(encoding="utf-8")
        command = workflow.split("source_fingerprint=$(git ls-files -s --", 1)[1].split(
            "| sha256sum", 1
        )[0]
        pathspecs = shlex.split(command.replace("\\\n", " "))
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            generator = repo / "source" / "tests" / "infer" / "gen_dpa4_spin.py"
            generator.parent.mkdir(parents=True)
            generator.write_text("from dpa4_fixtures import jitter\n", encoding="utf-8")
            dependencies = (
                "source/tests/dpa4_fixtures.py",
                "source/lib/src/tabulate.cc",
                "source/lib/include/tabulate.h",
                "source/CMakeLists.txt",
            )
            for path in dependencies:
                dependency = repo / path
                dependency.parent.mkdir(parents=True, exist_ok=True)
                dependency.write_text("original\n", encoding="utf-8")

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
            git("add", "--", "source")
            before = fingerprint()
            # Each dependency changes independently while the generator stays
            # unchanged: native kernels/headers matter just as Python helpers do.
            for path in dependencies:
                with self.subTest(dependency=path):
                    (repo / path).write_text("changed\n", encoding="utf-8")
                    git("add", "--", path)
                    after = fingerprint()
                    self.assertNotEqual(before, after)
                    before = after

            # Editing a C++ test must not trigger expensive fixture generation.
            unrelated = repo / "source/lib/tests/test_tabulate.cc"
            unrelated.parent.mkdir(parents=True)
            unrelated.write_text("new test\n", encoding="utf-8")
            git("add", "--", "source/lib/tests/test_tabulate.cc")
            self.assertEqual(before, fingerprint())


if __name__ == "__main__":
    unittest.main()
