# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import (
    Path,
)

SCRIPT = Path(__file__).resolve().parents[2] / "install" / "uv_with_retry.sh"


@unittest.skipUnless(shutil.which("bash"), "requires bash")
class TestUvWithRetry(unittest.TestCase):
    def run_uv(self, outcomes, delayed_tee=False):
        """Run the wrapper with fake uv and sleep, without network access."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "tmp"
            temporary.mkdir()
            cases = []
            for attempt, (code, message) in enumerate(outcomes, start=1):
                cases.append(
                    f"{attempt}) printf '%s\\n' {shlex.quote(message)} >&2; "
                    f"exit {code} ;;"
                )
            commands = {
                "uv": """#!/bin/bash
attempt=0
if [ -f "$STATE/attempt" ]; then
    read -r attempt < "$STATE/attempt"
fi
attempt=$((attempt + 1))
printf '%s\\n' "$attempt" > "$STATE/attempt"
printf '%s\\n' "$@" > "$STATE/args"
printf 'stdout %s\\n' "$attempt"
case "$attempt" in
"""
                + "\n".join(cases)
                + "\n*) exit 99 ;;\nesac\n",
                "sleep": '#!/bin/bash\nprintf "%s\\n" "$@" >> "$STATE/sleeps"\n',
            }
            if delayed_tee:
                commands["tee"] = (
                    "#!/bin/bash\n"
                    f"{shlex.quote(shutil.which('sleep'))} 0.1\n"
                    f'exec {shlex.quote(shutil.which("tee"))} "$@"\n'
                )
            for name, content in commands.items():
                executable = root / name
                executable.write_text(content, encoding="utf-8")
                executable.chmod(0o755)
            arguments = ["pip", "install", "package with spaces", "--find-links", "url"]
            result = subprocess.run(
                ["bash", str(SCRIPT), *arguments],
                env={
                    **os.environ,
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "STATE": str(root),
                    "TMPDIR": str(temporary),
                },
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual((root / "args").read_text().splitlines(), arguments)
            self.assertEqual(list(temporary.iterdir()), [])
            attempts = int((root / "attempt").read_text())
            sleeps = root / "sleeps"
            delays = sleeps.read_text().splitlines() if sleeps.exists() else []
            self.assertEqual(
                result.stderr,
                "".join(f"{message}\n" for _, message in outcomes[:attempts]),
            )
            for attempt in range(1, attempts + 1):
                self.assertIn(f"stdout {attempt}\n", result.stdout)
            return result, attempts, delays

    def test_success(self):
        result, attempts, delays = self.run_uv([(0, "installed")])
        self.assertEqual((result.returncode, attempts, delays), (0, 1, []))

    def test_transient_errors(self):
        messages = [
            "error decoding response body",
            "network timeout",
            "I/O operation failed during extraction",
            "Failed to download distribution",
            "HTTP status client error (429 Too Many Requests) for url (https://example.com)",
        ]
        for code in (500, 502, 503, 504, 599):
            messages.append(
                "error: Failed to read `--find-links` URL: "
                "https://www.paddlepaddle.org.cn/packages/nightly/cpu/paddlepaddle/\n"
                "  Caused by: Request failed after 3 retries\n"
                f"  Caused by: HTTP status server error ({code} Server Error) for url "
                "(https://www.paddlepaddle.org.cn/packages/nightly/cpu/paddlepaddle/)"
            )
        for message in messages:
            with self.subTest(message=message):
                result, attempts, delays = self.run_uv([(2, message), (0, "installed")])
                self.assertEqual((result.returncode, attempts, delays), (0, 2, ["5"]))

    def test_permanent_errors(self):
        for message in (
            "No solution found when resolving dependencies",
            "HTTP status client error (401 Unauthorized)",
            "HTTP status client error (403 Forbidden)",
            "HTTP status client error (404 Not Found)",
            "invalid argument: 502",
        ):
            with self.subTest(message=message):
                result, attempts, delays = self.run_uv([(2, message)])
                self.assertEqual((result.returncode, attempts, delays), (2, 1, []))

    def test_permanent_error_after_transient_error(self):
        result, attempts, delays = self.run_uv(
            [(1, "network timeout"), (2, "No solution found")]
        )
        self.assertEqual((result.returncode, attempts, delays), (2, 2, ["5"]))

    def test_success_on_final_attempt(self):
        result, attempts, delays = self.run_uv(
            [(1, "network timeout"), (1, "network timeout"), (0, "installed")]
        )
        self.assertEqual((result.returncode, attempts, delays), (0, 3, ["5", "10"]))

    def test_retry_limit_preserves_failure(self):
        result, attempts, delays = self.run_uv([(7, "network timeout")] * 3)
        self.assertEqual((result.returncode, attempts, delays), (7, 3, ["5", "10"]))
        self.assertIn("Max retry reached", result.stdout)
        self.assertEqual(result.stdout.count("Retrying uv"), 2)

    def test_waits_for_stderr(self):
        result, attempts, delays = self.run_uv(
            [(1, "HTTP status server error (502 Bad Gateway)"), (0, "installed")],
            delayed_tee=True,
        )
        self.assertEqual((result.returncode, attempts, delays), (0, 2, ["5"]))


if __name__ == "__main__":
    unittest.main()
