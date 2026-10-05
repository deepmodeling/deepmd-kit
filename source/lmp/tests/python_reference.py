# SPDX-License-Identifier: LGPL-3.0-or-later
"""Isolated Python reference calculations for native LAMMPS tests."""

import json
import subprocess
import sys
import tempfile
from pathlib import (
    Path,
)
from typing import (
    Any,
)


def run_python_reference(script: str) -> Any:
    """Run a reference calculation without sharing the native operator registry.

    Parameters
    ----------
    script
        Python code assigning its JSON-serializable return value to ``result``.
        Its diagnostics may use either stdout or stderr; the result travels
        through a separate file so dependency initialization cannot corrupt it.

    Returns
    -------
    Any
        The decoded reference value.

    Raises
    ------
    RuntimeError
        If the child process fails, with both diagnostic streams included.
    """
    with tempfile.TemporaryDirectory(prefix="deepmd-python-reference-") as directory:
        output = Path(directory) / "result.json"
        program = (
            "import sys\n"
            + script
            + "\nimport json\n"
            + "with open(sys.argv[1], 'w', encoding='utf-8') as stream:\n"
            + "    json.dump(result, stream)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", program, str(output)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to compute expected values (exit {proc.returncode}):\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        return json.loads(output.read_text(encoding="utf-8"))
