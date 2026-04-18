# SPDX-License-Identifier: LGPL-3.0-or-later
"""Helpers for preparing converted TensorFlow graph files in LAMMPS tests."""

from __future__ import (
    annotations,
)

import os
import subprocess as sp
import sys
import tempfile
import time
from pathlib import (
    Path,
)

_LOCK_TIMEOUT_SECONDS = 60.0
_LOCK_POLL_SECONDS = 0.1


def _is_up_to_date(source: Path, output: Path) -> bool:
    return output.exists() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns


def ensure_converted_pb(source: Path, output: Path) -> Path:
    """Convert ``source`` into ``output`` only when the target is missing or stale.

    The conversion is protected by a simple lock file and uses atomic replacement so
    repeated imports across multiple test modules do not regenerate the same model
    more than once.
    """
    source = source.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_file = output.with_name(f".{output.name}.lock")
    started = time.monotonic()

    while True:
        if _is_up_to_date(source, output):
            return output
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - started >= _LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(f"Timed out waiting for {lock_file}")
            time.sleep(_LOCK_POLL_SECONDS)
            continue
        break

    tmp_path: Path | None = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")

        if _is_up_to_date(source, output):
            return output

        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        sp.check_output(
            [
                sys.executable,
                "-m",
                "deepmd",
                "convert-from",
                "pbtxt",
                "-i",
                str(source),
                "-o",
                str(tmp_path),
            ]
        )
        tmp_path.replace(output)
        tmp_path = None
        return output
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        lock_file.unlink(missing_ok=True)
