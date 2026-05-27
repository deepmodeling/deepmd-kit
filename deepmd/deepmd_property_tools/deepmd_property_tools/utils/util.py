# SPDX-License-Identifier: LGPL-3.0-or-later
"""General utilities."""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
