# SPDX-License-Identifier: LGPL-3.0-or-later
"""Redirect to ``dp dpa`` — this CLI is superseded."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    print(
        "deepmd-property-tools is deprecated.\n"
        "Use 'dp dpa fit' for training and 'dp dpa predict' for inference.\n"
        "Use 'dp dpa data convert-smiles' for CSV+SMILES to deepmd/npy conversion.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
