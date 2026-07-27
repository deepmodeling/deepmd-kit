#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inject ``pair_exclude_types: [[0, 1]]`` into a serialized model YAML.

Used by ``convert-models.sh`` to derive a model-level ``pair_exclude_types``
variant of an existing ``.yaml`` model for the DeepPotJAX pair-exclusion
ingestion-seam test. The injection is a byte-preserving text replacement of the
single ``pair_exclude_types: []`` line, so the derived model has WEIGHTS
IDENTICAL to the baseline -- the only difference is the exclusion, which is
exactly what the test compares (excluded vs baseline).

Usage
-----
    python inject_pair_exclude.py <src.yaml> <dst.yaml>
"""

import sys

_ANCHOR = "\n  pair_exclude_types: []\n"
_INJECTED = "\n  pair_exclude_types:\n  - - 0\n    - 1\n"


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as fp:
        text = fp.read()
    if _ANCHOR not in text:
        raise SystemExit(
            f"anchor '  pair_exclude_types: []' not found in {src}; "
            "cannot inject exclusion"
        )
    text = text.replace(_ANCHOR, _INJECTED, 1)
    with open(dst, "w") as fp:
        fp.write(text)


if __name__ == "__main__":
    main()
