# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reader for sidecar reference files written by `gen_common.write_expected_ref`.

Mirrors the C++ loader in ``source/api_cc/tests/expected_ref.h`` so that
LAMMPS Python tests and the C++ unit tests both consume the same on-disk
reference data produced by ``source/tests/infer/gen_*.py``.
"""

import numpy as np


def read_expected_ref(path):
    """Parse a sidecar reference file into ``{section: {array_name: np.ndarray}}``."""
    sections = {}
    current_section = None
    with open(path) as fp:
        lines = fp.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            sections[current_section] = {}
            continue
        if current_section is None:
            raise ValueError(f"array '{line}' before any [section] in {path}")
        key, count = line.split()
        n = int(count)
        values = [float(lines[i + j].strip()) for j in range(n)]
        i += n
        sections[current_section][key] = np.array(values, dtype=np.float64)
    return sections
