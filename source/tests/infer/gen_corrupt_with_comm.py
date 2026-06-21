#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate ``deeppot_*_corrupt_with_comm.pt2`` fixtures.

The fixtures are copies of the corresponding multi-rank ``.pt2`` archives
in which the nested ``model/extra/forward_lower_with_comm.pt2`` entry has
been overwritten with garbage bytes. The outer metadata still claims
``has_comm_artifact: true``, so:

- ``DeepPotPTExpt::init`` / ``DeepSpinPTExpt::init`` exercise the
  try/catch fallback path on the with-comm AOTI loader.
- Single-rank dispatch (``nswap == 0``) keeps working via the regular
  artifact.
- Multi-rank dispatch (``nswap > 0``) hits the explicit dispatch-site
  throw added in PR #5430, instead of silently dropping the MPI
  ghost-embedding exchange.

Consumed by ``source/api_cc/tests/test_with_comm_load_failure_ptexpt.cc``.
"""

import os
import zipfile

WITH_COMM_ENTRY = "model/extra/forward_lower_with_comm.pt2"
GARBAGE = b"NOT_A_VALID_AOTI_ARCHIVE_" * 32


def corrupt_with_comm(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` with the nested with-comm entry replaced."""
    with (
        zipfile.ZipFile(src, "r") as zin,
        zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zout,
    ):
        replaced = False
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == WITH_COMM_ENTRY:
                data = GARBAGE
                replaced = True
            zout.writestr(info, data)
        if not replaced:
            raise RuntimeError(
                f"{src} does not contain {WITH_COMM_ENTRY}; cannot corrupt."
            )


def main() -> None:
    base_dir = os.path.dirname(__file__)
    pairs = [
        ("deeppot_dpa3_mpi.pt2", "deeppot_dpa3_mpi_corrupt_with_comm.pt2"),
        (
            "deeppot_dpa3_spin_mpi.pt2",
            "deeppot_dpa3_spin_mpi_corrupt_with_comm.pt2",
        ),
    ]
    for src_name, dst_name in pairs:
        src = os.path.join(base_dir, src_name)
        dst = os.path.join(base_dir, dst_name)
        if not os.path.exists(src):
            print(f"Skipping {dst_name}: source {src_name} not found.")  # noqa: T201
            continue
        corrupt_with_comm(src, dst)
        print(f"Wrote {dst}")  # noqa: T201


if __name__ == "__main__":
    main()
