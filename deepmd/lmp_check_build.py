# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.env import (
    GLOBAL_CONFIG,
)

if GLOBAL_CONFIG.get("lammps_version", "") == "":

    def get_op_dir() -> str:
        """Get the directory of the deepmd-kit OP library."""
        # empty
        return ""
else:
    from deepmd.lmp import (
        get_op_dir,
    )

__all__ = [
    "get_op_dir",
]
