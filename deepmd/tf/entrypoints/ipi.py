# SPDX-License-Identifier: LGPL-3.0-or-later
"""Use dp_ipi inside the Python package."""
import os
import subprocess
import sys
from typing import (
    List,
)

from deepmd.tf.lmp import (
    get_op_dir,
)

ROOT_DIR = get_op_dir()


def _program(name: str, args: List[str]):
    """Execuate a program.

    Parameters
    ----------
    name : str
        the name of the program
    args : list of str
        list of arguments
    """
    return subprocess.call([os.path.join(ROOT_DIR, name), *args], close_fds=False)


def dp_ipi():
    """dp_ipi."""
    suffix = ".exe" if os.name == "nt" else ""
    raise SystemExit(_program("dp_ipi" + suffix, sys.argv[1:]))
