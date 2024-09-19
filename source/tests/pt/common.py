# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.main import (
    main,
)


def run_dp(cmd: str) -> int:
    """Run DP directly from the entry point instead of the subprocess.

    It is quite slow to start DeePMD-kit with subprocess.

    Parameters
    ----------
    cmd : str
        The command to run.

    Returns
    -------
    int
        Always returns 0.
    """
    cmds = cmd.split()
    if cmds[0] == "dp":
        cmds = cmds[1:]
    else:
        raise RuntimeError("The command is not dp")

    main(cmds)
    return 0
