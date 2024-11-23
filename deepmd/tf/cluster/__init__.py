# SPDX-License-Identifier: LGPL-3.0-or-later
"""Module that reads node resources, auto detects if running local or on SLURM."""

from typing import (
    Optional,
)

from .local import get_resource as get_local_res

__all__ = ["get_resource"]


def get_resource() -> tuple[str, list[str], Optional[list[int]]]:
    """Get local or slurm resources: nodename, nodelist, and gpus.

    Returns
    -------
    tuple[str, list[str], Optional[list[int]]]
        nodename, nodelist, and gpus
    """
    return get_local_res()
