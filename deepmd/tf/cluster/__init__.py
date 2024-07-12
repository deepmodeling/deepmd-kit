# SPDX-License-Identifier: LGPL-3.0-or-later
"""Module that reads node resources, auto detects if running local or on SLURM."""

from typing import (
    List,
    Optional,
    Tuple,
)

from .local import get_resource as get_local_res

__all__ = ["get_resource"]


def get_resource() -> Tuple[str, List[str], Optional[List[int]]]:
    """Get local or slurm resources: nodename, nodelist, and gpus.

    Returns
    -------
    Tuple[str, List[str], Optional[List[int]]]
        nodename, nodelist, and gpus
    """
    return get_local_res()
