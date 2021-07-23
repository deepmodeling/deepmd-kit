"""Module that reads node resources, auto detects if running local or on SLURM."""

from .local import get_resource as get_local_res
from .slurm import get_resource as get_slurm_res
import os
from typing import List, Tuple, Optional

__all__ = ["get_resource"]


def get_resource() -> Tuple[str, List[str], Optional[List[int]]]:
    """Get local or slurm resources: nodename, nodelist, and gpus.

    Returns
    -------
    Tuple[str, List[str], Optional[List[int]]]
        nodename, nodelist, and gpus
    """
    if "SLURM_JOB_NODELIST" in os.environ:
        return get_slurm_res()
    else:
        return get_local_res()
