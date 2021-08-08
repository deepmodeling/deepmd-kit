"""MOdule to get resources on SLURM cluster.

References
----------
https://github.com/deepsense-ai/tensorflow_on_slurm ####
"""

import hostlist
import os

from deepmd.cluster import local
from typing import List, Tuple, Optional

__all__ = ["get_resource"]


def get_resource() -> Tuple[str, List[str], Optional[List[int]]]:
    """Get SLURM resources: nodename, nodelist, and gpus.

    Returns
    -------
    Tuple[str, List[str], Optional[List[int]]]
        nodename, nodelist, and gpus

    Raises
    ------
    RuntimeError
        if number of nodes could not be retrieved
    ValueError
        list of nodes is not of the same length sa number of nodes
    ValueError
        if current nodename is not found in node list
    """
    nodelist = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
    nodename = os.environ["SLURMD_NODENAME"]
    num_nodes_env = os.getenv("SLURM_JOB_NUM_NODES")
    if num_nodes_env:
        num_nodes = int(num_nodes_env)
    else:
        raise RuntimeError("Could not get SLURM number of nodes")

    if len(nodelist) != num_nodes:
        raise ValueError(
            f"Number of slurm nodes {len(nodelist)} not equal to {num_nodes}"
        )
    if nodename not in nodelist:
        raise ValueError(
            f"Nodename({nodename}) not in nodelist({nodelist}). This should not happen!"
        )
    gpus = local.get_gpus()
    return nodename, nodelist, gpus
