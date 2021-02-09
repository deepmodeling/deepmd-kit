"""MOdule to get resources on SLURM cluster.

References
----------
https://github.com/deepsense-ai/tensorflow_on_slurm ####
"""

import re
import os
from typing import List, Tuple, Optional, Iterable

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
    nodelist = _expand_nodelist(os.environ["SLURM_JOB_NODELIST"])
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
    gpus_env = os.getenv("CUDA_VISIBLE_DEVICES")
    if not gpus_env:
        gpus = None
    else:
        gpus = [int(gpu) for gpu in gpus_env.split(",")]
    return nodename, nodelist, gpus


def _pad_zeros(iterable: Iterable, length: int):
    return (str(t).rjust(length, "0") for t in iterable)


def _expand_ids(ids: str) -> List[str]:
    result = []
    for _id in ids.split(","):
        if "-" in _id:
            str_end = _id.split("-")[1]
            begin, end = [int(token) for token in _id.split("-")]
            result.extend(_pad_zeros(range(begin, end + 1), len(str_end)))
        else:
            result.append(_id)
    return result


def _expand_nodelist(nodelist: str) -> List[str]:
    result = []
    interval_list = nodelist.split(",")
    for interval in interval_list:
        match = re.search(r"(.*)\[(.*)\]", interval)
        if match:
            prefix = match.group(1)
            ids = match.group(2)
            ids_list = _expand_ids(ids)
            result.extend([f"{prefix}{_id}" for _id in ids_list])
        else:
            result.append(interval)
    return result
