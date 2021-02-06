"""Get local GPU resources from `CUDA_VISIBLE_DEVICES` enviroment variable."""

import os
import socket
from typing import List, Tuple, Optional


def get_resource() -> Tuple[str, List[str], Optional[List[int]]]:
    """Get loacl resources: nodename, nodelist, and gpus.

    Returns
    -------
    Tuple[str, List[str], Optional[List[int]]]
        nodename, nodelist, and gpus
    """
    nodename = socket.gethostname()
    nodelist = [nodename]
    gpus_env = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if gpus_env is None:
        gpus = None
    elif gpus_env == "":
        gpus = []
    else:
        gpus = [int(gpu) for gpu in gpus_env.split(",")]

    return nodename, nodelist, gpus
