"""Get local GPU resources."""
import logging
import os
import socket

import GPUtil

from deepmd.env import tf
from typing import List, Tuple, Optional

__all__ = ["get_gpus", "get_resource"]

log = logging.getLogger(__name__)


def get_gpus():
    """Get available IDs of GPU cards at local.
    These IDs are valid when used as the TensorFlow device ID.

    Returns:
    -------
    Optional[List[int]]
        List of available GPU IDs. Otherwise, None.
    """
    # TODO: Create a pull request of `GPUtil` to cover ROCM devices.
    # Currently, even if None is returned, a ROCM device is still visible in TensorFlow.
    available = GPUtil.getGPUs()
    num_gpus = len(available)
    if num_gpus == 0:
        return None

    # Print help messages
    gpu_str_list = ["- %d#%s" % (item.id, item.name) for item in available]
    log.info("Availalbe GPUs are:\n%s", "\n".join(gpu_str_list))

    # Ensure TensorFlow is compatible
    if num_gpus > 0 and not tf.test.is_built_with_gpu_support():
        log.warning("GPU devices are found while your installed TensorFlow has no GPU support!"
            + " Switch to CPU device for calculation.")
        return None

    # Warn for better GPU visibility
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if num_gpus > 1:
            log.warning("Multiple GPU devices are found while only the first one will be used!"
            + " It is recommended to limit GPU visibility by the environment variable"
            + " `CUDA_VISIBLE_DEVICES`.")
        return list(range(num_gpus))

    # In case where user set "CUDA_VISIBLE_DEVICES=-1" to disable GPU usage
    valid_ids = []
    for item in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
        idx = int(item)
        if idx >= 0 and idx < num_gpus:
            gpu_id = len(valid_ids)
            valid_ids.append(gpu_id)
    return valid_ids if len(valid_ids) > 0 else None  # Always None if no GPU available


def get_resource() -> Tuple[str, List[str], Optional[List[int]]]:
    """Get local resources: nodename, nodelist, and gpus.

    Returns
    -------
    Tuple[str, List[str], Optional[List[int]]]
        nodename, nodelist, and gpus
    """
    nodename = socket.gethostname()
    nodelist = [nodename]
    gpus = get_gpus()
    return nodename, nodelist, gpus
