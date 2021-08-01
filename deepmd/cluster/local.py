"""Get local GPU resources."""

import os
import socket
import subprocess as sp
import sys

from deepmd.env import tf
from typing import List, Tuple, Optional


__all__ = ["get_gpus", "get_resource"]


def get_gpus():
    """Get available IDs of GPU cards at local.
    These IDs are valid when used as the TensorFlow device ID.

    Returns:
    -------
    Optional[List[int]]
        List of available GPU IDs. Otherwise, None.
    """
    test_cmd = 'from tensorflow.python.client import device_lib; ' \
               'devices = device_lib.list_local_devices(); ' \
               'gpus = [d.name for d in devices if d.device_type == "GPU"]; ' \
               'print(len(gpus))'
    with sp.Popen([sys.executable, "-c", test_cmd], stderr=sp.PIPE, stdout=sp.PIPE) as p:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            decoded = stderr.decode('UTF-8')
            raise RuntimeError('Failed to detect availbe GPUs due to:\n%s' % decoded)
        decoded = stdout.decode('UTF-8').strip()
        num_gpus = int(decoded)
        return list(range(num_gpus)) if num_gpus > 0 else None


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
