# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get local GPU resources."""

import subprocess as sp
import sys
from typing import (
    List,
    Optional,
    Tuple,
)

from deepmd.tf.env import (
    tf,
)
from deepmd.utils.hostlist import (
    get_host_names,
)

__all__ = ["get_gpus", "get_resource"]


def get_gpus():
    """Get available IDs of GPU cards at local.
    These IDs are valid when used as the TensorFlow device ID.

    Returns
    -------
    Optional[List[int]]
        List of available GPU IDs. Otherwise, None.
    """
    if not tf.test.is_built_with_cuda() and not (
        hasattr(tf.test, "is_built_with_rocm") and tf.test.is_built_with_rocm()
    ):
        # TF is built with CPU only, skip expensive subprocess call
        return None
    test_cmd = (
        "from tensorflow.python.client import device_lib; "
        "devices = device_lib.list_local_devices(); "
        'gpus = [d.name for d in devices if d.device_type == "GPU"]; '
        "print(len(gpus))"
    )
    with sp.Popen(
        [sys.executable, "-c", test_cmd], stderr=sp.PIPE, stdout=sp.PIPE
    ) as p:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            decoded = stderr.decode("UTF-8")
            raise RuntimeError("Failed to detect availbe GPUs due to:\n%s" % decoded)
        decoded = stdout.decode("UTF-8").strip()
        num_gpus = int(decoded)
        return list(range(num_gpus)) if num_gpus > 0 else None


def get_resource() -> Tuple[str, List[str], Optional[List[int]]]:
    """Get local resources: nodename, nodelist, and gpus.

    Returns
    -------
    Tuple[str, List[str], Optional[List[int]]]
        nodename, nodelist, and gpus
    """
    nodename, nodelist = get_host_names()
    gpus = get_gpus()
    return nodename, nodelist, gpus
