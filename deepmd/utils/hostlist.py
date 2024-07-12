# SPDX-License-Identifier: LGPL-3.0-or-later
import socket
from typing import (
    List,
    Tuple,
)


def get_host_names() -> Tuple[str, List[str]]:
    """Get host names of all nodes in the cluster.

    If mpi4py is not installed or MPI is not used, then the
    host name of the current node is returned as those of all nodes.

    Returns
    -------
    str
        Host name of the current node
    List[str]
        List of host names of all nodes in the cluster
    """
    host_name = socket.gethostname()
    try:
        from mpi4py import (
            MPI,
        )
    except ImportError:
        return host_name, [host_name]

    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        return host_name, [host_name]
    host_names = comm.allgather(host_name)
    return host_name, host_names
