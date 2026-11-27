# SPDX-License-Identifier: LGPL-3.0-or-later
"""Launcher of the workers of the data-parallel tests.

A spawned process imports the module of its target before the target runs,
and DeePMD-kit fixes the device of a rank from ``LOCAL_RANK`` when it is
imported. This module imports nothing of DeePMD-kit, so that a worker starts
with the rank of its process and every rank computes on a device of its own.
"""

import importlib
import os
from typing import (
    Any,
)


def launch_worker(rank: int, worker: str, *args: Any) -> None:
    """
    Run a worker of the data-parallel tests on the device of its rank.

    Parameters
    ----------
    rank : int
        Index of the spawned process, which is the rank of the worker.
    worker : str
        Name of the worker function in ``test_training_ddp``.
    *args
        Arguments of the worker after its rank.
    """
    os.environ["LOCAL_RANK"] = str(rank)
    module = importlib.import_module(".test_training_ddp", package=__package__)
    getattr(module, worker)(rank, *args)
