# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)


def format_training_message(
    batch: int,
    wall_time: float,
):
    """Format a training message."""
    return f"batch {batch:7d}: " f"total wall time = {wall_time:.2f} s"


def format_training_message_per_task(
    batch: int,
    task_name: str,
    rmse: Dict[str, float],
    learning_rate: Optional[float],
):
    if task_name:
        task_name += ": "
    if learning_rate is None:
        lr = ""
    else:
        lr = f", lr = {learning_rate:8.2e}"
    # sort rmse
    rmse = dict(sorted(rmse.items()))
    return (
        f"batch {batch:7d}: {task_name}"
        f"{', '.join([f'{kk} = {vv:8.2e}' for kk, vv in rmse.items()])}"
        f"{lr}"
    )
