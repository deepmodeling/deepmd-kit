# SPDX-License-Identifier: LGPL-3.0-or-later
import datetime
from typing import (
    Optional,
)


def format_training_message(
    batch: int,
    wall_time: float,
    eta: Optional[int] = None,
) -> str:
    """Format a training message."""
    msg = f"batch {batch:7d}: total wall time = {wall_time:.2f} s"
    if isinstance(eta, int):
        msg += f", eta = {datetime.timedelta(seconds=int(eta))!s}"
    return msg


def format_training_message_per_task(
    batch: int,
    task_name: str,
    rmse: dict[str, float],
    learning_rate: Optional[float],
) -> str:
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
