# SPDX-License-Identifier: LGPL-3.0-or-later
import datetime
import logging
import math

log = logging.getLogger(__name__)


def format_training_message(
    batch: int,
    wall_time: float,
    eta: int | None = None,
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
    learning_rate: float | None,
    check_total_rmse_nan: bool = True,
) -> str:
    """Format training messages for a specific task.

    Parameters
    ----------
    batch : int
        The batch index
    task_name : str
        The task name
    rmse : dict[str, float]
        The root-mean-squared errors.
    learning_rate : float | None
        The learning rate
    check_total_rmse_nan : bool
        Whether to throw an error if the total RMSE is NaN
    """
    if task_name:
        task_name += ": "
    if learning_rate is None:
        lr = ""
    else:
        lr = f", lr = {learning_rate:8.2e}"
    # sort rmse
    rmse = dict(sorted(rmse.items()))
    msg = (
        f"batch {batch:7d}: {task_name}"
        f"{', '.join([f'{kk} = {vv:8.2e}' for kk, vv in rmse.items()])}"
        f"{lr}"
    )
    if check_total_rmse_nan and math.isnan(rmse.get("rmse", 0.0)):
        log.error(msg)
        err_msg = (
            f"NaN detected at batch {batch:7d}: {task_name}. "
            "Something went wrong, and it is meaningless to continue."
        )
        raise RuntimeError(err_msg)
    return msg
