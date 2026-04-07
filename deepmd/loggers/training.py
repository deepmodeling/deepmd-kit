# SPDX-License-Identifier: LGPL-3.0-or-later
import datetime
import logging
import math

log = logging.getLogger(__name__)


def _format_estimated_finish_time(
    eta_seconds: int,
    current_time: datetime.datetime | None = None,
) -> str:
    """Format the estimated local finish time.

    Parameters
    ----------
    eta_seconds : int
        Remaining time in seconds.
    current_time : datetime.datetime | None, optional
        Current local time used to estimate the finish timestamp. If ``None``,
        the current local time is used.

    Returns
    -------
    str
        Estimated local finish time in ``YYYY-MM-DD HH:MM`` format.
    """
    if current_time is None:
        current_time = datetime.datetime.now(datetime.timezone.utc).astimezone()
    elif current_time.tzinfo is not None:
        current_time = current_time.astimezone()
    finish_time = current_time + datetime.timedelta(seconds=eta_seconds)
    return finish_time.strftime("%Y-%m-%d %H:%M")


def format_training_message(
    batch: int,
    wall_time: float,
    eta: int | None = None,
    current_time: datetime.datetime | None = None,
) -> str:
    """Format the summary message for one training interval.

    Parameters
    ----------
    batch : int
        The batch index.
    wall_time : float
        Wall-clock time shown in the progress message in seconds.
    eta : int | None, optional
        Remaining time in seconds.
    current_time : datetime.datetime | None, optional
        Current local time used to estimate the finish timestamp. This is only
        used when ``eta`` is provided.

    Returns
    -------
    str
        The formatted training message.
    """
    msg = f"Batch {batch:7d}: total wall time = {wall_time:.2f} s"
    if isinstance(eta, int):
        eta_seconds = int(eta)
        msg += (
            f", eta = {datetime.timedelta(seconds=eta_seconds)!s} at "
            f"{_format_estimated_finish_time(eta_seconds, current_time=current_time)}"
        )
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

    Returns
    -------
    str
        The formatted training message for the task.
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
        f"Batch {batch:7d}: {task_name}"
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
