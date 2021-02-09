
"""Logger initialization for package."""

import logging
from typing import TYPE_CHECKING
from pathlib import Path


if TYPE_CHECKING:
    from pathlib import Path

logging.getLogger(__name__)

__all__ = ["set_log_handles"]

# logger formater
FFORMATTER = logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)-45s "
                               "%(message)s")
CFORMATTER = logging.Formatter("%(levelname)-7s |-> %(name)-45s %(message)s")


def set_log_handles(level: int, log_path: "Path"):
    """Set desired level for package loggers and add file handlers.

    Parameters
    ----------
    level: int
        logging level
    log_path: Path
        path to log file
    """
    root_log = logging.getLogger()

    # remove all old handlers
    root_log.setLevel(level)
    for hdlr in root_log.handlers[:]:
        root_log.removeHandler(hdlr)

    # add console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CFORMATTER)
    root_log.addHandler(ch)

    # add file handler
    ch = logging.FileHandler(log_path, mode="w")
    ch.setLevel(level)
    ch.setFormatter(FFORMATTER)
    root_log.addHandler(ch)

