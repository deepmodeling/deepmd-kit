"""Logger initialization for package."""

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pathlib import Path

    from mpi4py import MPI

    _MPI_APPEND_MODE = MPI.MODE_CREATE | MPI.MODE_APPEND

logging.getLogger(__name__)

__all__ = ["set_log_handles"]

# logger formater
FFORMATTER = logging.Formatter(
    "[%(asctime)s] %(app_name)s %(levelname)-7s %(name)-45s %(message)s"
)
CFORMATTER = logging.Formatter(
#    "%(app_name)s %(levelname)-7s |-> %(name)-45s %(message)s"
    "%(app_name)s %(levelname)-7s %(message)s"
)
FFORMATTER_MPI = logging.Formatter(
    "[%(asctime)s] %(app_name)s rank:%(rank)-2s %(levelname)-7s %(name)-45s %(message)s"
)
CFORMATTER_MPI = logging.Formatter(
#    "%(app_name)s rank:%(rank)-2s %(levelname)-7s |-> %(name)-45s %(message)s"
    "%(app_name)s rank:%(rank)-2s %(levelname)-7s %(message)s"
)


class _AppFilter(logging.Filter):
    """Add field `app_name` to log messages."""

    def filter(self, record):
        record.app_name = "DEEPMD"
        return True


class _MPIRankFilter(logging.Filter):
    """Add MPI rank number to log messages, adds field `rank`."""

    def __init__(self, rank: int) -> None:
        super().__init__(name="MPI_rank_id")
        self.mpi_rank = str(rank)

    def filter(self, record):
        record.rank = self.mpi_rank
        return True


class _MPIMasterFilter(logging.Filter):
    """Filter that lets through only messages emited from rank==0."""

    def __init__(self, rank: int) -> None:
        super().__init__(name="MPI_master_log")
        self.mpi_rank = rank

    def filter(self, record):
        if self.mpi_rank == 0:
            return True
        else:
            return False


class _MPIFileStream:
    """Wrap MPI.File` so it has the same API as python file streams.

    Parameters
    ----------
    filename : Path
        disk location of the file stream
    MPI : MPI
        MPI communicator object
    mode : str, optional
        file write mode, by default _MPI_APPEND_MODE
    """

    def __init__(
        self, filename: "Path", MPI: "MPI", mode: str = "_MPI_APPEND_MODE"
    ) -> None:
        self.stream = MPI.File.Open(MPI.COMM_WORLD, filename, mode)
        self.stream.Set_atomicity(True)
        self.name = "MPIfilestream"

    def write(self, msg: str):
        """Write to MPI shared file stream.

        Parameters
        ----------
        msg : str
            message to write
        """
        b = bytearray()
        b.extend(map(ord, msg))
        self.stream.Write_shared(b)

    def close(self):
        """Synchronize and close MPI file stream."""
        self.stream.Sync()
        self.stream.Close()


class _MPIHandler(logging.FileHandler):
    """Emulate `logging.FileHandler` with MPI shared File that all ranks can write to.

    Parameters
    ----------
    filename : Path
        file path
    MPI : MPI
        MPI communicator object
    mode : str, optional
        file access mode, by default "_MPI_APPEND_MODE"
    """

    def __init__(
        self,
        filename: "Path",
        MPI: "MPI",
        mode: str = "_MPI_APPEND_MODE",
    ) -> None:
        self.MPI = MPI
        super().__init__(filename, mode=mode, encoding=None, delay=False)

    def _open(self):
        return _MPIFileStream(self.baseFilename, self.MPI, self.mode)

    def setStream(self, stream):
        """Stream canot be reasigned in MPI mode."""
        raise NotImplementedError("Unable to do for MPI file handler!")


def set_log_handles(
    level: int,
    log_path: Optional["Path"] = None,
    mpi_log: Optional[str] = None,
    MPI: Optional["MPI"] = None,
):
    """Set desired level for package loggers and add file handlers.

    Parameters
    ----------
    level: int
        logging level
    log_path: Optional[str]
        path to log file, if None logs will be send only to console. If the parent
        directory does not exist it will be automatically created, by default None
    mpi_log : Optional[str], optional
        mpi log type. Has three options. `master` will output logs to file and console
        only from rank==0. `collect` will write messages from all ranks to one file
        opened under rank==0 and to console. `workers` will open one log file for each
        worker designated by its rank, console behaviour is the same as for `collect`.
        If this argument is specified than also `MPI` object must be passed in.
        by default None
    MPI : Optional[MPI, optional]
        `MPI` communicator object, must be specified if `mpi_log` is specified,
        by default None

    Raises
    ------
    RuntimeError
        if only one of the arguments `mpi_log`, `MPI` is specified

    References
    ----------
    https://groups.google.com/g/mpi4py/c/SaNzc8bdj6U
    https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
    https://stackoverflow.com/questions/56085015/suppress-openmp-debug-messages-when-running-tensorflow-on-cpu

    Notes
    -----
    Logging levels:

    +---------+--------------+----------------+----------------+----------------+
    |         | our notation | python logging | tensorflow cpp | OpenMP         |
    +=========+==============+================+================+================+
    | debug   | 10           | 10             | 0              | 1/on/true/yes  |
    +---------+--------------+----------------+----------------+----------------+
    | info    | 20           | 20             | 1              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+
    | warning | 30           | 30             | 2              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+
    | error   | 40           | 40             | 3              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+

    """
    # silence logging for OpenMP when running on CPU if level is any other than debug
    if level <= 10:
        os.environ["KMP_WARNINGS"] = "FALSE"

    # set TF cpp internal logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(int((level / 10) - 1))

    # get root logger
    root_log = logging.getLogger()

    # remove all old handlers
    root_log.setLevel(level)
    for hdlr in root_log.handlers[:]:
        root_log.removeHandler(hdlr)

    # check if arguments are present
    if (mpi_log and not MPI) or (not mpi_log and MPI):
        raise RuntimeError("You cannot specify only one of 'mpi_log', 'MPI' arguments")

    # * add console handler ************************************************************
    ch = logging.StreamHandler()
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        if mpi_log == "master":
            ch.setFormatter(CFORMATTER)
            ch.addFilter(_MPIMasterFilter(rank))
        else:
            ch.setFormatter(CFORMATTER_MPI)
            ch.addFilter(_MPIRankFilter(rank))
    else:
        ch.setFormatter(CFORMATTER)

    ch.setLevel(level)
    ch.addFilter(_AppFilter())
    root_log.addHandler(ch)

    # * add file handler ***************************************************************
    if log_path:

        # create directory
        log_path.parent.mkdir(exist_ok=True, parents=True)

        fh = None

        if mpi_log == "master":
            rank = MPI.COMM_WORLD.Get_rank()
            if rank == 0:
                fh = logging.FileHandler(log_path, mode="w")
                fh.addFilter(_MPIMasterFilter(rank))
                fh.setFormatter(FFORMATTER)
        elif mpi_log == "collect":
            rank = MPI.COMM_WORLD.Get_rank()
            fh = _MPIHandler(log_path, MPI, mode=MPI.MODE_WRONLY | MPI.MODE_CREATE)
            fh.addFilter(_MPIRankFilter(rank))
            fh.setFormatter(FFORMATTER_MPI)
        elif mpi_log == "workers":
            rank = MPI.COMM_WORLD.Get_rank()
            # if file has suffix than inser rank number before suffix
            # e.g deepmd.log -> deepmd_<rank>.log
            # if no suffix is present, insert rank as suffix
            # e.g. deepmdlog -> deepmdlog.<rank>
            if log_path.suffix:
                worker_log = (log_path.parent / f"{log_path.stem}_{rank}").with_suffix(
                    log_path.suffix
                )
            else:
                worker_log = log_path.with_suffix(f".{rank}")

            fh = logging.FileHandler(worker_log, mode="w")
            fh.setFormatter(FFORMATTER)
        else:
            fh = logging.FileHandler(log_path, mode="w")
            fh.setFormatter(FFORMATTER)

        if fh:
            fh.setLevel(level)
            fh.addFilter(_AppFilter())
            root_log.addHandler(fh)
