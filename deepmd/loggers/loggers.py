# SPDX-License-Identifier: LGPL-3.0-or-later
"""Logger initialization for package."""

import logging
import os
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    NoReturn,
    Optional,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )

    from mpi4py import (
        MPI,
    )

    _MPI_APPEND_MODE = MPI.MODE_CREATE | MPI.MODE_APPEND

logging.getLogger(__name__)

__all__ = ["WorkerLogConfig", "is_node_main_process", "set_log_handles"]

# logger formatter
FFORMATTER = logging.Formatter(
    "[%(asctime)s] %(app_name)s %(levelname)-7s %(name)-45s %(message)s"
)
CFORMATTER = logging.Formatter(
    #    "%(app_name)s %(levelname)-7s |-> %(name)-45s %(message)s"
    "[%(asctime)s] %(app_name)s %(levelname)-7s %(message)s"
)
FFORMATTER_MPI = logging.Formatter(
    "[%(asctime)s] %(app_name)s rank:%(rank)-2s %(levelname)-7s %(name)-45s %(message)s"
)
CFORMATTER_MPI = logging.Formatter(
    #    "%(app_name)s rank:%(rank)-2s %(levelname)-7s |-> %(name)-45s %(message)s"
    "[%(asctime)s] %(app_name)s rank:%(rank)-2s %(levelname)-7s %(message)s"
)
CFORMATTER_DISTRIBUTED = logging.Formatter(
    "[%(asctime)s] %(app_name)s %(levelname)-7s %(rank_prefix)s%(message)s"
)


def _rank_scope(record: logging.LogRecord) -> str:
    """Resolve a record's scope while retaining errors from every rank."""
    if record.levelno >= logging.ERROR:
        return "all"
    scope = getattr(record, "rank_scope", None)
    if scope is None:
        scope = "node" if logging.INFO <= record.levelno < logging.WARNING else "all"
    return scope


def is_node_main_process(rank: int | None = None) -> bool:
    """Return whether this rank reports node-level summaries.

    Parameters
    ----------
    rank : int or None, optional
        Global rank supplied by the caller. Launchers without ``LOCAL_RANK``
        retain global-chief reporting instead of guessing the node layout.
        If omitted, ``RANK`` is used, defaulting to zero outside a launcher.

    Returns
    -------
    bool
        Whether the local rank, or the supplied global rank, is zero.

    Notes
    -----
    This predicate only controls reporting. Checkpoint ownership and
    participation in distributed operations depend on the global rank.
    Configured rank metadata takes precedence over the environment so spawned
    workers use the context captured from their owning training process.
    """
    for handler in logging.getLogger("deepmd").handlers:
        for log_filter in handler.filters:
            if isinstance(log_filter, _DistributedLogFilter):
                context = log_filter.context
                local_rank = context.local_rank
                return (context.rank if local_rank is None else local_rank) == 0
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    return int(os.environ.get("LOCAL_RANK", rank)) == 0


@dataclass(frozen=True)
class _DistributedLogContext:
    """Launcher-provided process identity, available before communication starts."""

    rank: int
    local_rank: int | None

    @classmethod
    def from_environment(cls) -> "_DistributedLogContext | None":
        """Read torchrun-compatible rank metadata without importing a backend."""
        rank = os.environ.get("RANK")
        if rank is None:
            return None
        local_rank = os.environ.get("LOCAL_RANK")
        return cls(
            rank=int(rank),
            local_rank=int(local_rank) if local_rank is not None else None,
        )

    def allows(self, record: logging.LogRecord) -> bool:
        """Select console records while retaining every process's errors."""
        scope = _rank_scope(record)
        if scope == "all":
            return True
        if scope == "global":
            return self.rank == 0
        if scope == "node":
            rank = self.rank if self.local_rank is None else self.local_rank
            return rank == 0
        raise ValueError(
            f"Unknown logging rank_scope {scope!r}; expected 'node', 'global', or 'all'."
        )


class _DistributedLogFilter(logging.Filter):
    """Label per-rank events and optionally select console records by scope."""

    def __init__(self, context: _DistributedLogContext, *, filter_ranks: bool) -> None:
        super().__init__()
        self.context = context
        self.filter_ranks = filter_ranks

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.context.rank
        record.rank_prefix = (
            f"[rank={self.context.rank}] " if _rank_scope(record) == "all" else ""
        )
        return not self.filter_ranks or self.context.allows(record)


class _AppFilter(logging.Filter):
    """Add field `app_name` to log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.app_name = "DEEPMD"
        return True


class _MPIRankFilter(logging.Filter):
    """Add MPI rank number to log messages, adds field `rank`."""

    def __init__(self, rank: int) -> None:
        super().__init__(name="MPI_rank_id")
        self.mpi_rank = str(rank)

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.mpi_rank
        return True


class _MPIMasterFilter(logging.Filter):
    """Filter that lets through only messages emitted from rank==0."""

    def __init__(self, rank: int) -> None:
        super().__init__(name="MPI_master_log")
        self.mpi_rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
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

    def write(self, msg: str) -> None:
        """Write to MPI shared file stream.

        Parameters
        ----------
        msg : str
            message to write
        """
        b = bytearray()
        b.extend(map(ord, msg))
        self.stream.Write_shared(b)

    def close(self) -> None:
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

    def _open(self) -> "_MPIFileStream":
        return _MPIFileStream(self.baseFilename, self.MPI, self.mode)

    def setStream(self, stream: "_MPIFileStream") -> NoReturn:
        """Stream cannot be reasigned in MPI mode."""
        raise NotImplementedError("Unable to do for MPI file handler!")


def _replace_handlers(level: int, handlers: list[logging.Handler]) -> None:
    """Install local handlers without closing collective MPI file handles."""
    root_log = logging.getLogger("deepmd")
    root_log.propagate = False
    root_log.setLevel(level)
    for handler in root_log.handlers[:]:
        root_log.removeHandler(handler)
        # MPI file handles have collective lifetimes. Reconfiguration replaces
        # their local registration without entering an MPI close operation.
        if not isinstance(handler, _MPIHandler):
            handler.close()
    for handler in handlers:
        root_log.addHandler(handler)


@dataclass(frozen=True)
class _LogHandlerConfig:
    """Serializable handler settings without live streams or locks."""

    level: int
    formatter: logging.Formatter | None
    filters: tuple[logging.Filter, ...]
    filename: str | None

    def create_handler(self) -> logging.Handler:
        handler = (
            logging.FileHandler(self.filename, mode="a")
            if self.filename is not None
            else logging.StreamHandler()
        )
        handler.setLevel(self.level)
        handler.setFormatter(self.formatter)
        for log_filter in self.filters:
            handler.addFilter(log_filter)
        return handler


@dataclass(frozen=True)
class WorkerLogConfig:
    """Serializable DeePMD logging configuration for process-pool workers.

    Workers retain the parent's console selection, process-rank context,
    and ordinary file destinations. File streams reopen in append mode.
    MPI-IO handlers stay owned by the parent ranks because an independent
    worker cannot participate in their collective file operations.
    """

    level: int
    handlers: tuple[_LogHandlerConfig, ...]

    @classmethod
    def capture(cls) -> "WorkerLogConfig":
        """Capture configured handlers, or the effective level before CLI setup."""
        root_log = logging.getLogger("deepmd")
        return cls(
            level=root_log.getEffectiveLevel(),
            handlers=tuple(
                _LogHandlerConfig(
                    level=handler.level,
                    formatter=handler.formatter,
                    filters=tuple(handler.filters),
                    filename=(
                        handler.baseFilename
                        if isinstance(handler, logging.FileHandler)
                        else None
                    ),
                )
                for handler in root_log.handlers
                if not isinstance(handler, _MPIHandler)
            ),
        )

    def configure(self) -> None:
        """Install the parent's local logging policy in a worker process."""
        if not self.handlers:
            set_log_handles(self.level)
            return
        _replace_handlers(
            self.level, [handler.create_handler() for handler in self.handlers]
        )


def set_log_handles(
    level: int, log_path: Optional["Path"] = None, mpi_log: str | None = None
) -> None:
    """Set desired level for package loggers and add file handlers.

    Parameters
    ----------
    level : int
        logging level
    log_path : Optional[str]
        path to log file, if None logs will be send only to console. If the parent
        directory does not exist it will be automatically created, by default None.
        Under a torchrun-compatible launcher, each rank writes its emitted records
        to ``<stem>.rank<rank><suffix>`` at the requested logging level.
    mpi_log : Optional[str], optional
        mpi log type. Has three options. `master` will output logs to file and console
        only from rank==0. `collect` will write messages from all ranks to one file
        opened under rank==0 and to console. `workers` will open one log file for each
        worker designated by its rank, console behaviour is the same as for `collect`.
        If this argument is specified, package 'mpi4py' must be already installed.
        by default None

    Raises
    ------
    RuntimeError
        If the argument `mpi_log` is specified, package `mpi4py` is not installed.

    References
    ----------
    https://groups.google.com/g/mpi4py/c/SaNzc8bdj6U
    https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
    https://stackoverflow.com/questions/56085015/suppress-openmp-debug-messages-when-running-tensorflow-on-cpu

    Notes
    -----
    When ``RANK`` is present and MPI logging is not requested, console INFO
    records default to ``LOCAL_RANK == 0``. DEBUG and WARNING records default
    to every rank. ``extra={"rank_scope": "node" | "global" | "all"}``
    overrides this selection; ERROR records and higher always pass. Without
    ``LOCAL_RANK``, node scope falls back to global rank zero. File handlers
    retain all emitted records at the selected level. Native stderr output
    and Python warnings are not redirected or suppressed.

    Console records emitted by every rank carry a compact ``[rank=N]`` label.
    Node and global summaries keep the single-process format. Per-rank files
    identify their owner by filename and keep the standard file format.

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
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(int((level / 10) - 1))

    # check if arguments are present
    MPI = None
    if mpi_log:
        try:
            from mpi4py import (
                MPI,
            )
        except ImportError as e:
            raise RuntimeError(
                "You cannot specify 'mpi_log' when mpi4py not installed"
            ) from e

    context = None if MPI else _DistributedLogContext.from_environment()

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
    elif context is not None:
        ch.setFormatter(CFORMATTER_DISTRIBUTED)
        ch.addFilter(_DistributedLogFilter(context, filter_ranks=True))
    else:
        ch.setFormatter(CFORMATTER)

    ch.setLevel(level)
    ch.addFilter(_AppFilter())
    handlers: list[logging.Handler] = [ch]

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
            # if file has suffix than insert rank number before suffix
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
        elif context is not None:
            rank_log = log_path.with_name(
                f"{log_path.stem}.rank{context.rank}{log_path.suffix}"
            )
            # Worker configuration reopens this path in append mode as well.
            fh = logging.FileHandler(rank_log, mode="a")
            fh.addFilter(_DistributedLogFilter(context, filter_ranks=False))
            fh.setFormatter(FFORMATTER)
        else:
            fh = logging.FileHandler(log_path, mode="w")
            fh.setFormatter(FFORMATTER)

        if fh:
            fh.setLevel(level)
            fh.addFilter(_AppFilter())
            handlers.append(fh)

    _replace_handlers(level, handlers)
