# SPDX-License-Identifier: LGPL-3.0-or-later
"""Logger initialization for package."""

import logging
import os
from dataclasses import (
    dataclass,
    replace,
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
    """DeePMD handler policy without formatter, filter, or stream objects."""

    level: int
    filename: str | None = None
    context: _DistributedLogContext | None = None
    mpi_rank: int | None = None
    mpi_master: bool = False

    def create_handler(self, *, mode: str = "a") -> logging.Handler:
        """Construct a local handler from its rank and destination policy."""
        handler = (
            logging.FileHandler(self.filename, mode=mode)
            if self.filename is not None
            else logging.StreamHandler()
        )
        console = self.filename is None
        formatter = CFORMATTER if console else FFORMATTER
        if self.context is not None:
            handler.addFilter(_DistributedLogFilter(self.context, filter_ranks=console))
            if console:
                formatter = CFORMATTER_DISTRIBUTED
        elif self.mpi_rank is not None:
            if self.mpi_master:
                handler.addFilter(_MPIMasterFilter(self.mpi_rank))
            elif console:
                handler.addFilter(_MPIRankFilter(self.mpi_rank))
                formatter = CFORMATTER_MPI
        handler.setLevel(self.level)
        handler.setFormatter(formatter)
        handler.addFilter(_AppFilter())
        # Runtime logging integrations may attach non-serializable objects.
        handler._deepmd_log_config = self
        return handler


@dataclass(frozen=True)
class WorkerLogConfig:
    """Serializable DeePMD logging configuration for process-pool workers.

    Workers retain the parent's console selection, process-rank context,
    and DeePMD file destinations. File streams reopen in append mode. External
    logging integrations remain in the parent process, as do MPI-IO handlers
    whose collective file operations cannot involve an independent worker.
    """

    level: int
    handlers: tuple[_LogHandlerConfig, ...]

    @classmethod
    def capture(cls) -> "WorkerLogConfig":
        """Capture configured handlers, or the effective level before CLI setup."""
        root_log = logging.getLogger("deepmd")
        configs = []
        for handler in root_log.handlers:
            config = getattr(handler, "_deepmd_log_config", None)
            if isinstance(config, _LogHandlerConfig):
                configs.append(replace(config, level=handler.level))
        return cls(
            level=root_log.getEffectiveLevel(),
            handlers=tuple(configs),
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
    rank = MPI.COMM_WORLD.Get_rank() if MPI else None
    config = _LogHandlerConfig(
        level=level,
        context=context,
        mpi_rank=rank,
        mpi_master=mpi_log == "master",
    )
    handlers = [config.create_handler()]

    # * add file handler ***************************************************************
    if log_path:
        # create directory
        log_path.parent.mkdir(exist_ok=True, parents=True)

        if mpi_log == "collect":
            fh = _MPIHandler(log_path, MPI, mode=MPI.MODE_WRONLY | MPI.MODE_CREATE)
            fh.addFilter(_MPIRankFilter(rank))
            fh.setFormatter(FFORMATTER_MPI)
            fh.setLevel(level)
            fh.addFilter(_AppFilter())
            handlers.append(fh)
        elif mpi_log != "master" or rank == 0:
            if mpi_log == "workers":
                # MPI worker paths retain their existing suffix convention.
                # deepmd.log becomes deepmd_<rank>.log; deepmdlog gains .<rank>.
                log_path = (
                    (log_path.parent / f"{log_path.stem}_{rank}").with_suffix(
                        log_path.suffix
                    )
                    if log_path.suffix
                    else log_path.with_suffix(f".{rank}")
                )
            elif context is not None:
                log_path = log_path.with_name(
                    f"{log_path.stem}.rank{context.rank}{log_path.suffix}"
                )
            file_config = replace(config, filename=str(log_path.absolute()))
            handlers.append(
                file_config.create_handler(mode="a" if context is not None else "w")
            )

    _replace_handlers(level, handlers)
