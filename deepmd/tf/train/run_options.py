# SPDX-License-Identifier: LGPL-3.0-or-later
"""Module taking care of important package constants."""

import logging
import os
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
)

from packaging.version import (
    Version,
)

from deepmd.tf.cluster import (
    get_resource,
)
from deepmd.tf.env import (
    GLOBAL_CONFIG,
    TF_VERSION,
    tf,
)
from deepmd.tf.loggers import (
    set_log_handles,
)
from deepmd.utils.summary import SummaryPrinter as BaseSummaryPrinter

if TYPE_CHECKING:
    import horovod.tensorflow as HVD


__all__ = [
    "RunOptions",
]

log = logging.getLogger(__name__)


class SummaryPrinter(BaseSummaryPrinter):
    """Summary printer for TensorFlow."""

    def __init__(self, compute_device: str, ngpus: int) -> None:
        super().__init__()
        self.compute_device = compute_device
        self.ngpus = ngpus

    def is_built_with_cuda(self) -> bool:
        """Check if the backend is built with CUDA."""
        return tf.test.is_built_with_cuda()

    def is_built_with_rocm(self) -> bool:
        """Check if the backend is built with ROCm."""
        return tf.test.is_built_with_rocm()

    def get_compute_device(self) -> str:
        """Get Compute device."""
        return self.compute_device

    def get_ngpus(self) -> int:
        """Get the number of GPUs."""
        return self.ngpus

    def get_backend_info(self) -> dict:
        """Get backend information."""
        return {
            "Backend": "TensorFlow",
            "TF ver": tf.version.GIT_VERSION,
            "build with TF ver": TF_VERSION,
            "build with TF inc": GLOBAL_CONFIG["tf_include_dir"].replace(";", "\n"),
            "build with TF lib": GLOBAL_CONFIG["tf_libs"].replace(";", "\n"),
        }


class RunOptions:
    """Class with info on how to run training (cluster, MPI and GPU config).

    Attributes
    ----------
    gpus: Optional[List[int]]
        list of GPUs if any are present else None
    is_chief: bool
        in distribured training it is true for tha main MPI process in serail it is
        always true
    world_size: int
        total worker count
    my_rank: int
        index of the MPI task
    nodename: str
        name of the node
    node_list_ : List[str]
        the list of nodes of the current mpirun
    my_device: str
        deviice type - gpu or cpu
    """

    gpus: Optional[List[int]]
    world_size: int
    my_rank: int
    nodename: str
    nodelist: List[int]
    my_device: str

    _HVD: Optional["HVD"]
    _log_handles_already_set: bool = False

    def __init__(
        self,
        init_model: Optional[str] = None,
        init_frz_model: Optional[str] = None,
        finetune: Optional[str] = None,
        restart: Optional[str] = None,
        log_path: Optional[str] = None,
        log_level: int = 0,
        mpi_log: str = "master",
    ):
        self._try_init_distrib()

        # model init options
        self.restart = restart
        self.init_model = init_model
        self.init_frz_model = init_frz_model
        self.finetune = finetune
        self.init_mode = "init_from_scratch"

        if restart is not None:
            self.restart = os.path.abspath(restart)
            self.init_mode = "restart"
        elif init_model is not None:
            self.init_model = os.path.abspath(init_model)
            self.init_mode = "init_from_model"
        elif init_frz_model is not None:
            self.init_frz_model = os.path.abspath(init_frz_model)
            self.init_mode = "init_from_frz_model"
        elif finetune is not None:
            self.finetune = os.path.abspath(finetune)
            self.init_mode = "finetune"

        self._setup_logger(Path(log_path) if log_path else None, log_level, mpi_log)

    @property
    def is_chief(self):
        """Whether my rank is 0."""
        return self.my_rank == 0

    def print_resource_summary(self):
        """Print build and current running cluster configuration summary."""
        SummaryPrinter(self.my_device, len(self.gpus or []))()

    def _setup_logger(
        self,
        log_path: Optional[Path],
        log_level: int,
        mpi_log: Optional[str],
    ):
        """Set up package loggers.

        Parameters
        ----------
        log_level : int
            logging level
        log_path : Optional[str]
            path to log file, if None logs will be send only to console. If the parent
            directory does not exist it will be automatically created, by default None
        mpi_log : Optional[str], optional
            mpi log type. Has three options. `master` will output logs to file and
            console only from rank==0. `collect` will write messages from all ranks to
            one file opened under rank==0 and to console. `workers` will open one log
            file for each worker designated by its rank, console behaviour is the same
            as for `collect`.
        """
        if not self._log_handles_already_set:
            if not self._HVD:
                mpi_log = None
            set_log_handles(log_level, log_path, mpi_log=mpi_log)
            self._log_handles_already_set = True
            log.debug("Log handles were successfully set")
        else:
            log.warning(
                f"Log handles have already been set. It is not advisable to "
                f"reset them{', especially when runnig with MPI!' if self._HVD else ''}"
            )

    def _try_init_distrib(self):
        try:
            import horovod.tensorflow as HVD

            HVD.init()
            self.is_distrib = HVD.size() > 1
        except ImportError:
            log.warning("Switch to serial execution due to lack of horovod module.")
            self.is_distrib = False

        # Do real intialization
        if self.is_distrib:
            self._init_distributed(HVD)
            self._HVD = HVD
        else:
            self._init_serial()
            self._HVD = None

    def _init_distributed(self, HVD: "HVD"):
        """Initialize  settings for distributed training.

        Parameters
        ----------
        HVD : HVD
            horovod object
        """
        nodename, nodelist, gpus = get_resource()
        self.nodename = nodename
        self.nodelist = nodelist
        self.gpus = gpus
        self.my_rank = HVD.rank()
        self.world_size = HVD.size()

        if gpus is not None:
            gpu_idx = HVD.local_rank()
            if gpu_idx >= len(gpus):
                raise RuntimeError(
                    "Count of local processes is larger than that of available GPUs!"
                )
            self.my_device = f"gpu:{gpu_idx:d}"
            if Version(TF_VERSION) >= Version("1.14"):
                physical_devices = tf.config.experimental.list_physical_devices("GPU")
                tf.config.experimental.set_visible_devices(
                    physical_devices[gpu_idx], "GPU"
                )
        else:
            self.my_device = "cpu:0"

    def _init_serial(self):
        """Initialize setting for serial training."""
        nodename, _, gpus = get_resource()

        self.gpus = gpus
        self.world_size = 1
        self.my_rank = 0
        self.nodename = nodename
        self.nodelist = [nodename]

        if gpus is not None:
            self.my_device = "gpu:0"
        else:
            self.my_device = "cpu:0"

        self._HVD = None
