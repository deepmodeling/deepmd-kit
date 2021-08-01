"""Module taking care of important package constants."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from deepmd.cluster import get_resource
from deepmd.env import get_tf_default_nthreads, tf, GLOBAL_CONFIG, global_float_prec
from deepmd.loggers import set_log_handles

if TYPE_CHECKING:
    import horovod.tensorflow as HVD


__all__ = [
    "WELCOME",
    "CITATION",
    "BUILD",
    "RunOptions",
]

log = logging.getLogger(__name__)


# http://patorjk.com/software/taag. Font:Big"
WELCOME = (  # noqa
    " _____               _____   __  __  _____           _     _  _   ",
    "|  __ \             |  __ \ |  \/  ||  __ \         | |   (_)| |  ",
    "| |  | |  ___   ___ | |__) || \  / || |  | | ______ | | __ _ | |_ ",
    "| |  | | / _ \ / _ \|  ___/ | |\/| || |  | ||______|| |/ /| || __|",
    "| |__| ||  __/|  __/| |     | |  | || |__| |        |   < | || |_ ",
    "|_____/  \___| \___||_|     |_|  |_||_____/         |_|\_\|_| \__|",
)

CITATION = (
    "Please read and cite:",
    "Wang, Zhang, Han and E, Comput.Phys.Comm. 228, 178-184 (2018)",
)

_sep = "\n                      "
BUILD = (
    f"installed to:         {GLOBAL_CONFIG['install_prefix']}",
    f"source :              {GLOBAL_CONFIG['git_summ']}",
    f"source brach:         {GLOBAL_CONFIG['git_branch']}",
    f"source commit:        {GLOBAL_CONFIG['git_hash']}",
    f"source commit at:     {GLOBAL_CONFIG['git_date']}",
    f"build float prec:     {global_float_prec}",
    f"build with tf inc:    {GLOBAL_CONFIG['tf_include_dir']}",
    f"build with tf lib:    {GLOBAL_CONFIG['tf_libs'].replace(';', _sep)}"  # noqa
)


class RunOptions:
    """Class with inf oon how to run training (cluster, MPI and GPU config).

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
        restart: Optional[str] = None,
        log_path: Optional[str] = None,
        log_level: int = 0,
        mpi_log: str = "master"
    ):
        self._try_init_distrib()

        if all((init_model, restart)):
            raise RuntimeError(
                "--init-model and --restart should not be set at the same time"
            )

        # model init options
        self.restart = restart
        self.init_model = init_model
        self.init_mode = "init_from_scratch"

        if restart is not None:
            self.restart = os.path.abspath(restart)
            self.init_mode = "restart"
        elif init_model is not None:
            self.init_model = os.path.abspath(init_model)
            self.init_mode = "init_from_model"

        self._setup_logger(Path(log_path) if log_path else None, log_level, mpi_log)

    @property
    def is_chief(self):
        """Whether my rank is 0."""
        return self.my_rank == 0

    def print_resource_summary(self):
        """Print build and current running cluster configuration summary."""
        log.info("---Summary of the training---------------------------------------")
        if self.is_distrib:
            log.info("distributed")
            log.info(f"world size:           {self.world_size}")
            log.info(f"my rank:              {self.my_rank}")
            log.info(f"node list:            {self.nodelist}")
        log.info(f"running on:           {self.nodename}")
        log.info(f"computing device:     {self.my_device}")
        env_value = os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')
        log.info(f"CUDA_VISIBLE_DEVICES: {env_value}")
        log.info(f"Count of visible GPU: {len(self.gpus or [])}")
        intra, inter = get_tf_default_nthreads()
        log.info(f"num_intra_threads:    {intra:d}")
        log.info(f"num_inter_threads:    {inter:d}")
        log.info("-----------------------------------------------------------------")

    def _setup_logger(
        self,
        log_path: Optional[Path],
        log_level: int,
        mpi_log: Optional[str],
    ):
        """Set up package loggers.

        Parameters
        ----------
        log_level: int
            logging level
        log_path: Optional[str]
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
                raise RuntimeError('Count of local processes is larger than that of available GPUs!')
            self.my_device = f"gpu:{gpu_idx:d}"
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
