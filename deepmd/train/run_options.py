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
    from mpi4py import MPI

    try:
        from typing import Protocol  # python >=3.8
    except ImportError:
        from typing_extensions import Protocol  # type: ignore

    class TFServerV1(Protocol):
        """Prococol mimicking parser object."""

        server_def: tf.train.ServerDef
        target: str


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


def _is_distributed(MPI: "MPI") -> bool:
    """Check if there are more than one MPI processes.

    Parameters
    ----------
    MPI : MPI
        MPI object

    Returns
    -------
    bool
        True if we have more than 1 MPI process
    """
    return MPI.COMM_WORLD.Get_size() > 1


def _distributed_task_config(
    MPI: "MPI",
    node_name: str,
    node_list_: List[str],
    gpu_list: Optional[List[int]] = None,
    default_port: int = 2222,
) -> Tuple[Dict[str, List[str]], str, int, str, str]:
    """Create configuration for distributed tensorflow session.

    Parameters
    ----------
    MPI : mpi4py.MPI
        MPI module
    node_name : str
        the name of current node
    node_list_ : List[str]
        the list of nodes of the current mpirun
    gpu_list : Optional[List[int]], optional
        the list of GPUs on each node, by default None
    default_port : int, optional
        the default port for socket communication, by default 2222

    Returns
    -------
    Tuple[Dict[str, List[str]], str, int, str, str]
        cluster specification, job name of this task, index of this task,
        hostname:port socket of this task, the device for this task
    """
    # setup cluster
    node_list = list(set(node_list_))
    node_list.sort()
    node_color = node_list.index(node_name)
    world_idx = MPI.COMM_WORLD.Get_rank()
    node_comm = MPI.COMM_WORLD.Split(node_color, world_idx)
    node_task_idx = node_comm.Get_rank()
    node_numb_task = node_comm.Get_size()

    socket_list = []
    for ii in node_list:
        for jj in range(node_numb_task):
            socket_list.append(f"{ii}:{default_port + jj}")
    ps_map = socket_list[0:1]
    worker_map = socket_list[1:]

    if node_color == 0 and node_task_idx == 0:
        my_job = "ps"
        my_socket = ps_map[0]
        my_task_idx = ps_map.index(my_socket)
    else:
        my_job = "worker"
        my_socket = f"{node_name}:{default_port - node_task_idx}"
        assert my_socket in worker_map
        my_task_idx = worker_map.index(my_socket)

    # setup gpu/cpu devices
    if gpu_list is not None:
        numb_gpu = len(gpu_list)
        gpu_idx = node_numb_task - node_task_idx - 1
        if gpu_idx >= numb_gpu:
            my_device = "cpu:0"  # "cpu:%d" % node_task_idx
        else:
            my_device = f"gpu:{gpu_idx:d}"
    else:
        my_device = "cpu:0"  # "cpu:%d" % node_task_idx

    cluster = {"worker": worker_map, "ps": ps_map}
    return cluster, my_job, my_task_idx, my_socket, my_device


class RunOptions:
    """Class with inf oon how to run training (cluster, MPI and GPU config).

    Attributes
    ----------
    cluster: Optional[Dict[str, List[str]]]
        cluster informations as dict
    cluster_spec: Optional[tf.train.ClusterSpec]
        `tf.train.ClusterSpec` or None if training is serial
    gpus: Optional[List[int]]
        list of GPUs if any are present else None
    is_chief: bool
        in distribured training it is true for tha main MPI process in serail it is
        always true
    my_job_name: str
        name of the training job
    my_socket: Optional[str]
        communication socket for distributed training
    my_task_index: int
        index of the MPI task
    nodename: str
        name of the node
    num_ps: Optional[int]
        number of ps
    num_workers: Optional[int]
        number of workers
    server: Optional[tf.train.Server]
        `tf.train.Server` or `None` for serial training
    my_device: str
        deviice type - gpu or cpu
    """

    cluster: Optional[Dict[str, List[str]]]
    cluster_spec: Optional[tf.train.ClusterSpec]
    gpus: Optional[List[int]]
    is_chief: bool
    my_job_name: str
    my_socket: Optional[str]
    my_task_index: int
    nodename: str
    num_ps: Optional[int]
    num_workers: Optional[int]
    server: Optional["TFServerV1"]
    my_device: str

    _MPI: Optional["MPI"]
    _log_handles_already_set: bool = False

    def __init__(
        self,
        init_model: Optional[str] = None,
        restart: Optional[str] = None,
        log_path: Optional[str] = None,
        log_level: int = 0,
        mpi_log: str = "master",
        try_distrib: bool = False
    ):
        # distributed tasks
        if try_distrib:
            self._try_init_mpi()
        else:
            self.is_distrib = False
            self._init_serial()

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

    def print_resource_summary(self):
        """Print build and current running cluster configuration summary."""
        log.info("---Summary of the training---------------------------------------")
        if self.is_distrib:
            log.info("distributed")
            log.info(f"ps list:              {self.cluster['ps']}")
            log.info(f"worker list:          {self.cluster['worker']}")
            log.info(f"chief on:             {self.nodename}")
        else:
            log.info(f"running on:           {self.nodename}")
        if self.gpus is None:
            log.info(f"CUDA_VISIBLE_DEVICES: unset")
        else:
            log.info(f"CUDA_VISIBLE_DEVICES: {self.gpus}")
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
            as for `collect`. If this argument is specified than also `MPI` object must
            be passed in. by default None
        """
        if not self._log_handles_already_set:
            if not self._MPI:
                mpi_log = None
            set_log_handles(log_level, log_path, mpi_log=mpi_log, MPI=self._MPI)
            self._log_handles_already_set = True
            log.debug("Log handles were successfully set")
        else:
            log.warning(
                f"Log handles have already been set. It is not advisable to "
                f"reset them{', especially when runnig with MPI!' if self._MPI else ''}"
            )

    def _try_init_mpi(self):
        try:
            from mpi4py import MPI
        except ImportError:
            raise RuntimeError(
                "cannot import mpi4py module, cannot do distributed simulation"
            )
        else:
            self.is_distrib = _is_distributed(MPI)
            if self.is_distrib:
                self._init_distributed(MPI)
                self._MPI = MPI
            else:
                self._init_serial()
                self._MPI = None

    def _init_distributed(self, MPI: "MPI"):
        """Initialize  settings for distributed training.

        Parameters
        ----------
        MPI : MPI
            MPI object
        """
        nodename, nodelist, gpus = get_resource()
        self.nodename = nodename
        self.gpus = gpus
        (
            self.cluster,
            self.my_job_name,
            self.my_task_index,
            self.my_socket,
            self.my_device,
        ) = _distributed_task_config(MPI, nodename, nodelist, gpus)
        self.is_chief = self.my_job_name == "worker" and self.my_task_index == 0
        self.num_ps = len(self.cluster["ps"])
        self.num_workers = len(self.cluster["worker"])
        self.cluster_spec = tf.train.ClusterSpec(self.cluster)
        self.server = tf.train.Server(
            server_or_cluster_def=self.cluster_spec,
            job_name=self.my_job_name,
            task_index=self.my_task_index,
        )

    def _init_serial(self):
        """Initialize setting for serial training."""
        nodename, _, gpus = get_resource()

        self.cluster = None
        self.cluster_spec = None
        self.gpus = gpus
        self.is_chief = True
        self.my_job_name = nodename
        self.my_socket = None
        self.my_task_index = 0
        self.nodename = nodename
        self.num_ps = None
        self.num_workers = None
        self.server = None

        if gpus is not None:
            self.my_device = f"gpu:{gpus[0]:d}"
        else:
            self.my_device = "cpu:0"

        self._MPI = None
