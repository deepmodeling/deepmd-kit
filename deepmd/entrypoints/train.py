"""DeePMD training entrypoint script.

Can handle local or distributed training.
"""

import json
import logging
import time
import os
from typing import Dict, TYPE_CHECKING, List, Optional, Any

import numpy as np
from deepmd.common import data_requirement, expand_sys_str, j_loader, j_must_have
from deepmd.env import tf
from deepmd.infer.data_modifier import DipoleChargeModifier
from deepmd.train.run_options import BUILD, CITATION, WELCOME, RunOptions
from deepmd.train.trainer import DPTrainer
from deepmd.utils.argcheck import normalize
from deepmd.utils.compat import updata_deepmd_input
from deepmd.utils.data_system import DeepmdDataSystem

if TYPE_CHECKING:
    from deepmd.run_options import TFServerV1

__all__ = ["train"]

log = logging.getLogger(__name__)


def create_done_queue(
    cluster_spec: tf.train.ClusterSpec, task_index: int
) -> tf.FIFOQueue:
    """Create FIFO queue for distributed tasks.

    Parameters
    ----------
    cluster_spec : tf.train.ClusterSpec
        tf cluster specification object
    task_index : int
        identifying index of a task

    Returns
    -------
    tf.FIFOQueue
        tf distributed FIFI queue
    """
    with tf.device(f"/job:ps/task:{task_index:d}"):
        queue = tf.FIFOQueue(
            cluster_spec.num_tasks("worker"),
            tf.int32,
            shared_name=f"done_queue{task_index}",
        )
        return queue


def wait_done_queue(
    cluster_spec: tf.train.ClusterSpec,
    server: "TFServerV1",
    queue: tf.FIFOQueue,
    task_index: int,
):
    """Wait until all enqued operation in tf distributed queue are finished.

    Parameters
    ----------
    cluster_spec : tf.train.ClusterSpec
        tf cluster specification object
    server : TFServerV1
        tf server specification object
    queue : tf.FIFOQueue
        tf distributed queue
    task_index : int
        identifying index of a task
    """
    with tf.Session(server.target) as sess:
        for i in range(cluster_spec.num_tasks("worker")):
            sess.run(queue.dequeue())
            log.debug(f"ps:{task_index:d} received done from worker:{i:d}")
        log.debug(f"ps:{task_index:f} quitting")


def connect_done_queue(
    cluster_spec: tf.train.ClusterSpec, task_index: int
) -> List[tf.Operation]:
    """Create tf FIFO queue filling operations.

    Parameters
    ----------
    cluster_spec : tf.train.ClusterSpec
        tf cluster specification object
    task_index : int
        identifying index of a task

    Returns
    -------
    List[tf.Operation]
        list of tf operations that will populate the queue
    """
    done_ops = []
    for i in range(cluster_spec.num_tasks("ps")):
        with tf.device(f"/job:ps/task:{i:d}"):
            queue = tf.FIFOQueue(
                cluster_spec.num_tasks("worker"), tf.int32, shared_name=f"done_queue{i}"
            )
            done_ops.append(queue.enqueue(task_index))
    return done_ops


def fill_done_queue(
    cluster_spec: tf.train.ClusterSpec,
    server: "TFServerV1",
    done_ops: List[tf.Operation],
    task_index: int,
):
    """Run specified operations that will fill the tf distributed FIFO queue.

    Parameters
    ----------
    cluster_spec : tf.train.ClusterSpec
        tf cluster specification object
    server : TFServerV1
        tf server specification object
    done_ops : List[tf.Operation]
        a list of tf operations that will fill the queue
    task_index : int
        identifying index of a task
    """
    with tf.Session(server.target) as sess:
        for i in range(cluster_spec.num_tasks("ps")):
            sess.run(done_ops[i])
            log.debug(f"worker:{task_index:d} sending done to ps:{i:d}")


def train(
    *,
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    output: str,
    mpi_log: str,
    log_level: int,
    log_path: Optional[str],
    **kwargs,
):
    """Run DeePMD model training.

    Parameters
    ----------
    INPUT : str
        json/yaml control file
    init_model : Optional[str]
        path to checkpoint folder or None
    restart : Optional[str]
        path to checkpoint folder or None
    output : str
        path for dump file with arguments
    mpi_log : str
        mpi logging mode
    log_level : int
        logging level defined by int 0-3
    log_path : Optional[str]
        logging file path or None if logs are to be output only to stdout

    Raises
    ------
    RuntimeError
        if distributed training job nem is wrong
    """
    # load json database
    jdata = j_loader(INPUT)

    jdata = updata_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")

    jdata = normalize(jdata)
    with open(output, "w") as fp:
        json.dump(jdata, fp, indent=4)

    # run options
    run_opt = RunOptions(
        init_model=init_model,
        restart=restart,
        log_path=log_path,
        log_level=log_level,
        mpi_log=mpi_log,
        try_distrib=jdata.get("with_distrib", False),
    )

    for message in WELCOME + CITATION + BUILD:
        log.info(message)

    run_opt.print_resource_summary()

    if run_opt.is_distrib:
        # distributed training
        if run_opt.my_job_name == "ps":
            queue = create_done_queue(run_opt.cluster_spec, run_opt.my_task_index)
            wait_done_queue(
                run_opt.cluster_spec, run_opt.server, queue, run_opt.my_task_index
            )
            # server.join()
        elif run_opt.my_job_name == "worker":
            done_ops = connect_done_queue(run_opt.cluster_spec, run_opt.my_task_index)
            _do_work(jdata, run_opt)
            fill_done_queue(
                run_opt.cluster_spec, run_opt.server, done_ops, run_opt.my_task_index
            )
        else:
            raise RuntimeError("unknown job name")
    else:
        # serial training
        _do_work(jdata, run_opt)


def _do_work(jdata: Dict[str, Any], run_opt: RunOptions):
    """Run serial model training.

    Parameters
    ----------
    jdata : Dict[str, Any]
        arguments read form json/yaml control file
    run_opt : RunOptions
        object with run configuration

    Raises
    ------
    RuntimeError
        If unsupported modifier type is selected for model
    """
    # make necessary checks
    assert "training" in jdata

    # init the model
    model = DPTrainer(jdata, run_opt=run_opt)
    rcut = model.model.get_rcut()
    type_map = model.model.get_type_map()
    if len(type_map) == 0:
        ipt_type_map = None
    else:
        ipt_type_map = type_map

    #  init random seed
    seed = jdata["training"].get("seed", None)
    if seed is not None:
        seed = seed % (2 ** 32)
    np.random.seed(seed)

    # setup data modifier
    modifier = get_modifier(jdata["model"].get("modifier", None))

    # init data
    train_data = get_data(jdata["training"]["training_data"], rcut, ipt_type_map, modifier)
    train_data.print_summary("training")
    if jdata["training"].get("validation_data", None) is not None:
        valid_data = get_data(jdata["training"]["validation_data"], rcut, ipt_type_map, modifier)
        valid_data.print_summary("validation")
    else:
        valid_data = None

    # get training info
    stop_batch = j_must_have(jdata["training"], "numb_steps")
    model.build(train_data, stop_batch)

    # train the model with the provided systems in a cyclic way
    start_time = time.time()
    model.train(train_data, valid_data)
    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


def get_data(jdata: Dict[str, Any], rcut, type_map, modifier):
    systems = j_must_have(jdata, "systems")
    if isinstance(systems, str):
        systems = expand_sys_str(systems)
    help_msg = 'Please check your setting for data systems'
    # check length of systems
    if len(systems) == 0:
        msg = 'cannot find valid a data system'
        log.fatal(msg)
        raise IOError(msg, help_msg)
    # rougly check all items in systems are valid
    for ii in systems:
        if (not os.path.isdir(ii)):
            msg = f'dir {ii} is not a valid dir'
            log.fatal(msg)
            raise IOError(msg, help_msg)
        if (not os.path.isfile(os.path.join(ii, 'type.raw'))):
            msg = f'dir {ii} is not a valid data system dir'
            log.fatal(msg)
            raise IOError(msg, help_msg)

    batch_size = j_must_have(jdata, "batch_size")
    sys_probs = jdata.get("sys_probs", None)
    auto_prob = jdata.get("auto_prob", "prob_sys_size")

    data = DeepmdDataSystem(
        systems=systems,
        batch_size=batch_size,
        test_size=1,        # to satisfy the old api
        shuffle_test=True,  # to satisfy the old api
        rcut=rcut,
        type_map=type_map,
        modifier=modifier,
        trn_all_set=True,    # sample from all sets
        sys_probs=sys_probs,
        auto_prob_style=auto_prob
    )
    data.add_dict(data_requirement)

    return data


def get_modifier(modi_data=None):
    modifier: Optional[DipoleChargeModifier]
    if modi_data is not None:
        if modi_data["type"] == "dipole_charge":
            modifier = DipoleChargeModifier(
                modi_data["model_name"],
                modi_data["model_charge_map"],
                modi_data["sys_charge_map"],
                modi_data["ewald_h"],
                modi_data["ewald_beta"],
            )
        else:
            raise RuntimeError("unknown modifier type " + str(modi_data["type"]))
    else:
        modifier = None
    return modifier
