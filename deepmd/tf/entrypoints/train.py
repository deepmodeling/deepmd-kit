# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD training entrypoint script.

Can handle local or distributed training.
"""

import json
import logging
import time
from typing import (
    Any,
    Dict,
    Optional,
)

from deepmd.tf.common import (
    j_loader,
    j_must_have,
)
from deepmd.tf.env import (
    reset_default_tf_session_config,
    tf,
)
from deepmd.tf.infer.data_modifier import (
    DipoleChargeModifier,
)
from deepmd.tf.model.model import (
    Model,
)
from deepmd.tf.train.run_options import (
    RunOptions,
)
from deepmd.tf.train.trainer import (
    DPTrainer,
)
from deepmd.tf.utils import random as dp_random
from deepmd.tf.utils.argcheck import (
    normalize,
)
from deepmd.tf.utils.compat import (
    update_deepmd_input,
)
from deepmd.tf.utils.finetune import (
    replace_model_params_with_pretrained_model,
)
from deepmd.tf.utils.multi_init import (
    replace_model_params_with_frz_multi_model,
)
from deepmd.utils.data_system import (
    get_data,
)

__all__ = ["train"]

log = logging.getLogger(__name__)


def train(
    *,
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    output: str,
    init_frz_model: str,
    mpi_log: str,
    log_level: int,
    log_path: Optional[str],
    is_compress: bool = False,
    skip_neighbor_stat: bool = False,
    finetune: Optional[str] = None,
    **kwargs,
):
    """Run DeePMD model training.

    Parameters
    ----------
    INPUT : str
        json/yaml control file
    init_model : Optional[str]
        path prefix of checkpoint files or None
    restart : Optional[str]
        path prefix of checkpoint files or None
    output : str
        path for dump file with arguments
    init_frz_model : str
        path to frozen model or None
    mpi_log : str
        mpi logging mode
    log_level : int
        logging level defined by int 0-3
    log_path : Optional[str]
        logging file path or None if logs are to be output only to stdout
    is_compress : bool
        indicates whether in the model compress mode
    skip_neighbor_stat : bool, default=False
        skip checking neighbor statistics
    finetune : Optional[str]
        path to pretrained model or None
    **kwargs
        additional arguments

    Raises
    ------
    RuntimeError
        if distributed training job name is wrong
    """
    run_opt = RunOptions(
        init_model=init_model,
        restart=restart,
        init_frz_model=init_frz_model,
        finetune=finetune,
        log_path=log_path,
        log_level=log_level,
        mpi_log=mpi_log,
    )
    if run_opt.is_distrib and len(run_opt.gpus or []) > 1:
        # avoid conflict of visible gpus among multipe tf sessions in one process
        reset_default_tf_session_config(cpu_only=True)

    # load json database
    jdata = j_loader(INPUT)

    origin_type_map = None
    if run_opt.finetune is not None:
        jdata, origin_type_map = replace_model_params_with_pretrained_model(
            jdata, run_opt.finetune
        )

    if "fitting_net_dict" in jdata["model"] and run_opt.init_frz_model is not None:
        jdata = replace_model_params_with_frz_multi_model(jdata, run_opt.init_frz_model)

    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")

    jdata = normalize(jdata)

    if not is_compress and not skip_neighbor_stat:
        jdata = update_sel(jdata)

    with open(output, "w") as fp:
        json.dump(jdata, fp, indent=4)

    # save the training script into the graph
    # remove white spaces as it is not compressed
    tf.constant(
        json.dumps(jdata, separators=(",", ":")),
        name="train_attr/training_script",
        dtype=tf.string,
    )

    run_opt.print_resource_summary()
    if origin_type_map is not None:
        jdata["model"]["origin_type_map"] = origin_type_map
    _do_work(jdata, run_opt, is_compress)


def _do_work(jdata: Dict[str, Any], run_opt: RunOptions, is_compress: bool = False):
    """Run serial model training.

    Parameters
    ----------
    jdata : Dict[str, Any]
        arguments read form json/yaml control file
    run_opt : RunOptions
        object with run configuration
    is_compress : Bool
        indicates whether in model compress mode

    Raises
    ------
    RuntimeError
        If unsupported modifier type is selected for model
    """
    # make necessary checks
    assert "training" in jdata

    # init the model
    model = DPTrainer(jdata, run_opt=run_opt, is_compress=is_compress)
    rcut = model.model.get_rcut()
    type_map = model.model.get_type_map()
    if len(type_map) == 0:
        ipt_type_map = None
    else:
        ipt_type_map = type_map

    # init random seed of data systems
    seed = jdata["training"].get("seed", None)
    if seed is not None:
        # avoid the same batch sequence among workers
        seed += run_opt.my_rank
        seed = seed % (2**32)
    dp_random.seed(seed)

    # setup data modifier
    modifier = get_modifier(jdata["model"].get("modifier", None))

    # check the multi-task mode
    multi_task_mode = "fitting_net_dict" in jdata["model"]

    # decouple the training data from the model compress process
    train_data = None
    valid_data = None
    if not is_compress:
        # init data
        if not multi_task_mode:
            train_data = get_data(
                jdata["training"]["training_data"], rcut, ipt_type_map, modifier
            )
            train_data.print_summary("training")
            if jdata["training"].get("validation_data", None) is not None:
                valid_data = get_data(
                    jdata["training"]["validation_data"],
                    rcut,
                    train_data.type_map,
                    modifier,
                )
                valid_data.print_summary("validation")
        else:
            train_data = {}
            valid_data = {}
            for data_systems in jdata["training"]["data_dict"]:
                if (
                    jdata["training"]["fitting_weight"][data_systems] > 0.0
                ):  # check only the available pair
                    train_data[data_systems] = get_data(
                        jdata["training"]["data_dict"][data_systems]["training_data"],
                        rcut,
                        ipt_type_map,
                        modifier,
                        multi_task_mode,
                    )
                    train_data[data_systems].print_summary(
                        f"training in {data_systems}"
                    )
                    if (
                        jdata["training"]["data_dict"][data_systems].get(
                            "validation_data", None
                        )
                        is not None
                    ):
                        valid_data[data_systems] = get_data(
                            jdata["training"]["data_dict"][data_systems][
                                "validation_data"
                            ],
                            rcut,
                            train_data[data_systems].type_map,
                            modifier,
                            multi_task_mode,
                        )
                        valid_data[data_systems].print_summary(
                            f"validation in {data_systems}"
                        )
    else:
        if modifier is not None:
            modifier.build_fv_graph()

    # get training info
    stop_batch = j_must_have(jdata["training"], "numb_steps")
    origin_type_map = jdata["model"].get("origin_type_map", None)
    if (
        origin_type_map is not None and not origin_type_map
    ):  # get the type_map from data if not provided
        origin_type_map = get_data(
            jdata["training"]["training_data"], rcut, None, modifier
        ).get_type_map()
    model.build(train_data, stop_batch, origin_type_map=origin_type_map)

    if not is_compress:
        # train the model with the provided systems in a cyclic way
        start_time = time.time()
        model.train(train_data, valid_data)
        end_time = time.time()
        log.info("finished training")
        log.info(f"wall time: {(end_time - start_time):.3f} s")
    else:
        model.save_compressed()
        log.info("finished compressing")


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


def update_sel(jdata):
    log.info(
        "Calculate neighbor statistics... (add --skip-neighbor-stat to skip this step)"
    )
    jdata_cpy = jdata.copy()
    jdata_cpy["model"] = Model.update_sel(jdata, jdata["model"])
    return jdata_cpy
