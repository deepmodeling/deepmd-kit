"""DeePMD training entrypoint script.

Can handle local or distributed training.
"""

import json
import logging
import time
import os
from typing import Dict, List, Optional, Any

import numpy as np
from deepmd.common import data_requirement, expand_sys_str, j_loader, j_must_have
from deepmd.env import reset_default_tf_session_config
from deepmd.infer.data_modifier import DipoleChargeModifier
from deepmd.train.run_options import BUILD, CITATION, WELCOME, RunOptions
from deepmd.train.trainer import DPTrainer
from deepmd.train.trainer_mt import DPTrainer_mt
from deepmd.utils.argcheck import normalize
from deepmd.utils.argcheck_mt import normalize_mt
from deepmd.utils.compat import updata_deepmd_input
from deepmd.utils.data_system import DeepmdDataSystem,DeepmdDataDocker
from deepmd.utils.sess import run_sess
from deepmd.utils.neighbor_stat import NeighborStat

__all__ = ["train"]

log = logging.getLogger(__name__)


def train(
    *,
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    output: str,
    mpi_log: str,
    log_level: int,
    log_path: Optional[str],
    multi_task : bool,
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
    mulit-task : bool
        whether using logic of multi_tasking
    Raises
    ------
    RuntimeError
        if distributed training job nem is wrong
    """
    # load json database
    jdata = j_loader(INPUT)

    jdata = updata_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")

    if multi_task:
        jdata = normalize_mt(jdata)
    else:
        jdata = normalize(jdata)


    jdata = update_sel(jdata)

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
        multi_task = multi_task,

    )

    for message in WELCOME + CITATION + BUILD:
        log.info(message)

    run_opt.print_resource_summary()
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

    # avoid conflict of visible gpus among multipe tf sessions in one process
    if run_opt.is_distrib and len(run_opt.gpus or []) > 1:
        reset_default_tf_session_config(cpu_only=True)

    # init the model
    if not run_opt.multi_task:
        model = DPTrainer(jdata, run_opt=run_opt)
        rcut = model.model.get_rcut()
        type_map = model.model.get_type_map()
    else:
        model = DPTrainer_mt(jdata, run_opt=run_opt)
        for model_name in model.model_dict.keys():
            sub_model = model.model_dict[model_name]
            rcut = sub_model.get_rcut()
            type_map = sub_model.get_type_map()

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
    if run_opt.multi_task:
        train_data = get_data_mt(jdata["training"]["training_data"], rcut, ipt_type_map, modifier)
        train_data.print_summary("training")
        if jdata["training"].get("validation_data", None) is not None:
            valid_data = get_data_mt(jdata["training"]["validation_data"], rcut, ipt_type_map, modifier)
            valid_data.print_summary("validation")
        else:
            valid_data = None
    else:
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

def get_data_mt(jdata: Dict[str, Any], rcut, type_map, modifier):
    systems = j_must_have(jdata, "systems")
    batch_size = j_must_have(jdata, "batch_size")
    sys_probs = jdata.get("sys_probs", None)
    auto_prob = jdata.get("auto_prob", "prob_sys_size")
    total_data = []
    for sub_sys in systems:
        data = DeepmdDataSystem(
            systems=sub_sys['data'],
            batch_size=batch_size,
            test_size=1,        # to satisfy the old api
            shuffle_test=True,  # to satisfy the old api
            rcut=rcut,
            #type_map=sub_fitting['type_map'],  # this is the local type map
            type_map=type_map,  # this is the local type map
            modifier=modifier,
            trn_all_set=True,    # sample from all sets
            sys_probs=sys_probs,
            auto_prob_style=auto_prob,
            name = sub_sys['name']
        )
        data.add_dict(data_requirement)
        total_data.append(data)
    docker = DeepmdDataDocker(
        data_systems=total_data,
        batch_size = batch_size,
        type_map=type_map   # in the data docker is the total type
    )
    return docker


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


def get_rcut(jdata):
    descrpt_data = jdata['model']['descriptor']
    rcut_list = []
    if descrpt_data['type'] == 'hybrid':
        for ii in descrpt_data['list']:
            rcut_list.append(ii['rcut'])
    else:
        rcut_list.append(descrpt_data['rcut'])
    return max(rcut_list)


def get_type_map(jdata):
    return jdata['model'].get('type_map', None)


def get_sel(jdata, rcut):
    max_rcut = get_rcut(jdata)
    type_map = get_type_map(jdata)

    if type_map and len(type_map) == 0:
        type_map = None
    train_data = get_data(jdata["training"]["training_data"], max_rcut, type_map, None)
    train_data.get_batch()
    data_ntypes = train_data.get_ntypes()
    if type_map is not None:
        map_ntypes = len(type_map)
    else:
        map_ntypes = data_ntypes
    ntypes = max([map_ntypes, data_ntypes])

    neistat = NeighborStat(ntypes, rcut)

    min_nbor_dist, max_nbor_size = neistat.get_stat(train_data)

    return max_nbor_size


def parse_auto_sel(sel):
    if type(sel) is not str:
        return False
    words = sel.split(':')
    if words[0] == 'auto':
        return True
    else:
        return False

    
def parse_auto_sel_ratio(sel):
    if not parse_auto_sel(sel):
        raise RuntimeError(f'invalid auto sel format {sel}')
    else:
        words = sel.split(':')
        if len(words) == 1:
            ratio = 1.1
        elif len(words) == 2:
            ratio = float(words[1])
        else:
            raise RuntimeError(f'invalid auto sel format {sel}')
        return ratio


def wrap_up_4(xx):
    return 4 * ((int(xx) + 3) // 4)


def update_one_sel(jdata, descriptor):
    rcut = descriptor['rcut']
    tmp_sel = get_sel(jdata, rcut)
    if parse_auto_sel(descriptor['sel']) :
        ratio = parse_auto_sel_ratio(descriptor['sel'])
        descriptor['sel'] = [int(wrap_up_4(ii * ratio)) for ii in tmp_sel]
    else:
        # sel is set by user
        for ii, (tt, dd) in enumerate(zip(tmp_sel, descriptor['sel'])):
            if dd and tt > dd:
                # we may skip warning for sel=0, where the user is likely
                # to exclude such type in the descriptor
                log.warning(
                    "sel of type %d is not enough! The expected value is "
                    "not less than %d, but you set it to %d. The accuracy"
                    " of your model may get worse." %(ii, tt, dd)
                )
    return descriptor


def update_sel(jdata):    
    descrpt_data = jdata['model']['descriptor']
    if isinstance(descrpt_data,list):
        update_descrpt = []
        for sub_descrpt in descrpt_data:
            if sub_descrpt['type'] == 'hybrid':
                for ii in range(len(sub_descrpt['list'])):
                    sub_descrpt['list'][ii] = update_one_sel(jdata, sub_descrpt['list'][ii])
            else:
                sub_descrpt = update_one_sel(jdata, sub_descrpt)
            update_descrpt.append(sub_descrpt)
        jdata['model']['descriptor'] = update_descrpt
    else:
        if descrpt_data['type'] == 'hybrid':
            for ii in range(len(descrpt_data['list'])):
                descrpt_data['list'][ii] = update_one_sel(jdata, descrpt_data['list'][ii])
        else:
            descrpt_data = update_one_sel(jdata, descrpt_data)
        jdata['model']['descriptor'] = descrpt_data
    return jdata
