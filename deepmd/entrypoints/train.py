"""DeePMD training entrypoint script.

Can handle local or distributed training.
"""

import json
import logging
import time
import os
from typing import Dict, List, Optional, Any

from deepmd.common import data_requirement, expand_sys_str, j_loader, j_must_have
from deepmd.env import tf, reset_default_tf_session_config, GLOBAL_ENER_FLOAT_PRECISION
from deepmd.infer.data_modifier import DipoleChargeModifier
from deepmd.train.run_options import BUILD, CITATION, WELCOME, RunOptions
from deepmd.train.trainer import DPTrainer
from deepmd.utils import random as dp_random
from deepmd.utils.argcheck import normalize
from deepmd.utils.compat import updata_deepmd_input
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils.sess import run_sess
from deepmd.utils.neighbor_stat import NeighborStat
from deepmd.utils.path import DPPath

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
    init_frz_model : str
        path to frozen model or None
    mpi_log : str
        mpi logging mode
    log_level : int
        logging level defined by int 0-3
    log_path : Optional[str]
        logging file path or None if logs are to be output only to stdout
    is_compress: bool
        indicates whether in the model compress mode

    Raises
    ------
    RuntimeError
        if distributed training job nem is wrong
    """
    run_opt = RunOptions(
        init_model=init_model,
        restart=restart,
        init_frz_model=init_frz_model,
        log_path=log_path,
        log_level=log_level,
        mpi_log=mpi_log
    )
    if run_opt.is_distrib and len(run_opt.gpus or []) > 1:
        # avoid conflict of visible gpus among multipe tf sessions in one process
        reset_default_tf_session_config(cpu_only=True)

    # load json database
    jdata = j_loader(INPUT)

    jdata = updata_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")

    jdata = normalize(jdata)

    if not is_compress:
        jdata = update_sel(jdata)

    with open(output, "w") as fp:
        json.dump(jdata, fp, indent=4)

    # save the training script into the graph
    tf.constant(json.dumps(jdata), name='train_attr/training_script', dtype=tf.string)

    for message in WELCOME + CITATION + BUILD:
        log.info(message)

    run_opt.print_resource_summary()
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
    model = DPTrainer(jdata, run_opt=run_opt, is_compress = is_compress)
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
        seed = seed % (2 ** 32)
    dp_random.seed(seed)

    # setup data modifier
    modifier = get_modifier(jdata["model"].get("modifier", None))

    # decouple the training data from the model compress process
    train_data = None
    valid_data = None
    if not is_compress:
        # init data
        train_data = get_data(jdata["training"]["training_data"], rcut, ipt_type_map, modifier)
        train_data.print_summary("training")
        if jdata["training"].get("validation_data", None) is not None:
            valid_data = get_data(jdata["training"]["validation_data"], rcut, ipt_type_map, modifier)
            valid_data.print_summary("validation")

    # get training info
    stop_batch = j_must_have(jdata["training"], "numb_steps")
    model.build(train_data, stop_batch)

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
        ii = DPPath(ii)
        if (not ii.is_dir()):
            msg = f'dir {ii} is not a valid dir'
            log.fatal(msg)
            raise IOError(msg, help_msg)
        if (not (ii / 'type.raw').is_file()):
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


def get_nbor_stat(jdata, rcut):
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

    # moved from traier.py as duplicated
    # TODO: this is a simple fix but we should have a clear
    #       architecture to call neighbor stat
    tf.constant(min_nbor_dist,
        name = 'train_attr/min_nbor_dist',
        dtype = GLOBAL_ENER_FLOAT_PRECISION)
    tf.constant(max_nbor_size,
        name = 'train_attr/max_nbor_size',
        dtype = tf.int32)
    return min_nbor_dist, max_nbor_size

def get_sel(jdata, rcut):
    _, max_nbor_size = get_nbor_stat(jdata, rcut)
    return max_nbor_size

def get_min_nbor_dist(jdata, rcut):
    min_nbor_dist, _ = get_nbor_stat(jdata, rcut)
    return min_nbor_dist

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
    if descrpt_data['type'] == 'hybrid':
        for ii in range(len(descrpt_data['list'])):
            descrpt_data['list'][ii] = update_one_sel(jdata, descrpt_data['list'][ii])
    else:
        descrpt_data = update_one_sel(jdata, descrpt_data)
    jdata['model']['descriptor'] = descrpt_data
    return jdata
