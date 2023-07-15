import os
import json
import logging
from deepmd.env import tf
from typing import Optional

from deepmd.common import j_loader
from deepmd.env import tf, GLOBAL_ENER_FLOAT_PRECISION
from deepmd.utils.argcheck import normalize
from deepmd.utils.compat import update_deepmd_input
from deepmd.utils.errors import GraphTooLargeError, GraphWithoutTensorError
from deepmd.utils.graph import get_tensor_by_name
from deepmd.entrypoints.freeze import freeze
from deepmd.entrypoints.train import train, get_rcut, get_min_nbor_dist
from deepmd.entrypoints.transfer import transfer

log = logging.getLogger(__name__)
dp_float_prec = os.environ.get("DP_INTERFACE_PREC", "high").lower()

def mix_precision(
    input: str,
    output: str,
    checkpoint_folder: str,
    training_script: str,
    mpi_log: str,
    log_path: Optional[str],
    log_level: int,
    **kwargs
):
    """convert trained model to Ascend mix-precision model.

    Generate a Ascend_transfer mix-precision model.

    Parameters
    ----------
    input : str
        frozen model file to compress
    output : str
        compressed model filename
    checkpoint_folder : str
        trining checkpoint folder for freezing
    training_script : str
        training script of the input frozen model
    mpi_log : str
        mpi logging mode for training
    log_path : Optional[str]
        if speccified log will be written to this file
    log_level : int
        logging level
    """
    assert (dp_float_prec == "ascend_mix"), \
        "Transfering Ascend_transfer mix-precision model needs to set environment variable DP_INTERFACE_PREC=ascend_mix!"
    try:
        t_jdata = get_tensor_by_name(input, 'train_attr/training_script')
        t_min_nbor_dist = get_tensor_by_name(input, 'train_attr/min_nbor_dist')
        jdata = json.loads(t_jdata)
    except GraphWithoutTensorError as e:
        if training_script == None:
            raise RuntimeError(
                "The input transfering model: %s has no training script or min_nbor_dist information, "
                "which is not supported by the transfer_to_ascend interface. "
                "Please consider using the --training-script command within the transfer_to_ascend interface to provide the training script of the input transfering model. "
                "Note that the input transfering script must contain the correct path to the training data." % input
            ) from e
        elif not os.path.exists(training_script):
            raise RuntimeError(
                "The input training script %s (%s) does not exist! Please check the path of the training script. " % (input, os.path.abspath(input))
            ) from e
        else:
            log.info("stage 0: compute the min_nbor_dist")
            jdata = j_loader(training_script)
            jdata = update_deepmd_input(jdata)
            t_min_nbor_dist = get_min_nbor_dist(jdata, get_rcut(jdata))

    _check_descrpt_type(jdata)
    _check_transfer_model_type(input)

    tf.constant(t_min_nbor_dist,
        name = 'train_attr/min_nbor_dist',
        dtype = GLOBAL_ENER_FLOAT_PRECISION)
    jdata["model"]["descriptor"]["precision"] = "float16"
    jdata["model"]["fitting_net"]["precision"] = "float16"
    jdata["model"]["transfered_from_model"] = True
    jdata["training"]["save_ckpt"] = os.path.join("model-transfer", "model.ckpt")
    jdata = update_deepmd_input(jdata)
    jdata = normalize(jdata)

    # check the descriptor info of the input file
    # move to the specific Descriptor class

    # stage 1: training or refining the model with tabulation
    log.info("\n\n")
    log.info("stage 1: generate the mix-precision model")
    control_file = "ascend-transfer.json"
    with open(control_file, "w") as fp:
        json.dump(jdata, fp, indent=4)
    try:
        train(
            INPUT=control_file,
            init_model=None,
            restart=None,
            init_frz_model=input,
            output=control_file,
            mpi_log=mpi_log,
            log_level=log_level,
            log_path=log_path,
            is_compress=False,
            skip_neighbor_stat=True,
        )
    except GraphTooLargeError as e:
        raise RuntimeError(
            "The ascend transfered mix-precision model size is too large, "
            "exceeding protobuf's limitation (2 GB)."
        ) from e

    # reset the graph, otherwise the size limitation will be only 2 GB / 2 = 1 GB
    tf.reset_default_graph()

    # stage 2: freeze the mix-precision model
    log.info("\n\n")
    log.info("stage 2: freeze the mix-precision model")
    try:
        freeze(checkpoint_folder=checkpoint_folder, output=output, node_names=None)
    except GraphTooLargeError as e:
        raise RuntimeError(
            "The ascend transfered mix-precision model size is too large, "
            "exceeding protobuf's limitation (2 GB)."
        ) from e

    # stage 3: transfer the mix-precision model
    log.info("\n\n")
    log.info("stage 3: transfer the mix-precision model")
    if output.endswith(".pb"):
        const_out = output[:-3] + "_const.pb"
    else:
        const_out = output + "_const"
    transfer(old_model=input, raw_model=output, output=output, ascend_graph=const_out)

def _check_transfer_model_type(model_file):
    try:
        t_model_type = bytes.decode(get_tensor_by_name(model_file, 'model_type'))
    except GraphWithoutTensorError as e:
        # Compatible with the upgraded model, which has no 'model_type' info
        t_model_type = None
    
    if t_model_type == "ascend_transfer_model":
        raise RuntimeError("The input model %s has already been transfered to ascend mix-precision model! Please do not transfer the model repeatedly. " % model_file)

def _check_descrpt_type(t_jdata):
    try:
        descrpt_type = t_jdata["model"]["descriptor"]["type"]
    except GraphWithoutTensorError as e:
        # Compatible with the upgraded model, which has no 'model_type' info
        descrpt_type = "se_a"
    
    if descrpt_type != "se_a" and descrpt_type != "se_e2_a":
        raise RuntimeError("Now the Ascend platform only supports se_a descriptor. There will be more descriptor support in the future!")