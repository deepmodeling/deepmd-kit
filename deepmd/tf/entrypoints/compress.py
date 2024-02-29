# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress a model, which including tabulating the embedding-net."""

import json
import logging
import os
from typing import (
    Optional,
)

from deepmd.tf.common import (
    j_loader,
)
from deepmd.tf.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.utils.argcheck import (
    normalize,
)
from deepmd.tf.utils.compat import (
    update_deepmd_input,
)
from deepmd.tf.utils.errors import (
    GraphTooLargeError,
    GraphWithoutTensorError,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name_from_graph,
    load_graph_def,
)
from deepmd.tf.utils.update_sel import (
    UpdateSel,
)

from .freeze import (
    freeze,
)
from .train import (
    train,
)

__all__ = ["compress"]

log = logging.getLogger(__name__)


def compress(
    *,
    input: str,
    output: str,
    extrapolate: int,
    step: float,
    frequency: str,
    checkpoint_folder: str,
    training_script: str,
    mpi_log: str,
    log_path: Optional[str],
    log_level: int,
    **kwargs,
):
    """Compress model.

    The table is composed of fifth-order polynomial coefficients and is assembled from
    two sub-tables. The first table takes the step parameter as the domain's uniform step size,
    while the second table takes 10 * step as it's uniform step size. The range of the
    first table is automatically detected by the code, while the second table ranges
    from the first table's upper boundary(upper) to the extrapolate(parameter) * upper.

    Parameters
    ----------
    input : str
        frozen model file to compress
    output : str
        compressed model filename
    extrapolate : int
        scale of model extrapolation
    step : float
        uniform step size of the tabulation's first table
    frequency : str
        frequency of tabulation overflow check
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
    **kwargs
        additional arguments
    """
    graph, _ = load_graph_def(input)
    try:
        t_jdata = get_tensor_by_name_from_graph(graph, "train_attr/training_script")
        t_min_nbor_dist = get_tensor_by_name_from_graph(
            graph, "train_attr/min_nbor_dist"
        )
        jdata = json.loads(t_jdata)
    except GraphWithoutTensorError as e:
        if training_script is None:
            raise RuntimeError(
                "The input frozen model: %s has no training script or min_nbor_dist information, "
                "which is not supported by the model compression interface. "
                "Please consider using the --training-script command within the model compression interface to provide the training script of the input frozen model. "
                "Note that the input training script must contain the correct path to the training data."
                % input
            ) from e
        elif not os.path.exists(training_script):
            raise RuntimeError(
                f"The input training script {input} ({os.path.abspath(input)}) does not exist! Please check the path of the training script. "
            ) from e
        else:
            log.info("stage 0: compute the min_nbor_dist")
            jdata = j_loader(training_script)
            jdata = update_deepmd_input(jdata)
            update_sel = UpdateSel()
            t_min_nbor_dist = update_sel.get_min_nbor_dist(
                jdata, update_sel.get_rcut(jdata)
            )

    _check_compress_type(graph)

    tf.constant(
        t_min_nbor_dist,
        name="train_attr/min_nbor_dist",
        dtype=GLOBAL_ENER_FLOAT_PRECISION,
    )
    jdata["model"]["compress"] = {}
    jdata["model"]["compress"]["model_file"] = input
    jdata["model"]["compress"]["min_nbor_dist"] = t_min_nbor_dist
    jdata["model"]["compress"]["table_config"] = [
        extrapolate,
        step,
        10 * step,
        int(frequency),
    ]
    jdata["training"]["save_ckpt"] = os.path.join("model-compression", "model.ckpt")
    jdata = update_deepmd_input(jdata)
    jdata = normalize(jdata)

    # check the descriptor info of the input file
    # move to the specific Descriptor class

    # stage 1: training or refining the model with tabulation
    log.info("\n\n")
    log.info("stage 1: compress the model")
    control_file = "compress.json"
    with open(control_file, "w") as fp:
        json.dump(jdata, fp, indent=4)
    try:
        train(
            INPUT=control_file,
            init_model=None,
            restart=None,
            init_frz_model=None,
            output=control_file,
            mpi_log=mpi_log,
            log_level=log_level,
            log_path=log_path,
            is_compress=True,
        )
    except GraphTooLargeError as e:
        raise RuntimeError(
            "The uniform step size of the tabulation's first table is %f, "
            "which is too small. This leads to a very large graph size, "
            "exceeding protobuf's limitation (2 GB). You should try to "
            "increase the step size." % step
        ) from e

    # reset the graph, otherwise the size limitation will be only 2 GB / 2 = 1 GB
    tf.reset_default_graph()

    # stage 2: freeze the model
    log.info("\n\n")
    log.info("stage 2: freeze the model")
    try:
        freeze(checkpoint_folder=checkpoint_folder, output=output, node_names=None)
    except GraphTooLargeError as e:
        raise RuntimeError(
            "The uniform step size of the tabulation's first table is %f, "
            "which is too small. This leads to a very large graph size, "
            "exceeding protobuf's limitation (2 GB). You should try to "
            "increase the step size." % step
        ) from e


def _check_compress_type(graph: tf.Graph):
    try:
        t_model_type = bytes.decode(get_tensor_by_name_from_graph(graph, "model_type"))
    except GraphWithoutTensorError as e:
        # Compatible with the upgraded model, which has no 'model_type' info
        t_model_type = None

    if t_model_type == "compressed_model":
        raise RuntimeError(
            "The input frozen model has already been compressed! Please do not compress the model repeatedly. "
        )
