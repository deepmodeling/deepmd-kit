# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import tempfile

from deepmd.entrypoints.freeze import (
    freeze,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    tf,
)
from deepmd.model.model import (
    Model,
)
from deepmd.utils.graph import (
    get_tensor_by_name_from_graph,
    load_graph_def,
)
from deepmd.utils.sess import (
    run_sess,
)


def convert_pb_to_dp(input_model: str, output_model: str):
    """Convert a frozen model to a native DP model.

    Parameters
    ----------
    input_model : str
        The input frozen model.
    output_model : str
        The output DP model.
    """
    graph, graph_def = load_graph_def(input_model)
    t_jdata = get_tensor_by_name_from_graph(graph, "train_attr/training_script")
    jdata = json.loads(t_jdata)
    model = Model(**jdata["model"])
    model.save_model(output_model, graph, graph_def)


def convert_dp_to_pb(input_model: str, output_model: str):
    """Convert a native DP model to a frozen model.

    Parameters
    ----------
    input_model : str
        The input DP model.
    output_model : str
        The output frozen model.
    """
    model = Model.load_model(input_model)
    with tf.Session() as sess:
        place_holders = {}
        for ii in ["coord", "box"]:
            place_holders[ii] = tf.placeholder(
                GLOBAL_NP_FLOAT_PRECISION, [None, None], name="t_" + ii
            )
        place_holders["type"] = tf.placeholder(tf.int32, [None], name="t_type")
        place_holders["natoms_vec"] = tf.placeholder(
            tf.int32, [model.get_ntypes() + 2], name="t_natoms"
        )
        place_holders["default_mesh"] = tf.placeholder(tf.int32, [None], name="t_mesh")
        # TODO: fparam, aparam

        model.build(
            place_holders["coord"],
            place_holders["type"],
            place_holders["natoms_vec"],
            place_holders["box"],
            place_holders["default_mesh"],
            place_holders,
            reuse=False,
        )
        init = tf.global_variables_initializer()
        run_sess(sess, init)
        saver = tf.train.Saver()
        with tempfile.TemporaryDirectory() as nt:
            saver.save(
                sess,
                os.path.join(nt, "model.ckpt"),
                global_step=0,
            )
            freeze(checkpoint_folder=nt, output=output_model, node_names=None)
