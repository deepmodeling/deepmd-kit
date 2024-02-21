# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import tempfile

from deepmd.tf.entrypoints import (
    freeze,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.model.model import (
    Model,
)
from deepmd.tf.utils.errors import (
    GraphWithoutTensorError,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name_from_graph,
    load_graph_def,
)
from deepmd.tf.utils.sess import (
    run_sess,
)


def serialize_from_file(model_file: str) -> dict:
    """Serialize the model file to a dictionary.

    Parameters
    ----------
    model_file : str
        The model file to be serialized.

    Returns
    -------
    dict
        The serialized model file.
    """
    graph, graph_def = load_graph_def(model_file)
    t_jdata = get_tensor_by_name_from_graph(graph, "train_attr/training_script")
    jdata = json.loads(t_jdata)
    model = Model(**jdata["model"])
    # important! must be called before serialize
    model.init_variables(graph=graph, graph_def=graph_def)
    model_dict = model.serialize()
    data = {
        "backend": "TensorFlow",
        "model": model_dict,
    }
    # neighbor stat information
    try:
        t_min_nbor_dist = get_tensor_by_name_from_graph(
            graph, "train_attr/min_nbor_dist"
        )
    except GraphWithoutTensorError as e:
        pass
    else:
        data.setdefault("@variables", {})
        data["@variables"]["min_nbor_dist"] = t_min_nbor_dist
    return model_dict


def deserialize_to_file(data: dict, model_file: str) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    data : dict
        The dictionary to be deserialized.
    model_file : str
        The model file to be saved.
    """
    model = Model.deserialize(data["model"])
    with tf.Session() as sess:
        place_holders = {}
        for ii in ["coord", "box"]:
            place_holders[ii] = tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION, [None, None], name="t_" + ii
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
            freeze(checkpoint_folder=nt, output=model_file, node_names=None)
