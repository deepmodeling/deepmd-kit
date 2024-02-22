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
        The serialized model data.
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
        "tf_version": tf.__version__,
        "model": model_dict,
        "model_def_script": jdata["model"],
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
    return data


def deserialize_to_file(model_file: str, data: dict) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    model_file : str
        The model file to be saved.
    data : dict
        The dictionary to be deserialized.
    """
    model = Model.deserialize(data["model"])
    with tf.Graph().as_default() as graph, tf.Session(graph=graph) as sess:
        place_holders = {}
        for ii in ["coord", "box"]:
            place_holders[ii] = tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION, [None], name="t_" + ii
            )
        place_holders["type"] = tf.placeholder(tf.int32, [None], name="t_type")
        place_holders["natoms_vec"] = tf.placeholder(
            tf.int32, [model.get_ntypes() + 2], name="t_natoms"
        )
        place_holders["default_mesh"] = tf.placeholder(tf.int32, [None], name="t_mesh")
        inputs = {}
        # fparam, aparam
        if model.get_numb_fparam() > 0:
            inputs["fparam"] = tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION,
                [None, model.get_numb_fparam()],
                name="t_fparam",
            )
        if model.get_numb_aparam() > 0:
            inputs["aparam"] = tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION,
                [None, model.get_numb_aparam()],
                name="t_aparam",
            )
        model.build(
            place_holders["coord"],
            place_holders["type"],
            place_holders["natoms_vec"],
            place_holders["box"],
            place_holders["default_mesh"],
            inputs,
            reuse=False,
        )
        init = tf.global_variables_initializer()
        tf.constant(
            json.dumps({"model": data["model_def_script"]}, separators=(",", ":")),
            name="train_attr/training_script",
            dtype=tf.string,
        )
        if "min_nbor_dist" in data.get("@variables", {}):
            tf.constant(
                data["@variables"]["min_nbor_dist"],
                name="train_attr/min_nbor_dist",
                dtype=GLOBAL_TF_FLOAT_PRECISION,
            )
        run_sess(sess, init)
        saver = tf.train.Saver()
        with tempfile.TemporaryDirectory() as nt:
            saver.save(
                sess,
                os.path.join(nt, "model.ckpt"),
                global_step=0,
            )
            freeze(checkpoint_folder=nt, output=model_file, node_names=None)
