#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.nvnmd.utils.fio import (
    FioDic,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name_from_graph,
)


def filter_tensorVariableList(tensorVariableList) -> dict:
    r"""Get the name of variable for NVNMD.

    | :code:`train_attr/min_nbor_dist`
    | :code:`descrpt_attr/t_avg:0`
    | :code:`descrpt_attr/t_std:0`
    | :code:`type_embed_net/matrix_{layer l}:0`
    | :code:`type_embed_net/bias_{layer l}:0`

        version 0:
        | :code:`filter_type_{atom i}/matrix_{layer l}_{atomj}:0`
    | :code:`filter_type_{atom i}/bias_{layer l}_{atomj}:0`
    | :code:`layer_{layer l}_type_{atom i}/matrix:0`
    | :code:`layer_{layer l}_type_{atom i}/bias:0`
    | :code:`final_layer_type_{atom i}/matrix:0`
    | :code:`final_layer_type_{atom i}/bias:0`

        version 1:
        | :code:`filter_type_all/matrix_{layer l}:0`
    | :code:`filter_type_all/bias_{layer l}:0`
    | :code:`filter_type_all/matrix_{layer l}_two_side_ebd:0`
    | :code:`filter_type_all/bias_{layer l}_two_side_ebd:0`
    | :code:`layer_{layer l}/matrix:0`
    | :code:`layer_{layer l}/bias:0`
    | :code:`final_layer/matrix:0`
    | :code:`final_layer/bias:0`
    """
    nameList = [tv.name for tv in tensorVariableList]
    nameList = [name.replace(":0", "") for name in nameList]
    nameList = [name.replace("/", ".") for name in nameList]

    dic_name_tv = {}
    for ii in range(len(nameList)):
        name = nameList[ii]
        tv = tensorVariableList[ii]
        p1 = name.startswith("descrpt_attr")
        p1 = p1 or name.startswith("type_embed_net")
        p1 = p1 or name.startswith("filter_type_")
        p1 = p1 or name.startswith("layer_")
        p1 = p1 or name.startswith("final_layer")
        p1 = p1 or name.endswith("t_bias_atom_e")
        p2 = "Adam" not in name
        p3 = "XXX" not in name
        if p1 and p2 and p3:
            dic_name_tv[name] = tv
    return dic_name_tv


def save_weight(sess, file_name: str = "nvnmd/weight.npy"):
    r"""Save the dictionary of weight to a npy file."""
    tvs = tf.global_variables()
    dic_key_tv = filter_tensorVariableList(tvs)
    dic_key_value = {}
    for key in dic_key_tv.keys():
        value = sess.run(dic_key_tv[key])
        dic_key_value[key] = value
    namelist = [n.name for n in tf.get_default_graph().as_graph_def().node]
    if "train_attr/min_nbor_dist" in namelist:
        min_dist = get_tensor_by_name_from_graph(
            tf.get_default_graph(), "train_attr/min_nbor_dist"
        )
    else:
        min_dist = 0.0
    dic_key_value["train_attr.min_nbor_dist"] = min_dist
    dic_key_value["t_bias_atom_e"] = dic_key_value["fitting_attr.t_bias_atom_e"]
    FioDic().save(file_name, dic_key_value)
