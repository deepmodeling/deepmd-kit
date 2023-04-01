#!/usr/bin/env python3

from deepmd.env import (
    tf,
)
from deepmd.nvnmd.utils.fio import (
    FioDic,
)
from deepmd.utils.graph import (
    get_tensor_by_name_from_graph,
)


def filter_tensorVariableList(tensorVariableList) -> dict:
    r"""Get the name of variable for NVNMD.

    | :code:`train_attr/min_nbor_dist`
    | :code:`descrpt_attr/t_avg:0`
    | :code:`descrpt_attr/t_std:0`
    | :code:`filter_type_{atom i}/matrix_{layer l}_{atomj}:0`
    | :code:`filter_type_{atom i}/bias_{layer l}_{atomj}:0`
    | :code:`layer_{layer l}_type_{atom i}/matrix:0`
    | :code:`layer_{layer l}_type_{atom i}/bias:0`
    | :code:`final_layer_type_{atom i}/matrix:0`
    | :code:`final_layer_type_{atom i}/bias:0`
    """
    nameList = [tv.name for tv in tensorVariableList]
    nameList = [name.replace(":0", "") for name in nameList]
    nameList = [name.replace("/", ".") for name in nameList]

    dic_name_tv = {}
    for ii in range(len(nameList)):
        name = nameList[ii]
        tv = tensorVariableList[ii]
        p1 = name.startswith("descrpt_attr")
        p1 = p1 or name.startswith("filter_type_")
        p1 = p1 or name.startswith("layer_")
        p1 = p1 or name.startswith("final_layer_type_")
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
    FioDic().save(file_name, dic_key_value)
