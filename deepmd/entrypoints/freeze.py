#!/usr/bin/env python3
"""Script for freezing TF trained graph so it can be used with LAMMPS and i-PI.

References
----------
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
"""

import json
import logging
from typing import (
    List,
    Optional,
    Union,
)

# load grad of force module
import deepmd.op  # noqa: F401
from deepmd.env import (
    FITTING_NET_PATTERN,
    REMOVE_SUFFIX_DICT,
    tf,
)
from deepmd.utils.graph import (
    get_pattern_nodes_from_graph_def,
)
from deepmd.utils.sess import (
    run_sess,
)

__all__ = ["freeze"]

log = logging.getLogger(__name__)


def _transfer_fitting_net_trainable_variables(sess, old_graph_def, raw_graph_def):
    old_pattern = FITTING_NET_PATTERN
    raw_pattern = (
        FITTING_NET_PATTERN.replace("idt", r"idt+_\d+")
        .replace("bias", r"bias+_\d+")
        .replace("matrix", r"matrix+_\d+")
    )
    old_graph_nodes = get_pattern_nodes_from_graph_def(old_graph_def, old_pattern)
    try:
        raw_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            raw_graph_def,  # The graph_def is used to retrieve the nodes
            [
                n + "_1" for n in old_graph_nodes
            ],  # The output node names are used to select the usefull nodes
        )
    except AssertionError:
        # if there's no additional nodes
        return old_graph_def

    raw_graph_nodes = get_pattern_nodes_from_graph_def(raw_graph_def, raw_pattern)
    for node in old_graph_def.node:
        if node.name not in old_graph_nodes.keys():
            continue
        tensor = tf.make_ndarray(raw_graph_nodes[node.name + "_1"])
        node.attr["value"].tensor.tensor_content = tensor.tostring()
    return old_graph_def


def _remove_fitting_net_suffix(output_graph_def, out_suffix):
    """Remove fitting net suffix for multi-task mode.

    Parameters
    ----------
    output_graph_def : tf.GraphDef
        The output graph to remove suffix.
    out_suffix : str
        The suffix to remove.
    """

    def change_name(name, suffix):
        if suffix in name:
            for item in REMOVE_SUFFIX_DICT:
                if item.format(suffix) in name:
                    name = name.replace(item.format(suffix), REMOVE_SUFFIX_DICT[item])
                    break
            assert suffix not in name, "fitting net name illegal!"
        return name

    for node in output_graph_def.node:
        if out_suffix in node.name:
            node.name = change_name(node.name, out_suffix)
        for idx in range(len(node.input)):
            if out_suffix in node.input[idx]:
                node.input[idx] = change_name(node.input[idx], out_suffix)
        attr_list = node.attr["_class"].list.s
        for idx in range(len(attr_list)):
            if out_suffix in bytes.decode(attr_list[idx]):
                attr_list[idx] = bytes(
                    change_name(bytes.decode(attr_list[idx]), out_suffix),
                    encoding="utf8",
                )
    return output_graph_def


def _modify_model_suffix(output_graph_def, out_suffix, freeze_type):
    """Modify model suffix in graph nodes for multi-task mode, including fitting net, model attr and training script.

    Parameters
    ----------
    output_graph_def : tf.GraphDef
        The output graph to remove suffix.
    out_suffix : str
        The suffix to remove.
    freeze_type : str
        The model type to freeze.
    """
    output_graph_def = _remove_fitting_net_suffix(output_graph_def, out_suffix)
    for node in output_graph_def.node:
        if "model_attr/model_type" in node.name:
            node.attr["value"].tensor.string_val[0] = bytes(
                freeze_type, encoding="utf8"
            )
        # change the input script for frozen model
        elif "train_attr/training_script" in node.name:
            jdata = json.loads(node.attr["value"].tensor.string_val[0])
            # fitting net
            assert out_suffix in jdata["model"]["fitting_net_dict"]
            jdata["model"]["fitting_net"] = jdata["model"].pop("fitting_net_dict")[
                out_suffix
            ]
            # data systems
            systems = jdata["training"].pop("data_dict")
            if out_suffix in systems:
                jdata["training"]["training_data"] = systems[out_suffix][
                    "training_data"
                ]
                if "validation_data" in systems[out_suffix]:
                    jdata["training"]["validation_data"] = systems[out_suffix][
                        "validation_data"
                    ]
            else:
                jdata["training"]["training_data"] = {}
                log.warning(
                    "The fitting net {} has no training data in input script, resulting in "
                    "untrained frozen model, and cannot be compressed directly! ".format(
                        out_suffix
                    )
                )
            # loss
            if "loss_dict" in jdata:
                loss_dict = jdata.pop("loss_dict")
                if out_suffix in loss_dict:
                    jdata["loss"] = loss_dict[out_suffix]
            # learning_rate
            if "learning_rate_dict" in jdata:
                learning_rate_dict = jdata.pop("learning_rate_dict")
                if out_suffix in learning_rate_dict:
                    jdata["learning_rate"] = learning_rate_dict[out_suffix]
            # fitting weight
            if "fitting_weight" in jdata["training"]:
                jdata["training"].pop("fitting_weight")
            node.attr["value"].tensor.string_val[0] = bytes(
                json.dumps(jdata), encoding="utf8"
            )
    return output_graph_def


def _make_node_names(
    model_type: str,
    modifier_type: Optional[str] = None,
    out_suffix: str = "",
    node_names: Optional[Union[str, list]] = None,
) -> List[str]:
    """Get node names based on model type.

    Parameters
    ----------
    model_type : str
        str type of model
    modifier_type : Optional[str], optional
        modifier type if any, by default None
    out_suffix : str
        suffix for output nodes
    node_names : Optional[str], optional
        Names of nodes to output, by default None.

    Returns
    -------
    List[str]
        list with all node names to freeze

    Raises
    ------
    RuntimeError
        if unknown model type
    """
    nodes = [
        "model_type",
        "descrpt_attr/rcut",
        "descrpt_attr/ntypes",
        "model_attr/tmap",
        "model_attr/model_type",
        "model_attr/model_version",
        "train_attr/min_nbor_dist",
        "train_attr/training_script",
    ]

    if model_type == "ener":
        nodes += [
            "o_energy",
            "o_force",
            "o_virial",
            "o_atom_energy",
            "o_atom_virial",
            "spin_attr/ntypes_spin",
            "fitting_attr/dfparam",
            "fitting_attr/daparam",
        ]
    elif model_type == "dos":
        nodes += [
            "o_dos",
            "fitting_attr/numb_dos",
            "fitting_attr/dfparam",
            "fitting_attr/daparam",
        ]
    elif model_type == "wfc":
        nodes += [
            "o_wfc",
            "model_attr/sel_type",
            "model_attr/output_dim",
        ]
    elif model_type == "dipole":
        nodes += [
            "o_dipole",
            "o_global_dipole",
            "o_force",
            "o_virial",
            "o_atom_virial",
            "o_rmat",
            "o_rmat_deriv",
            "o_nlist",
            "o_rij",
            "descrpt_attr/sel",
            "descrpt_attr/ndescrpt",
            "model_attr/sel_type",
            "model_attr/output_dim",
        ]
    elif model_type == "polar":
        nodes += [
            "o_polar",
            "o_global_polar",
            "o_force",
            "o_virial",
            "o_atom_virial",
            "model_attr/sel_type",
            "model_attr/output_dim",
        ]
    elif model_type == "global_polar":
        nodes += [
            "o_global_polar",
            "model_attr/sel_type",
            "model_attr/output_dim",
        ]
    elif model_type == "multi_task":
        assert (
            node_names is not None
        ), "node_names must be defined in multi-task united model! "
    else:
        raise RuntimeError(f"unknown model type {model_type}")
    if modifier_type == "dipole_charge":
        nodes += [
            "modifier_attr/type",
            "modifier_attr/mdl_name",
            "modifier_attr/mdl_charge_map",
            "modifier_attr/sys_charge_map",
            "modifier_attr/ewald_h",
            "modifier_attr/ewald_beta",
            "dipole_charge/model_type",
            "dipole_charge/descrpt_attr/rcut",
            "dipole_charge/descrpt_attr/ntypes",
            "dipole_charge/model_attr/tmap",
            "dipole_charge/model_attr/model_type",
            "dipole_charge/model_attr/model_version",
            "o_dm_force",
            "dipole_charge/model_attr/sel_type",
            "dipole_charge/o_dipole",
            "dipole_charge/model_attr/output_dim",
            "o_dm_virial",
            "o_dm_av",
        ]
    if node_names is not None:
        if isinstance(node_names, str):
            nodes = node_names.split(",")
        elif isinstance(node_names, list):
            nodes = node_names
        else:
            raise RuntimeError(f"unknown node names type {type(node_names)}")
    if out_suffix != "":
        for ind in range(len(nodes)):
            if (
                (
                    nodes[ind][:2] == "o_"
                    and nodes[ind] not in ["o_rmat", "o_rmat_deriv", "o_nlist", "o_rij"]
                )
                or nodes[ind] == "model_attr/sel_type"
                or nodes[ind] == "model_attr/output_dim"
            ):
                nodes[ind] += f"_{out_suffix}"
            elif "fitting_attr" in nodes[ind]:
                content = nodes[ind].split("/")[1]
                nodes[ind] = f"fitting_attr_{out_suffix}/{content}"
    return nodes


def freeze_graph(
    model_file: str,
    output: str,
    # sess,
    # input_graph,
    # input_node,
    # freeze_type,
    # modifier,
    # out_graph_name,
    # node_names=None,
    # out_suffix="",
):
    """Freeze the single graph with chosen out_suffix.

    Parameters
    ----------
    model_file : str
        Location of the *.pdparams file
    output : str
        output file name
    """
    import paddle

    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(
        model_file,
        load_prefix="load",
        default_tf_graph=False,
    )
    dp.model.eval()
    from paddle.static import (
        InputSpec,
    )

    st_model = paddle.jit.to_static(
        dp.model,
        input_spec=[
            InputSpec(shape=[None], dtype="float64"),  # coord_
            InputSpec(shape=[None], dtype="int32"),  # atype_
            InputSpec(shape=[2 + dp.model.descrpt.ntypes], dtype="int32"),  # natoms
            InputSpec(shape=[None], dtype="float64"),  # box
            InputSpec(shape=[6], dtype="int32"),  # mesh
            {
                "box": InputSpec(shape=[None], dtype="float64"),
            },
            "",
            False,
        ],
    )
    for name, param in st_model.named_buffers():
        print(
            f"[{name}, {param.dtype}, {param.shape}] generated name in static_model is: {param.name}"
        )
    # skip pruning for program so as to keep buffers into files
    skip_prune_program = True
    print(f"==>> Set skip_prune_program = {skip_prune_program}")
    paddle.jit.save(st_model, output, skip_prune_program=skip_prune_program)
    print(f"Infernece model has been saved to: {output}")


def freeze_graph_multi(
    sess,
    input_graph,
    input_node,
    modifier,
    out_graph_name,
    node_names,
    united_model: bool = False,
):
    """Freeze multiple graphs for multi-task model.

    Parameters
    ----------
    sess : tf.Session
        The default session.
    input_graph : tf.GraphDef
        The input graph_def stored from the checkpoint.
    input_node : List[str]
        The expected nodes to freeze.
    modifier : Optional[str], optional
        Modifier type if any, by default None.
    out_graph_name : str
        The output graph.
    node_names : Optional[str], optional
        Names of nodes to output, by default None.
    united_model : bool
        If freeze all nodes into one unit model
    """
    input_script = json.loads(
        run_sess(sess, "train_attr/training_script:0", feed_dict={})
    )
    assert (
        "model" in input_script.keys() and "fitting_net_dict" in input_script["model"]
    )
    if not united_model:
        for fitting_key in input_script["model"]["fitting_net_dict"]:
            fitting_type = input_script["model"]["fitting_net_dict"][fitting_key][
                "type"
            ]
            if out_graph_name[-3:] == ".pb":
                output_graph_item = out_graph_name[:-3] + f"_{fitting_key}.pb"
            else:
                output_graph_item = out_graph_name + f"_{fitting_key}"
            freeze_graph(
                sess,
                input_graph,
                input_node,
                fitting_type,
                modifier,
                output_graph_item,
                node_names,
                out_suffix=fitting_key,
            )
    else:
        node_multi = []
        for fitting_key in input_script["model"]["fitting_net_dict"]:
            fitting_type = input_script["model"]["fitting_net_dict"][fitting_key][
                "type"
            ]
            node_multi += _make_node_names(
                fitting_type, modifier, out_suffix=fitting_key
            )
        node_multi = list(set(node_multi))
        if node_names is not None:
            node_multi = node_names
        freeze_graph(
            sess,
            input_graph,
            input_node,
            "multi_task",
            modifier,
            out_graph_name,
            node_multi,
        )


def freeze(
    *,
    input_file: str,
    output: str,
    **kwargs,
):
    """Freeze the graph in supplied folder.

    Parameters
    ----------
    input_file : str
        location of the *.pdparams file
    output : str
        output file name
    **kwargs
        other arguments
    """
    freeze_graph(
        input_file,
        output,
    )
