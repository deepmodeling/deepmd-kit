# SPDX-License-Identifier: LGPL-3.0-or-later
import re
from typing import (
    Dict,
    Tuple,
)

import numpy as np

from deepmd.tf.env import (
    ATTENTION_LAYER_PATTERN,
    EMBEDDING_NET_PATTERN,
    FITTING_NET_PATTERN,
    TYPE_EMBEDDING_PATTERN,
    tf,
)
from deepmd.tf.utils.errors import (
    GraphWithoutTensorError,
)
from deepmd.tf.utils.sess import (
    run_sess,
)


# TODO (JZ): I think in this file we can merge some duplicated lines into one method...
def load_graph_def(model_file: str) -> Tuple[tf.Graph, tf.GraphDef]:
    """Load graph as well as the graph_def from the frozen model(model_file).

    Parameters
    ----------
    model_file : str
        The input frozen model path

    Returns
    -------
    tf.Graph
        The graph loaded from the frozen model
    tf.GraphDef
        The graph_def loaded from the frozen model
    """
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph, graph_def


def get_tensor_by_name_from_graph(graph: tf.Graph, tensor_name: str) -> tf.Tensor:
    """Load tensor value from the given tf.Graph object.

    Parameters
    ----------
    graph : tf.Graph
        The input TensorFlow graph
    tensor_name : str
        Indicates which tensor which will be loaded from the frozen model

    Returns
    -------
    tf.Tensor
        The tensor which was loaded from the frozen model

    Raises
    ------
    GraphWithoutTensorError
        Whether the tensor_name is within the frozen model
    """
    try:
        tensor = graph.get_tensor_by_name(tensor_name + ":0")
    except KeyError as e:
        raise GraphWithoutTensorError() from e
    with tf.Session(graph=graph) as sess:
        tensor = run_sess(sess, tensor)
    return tensor


def get_tensor_by_name(model_file: str, tensor_name: str) -> tf.Tensor:
    """Load tensor value from the frozen model(model_file).

    Parameters
    ----------
    model_file : str
        The input frozen model path
    tensor_name : str
        Indicates which tensor which will be loaded from the frozen model

    Returns
    -------
    tf.Tensor
        The tensor which was loaded from the frozen model

    Raises
    ------
    GraphWithoutTensorError
        Whether the tensor_name is within the frozen model
    """
    graph, _ = load_graph_def(model_file)
    return get_tensor_by_name_from_graph(graph, tensor_name)


def get_tensor_by_type(node, data_type: np.dtype) -> tf.Tensor:
    """Get the tensor value within the given node according to the input data_type.

    Parameters
    ----------
    node
        The given tensorflow graph node
    data_type
        The data type of the node

    Returns
    -------
    tf.Tensor
        The tensor value of the given node
    """
    if data_type == np.float64:
        tensor = np.array(node.double_val)
    elif data_type == np.float32:
        tensor = np.array(node.float_val)
    else:
        raise RuntimeError("model compression does not support the half precision")
    return tensor


def get_pattern_nodes_from_graph_def(graph_def: tf.GraphDef, pattern: str) -> Dict:
    """Get the pattern nodes with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    pattern
        The node pattern within the graph_def

    Returns
    -------
    Dict
        The fitting net nodes within the given tf.GraphDef object
    """
    nodes = {}
    pattern = re.compile(pattern)
    for node in graph_def.node:
        if re.fullmatch(pattern, node.name) is not None:
            nodes[node.name] = node.attr["value"].tensor
    return nodes


def get_embedding_net_nodes_from_graph_def(
    graph_def: tf.GraphDef, suffix: str = ""
) -> Dict:
    """Get the embedding net nodes with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix : str, optional
        The scope suffix

    Returns
    -------
    Dict
        The embedding net nodes within the given tf.GraphDef object
    """
    # embedding_net_pattern = f"filter_type_\d+{suffix}/matrix_\d+_\d+|filter_type_\d+{suffix}/bias_\d+_\d+|filter_type_\d+{suffix}/idt_\d+_\d+|filter_type_all{suffix}/matrix_\d+_\d+|filter_type_all{suffix}/matrix_\d+_\d+_\d+|filter_type_all{suffix}/bias_\d+_\d+|filter_type_all{suffix}/bias_\d+_\d+_\d+|filter_type_all{suffix}/idt_\d+_\d+"
    if suffix != "":
        embedding_net_pattern = (
            EMBEDDING_NET_PATTERN.replace("/(idt)", suffix + "/(idt)")
            .replace("/(bias)", suffix + "/(bias)")
            .replace("/(matrix)", suffix + "/(matrix)")
        )
    else:
        embedding_net_pattern = EMBEDDING_NET_PATTERN

    embedding_net_nodes = get_pattern_nodes_from_graph_def(
        graph_def, embedding_net_pattern
    )
    return embedding_net_nodes


def get_embedding_net_nodes(model_file: str, suffix: str = "") -> Dict:
    """Get the embedding net nodes with the given frozen model(model_file).

    Parameters
    ----------
    model_file
        The input frozen model path
    suffix : str, optional
        The suffix of the scope

    Returns
    -------
    Dict
        The embedding net nodes with the given frozen model
    """
    _, graph_def = load_graph_def(model_file)
    return get_embedding_net_nodes_from_graph_def(graph_def, suffix=suffix)


def get_embedding_net_variables_from_graph_def(
    graph_def: tf.GraphDef, suffix: str = ""
) -> Dict:
    """Get the embedding net variables with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix : str, optional
        The suffix of the scope

    Returns
    -------
    Dict
        The embedding net variables within the given tf.GraphDef object
    """
    embedding_net_variables = {}
    embedding_net_nodes = get_embedding_net_nodes_from_graph_def(
        graph_def, suffix=suffix
    )
    for item in embedding_net_nodes:
        node = embedding_net_nodes[item]
        dtype = tf.as_dtype(node.dtype).as_numpy_dtype
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(
                node.tensor_content, dtype=tf.as_dtype(node.dtype).as_numpy_dtype
            )
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        embedding_net_variables[item] = np.reshape(tensor_value, tensor_shape)
    return embedding_net_variables


def get_extra_embedding_net_suffix(type_one_side: bool):
    """Get the extra embedding net suffix according to the value of type_one_side.

    Parameters
    ----------
    type_one_side
        The value of type_one_side

    Returns
    -------
    str
        The extra embedding net suffix
    """
    if type_one_side:
        extra_suffix = "_one_side_ebd"
    else:
        extra_suffix = "_two_side_ebd"
    return extra_suffix


def get_variables_from_graph_def_as_numpy_array(graph_def: tf.GraphDef, pattern: str):
    """Get variables from the given tf.GraphDef object, with numpy array returns.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    pattern : str
        The name of variable

    Returns
    -------
    np.ndarray
        The numpy array of the variable
    """
    node = get_pattern_nodes_from_graph_def(graph_def, pattern)[pattern]
    dtype = tf.as_dtype(node.dtype).as_numpy_dtype
    tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
    if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
        tensor_value = np.frombuffer(
            node.tensor_content,
            dtype=tf.as_dtype(node.dtype).as_numpy_dtype,
        )
    else:
        tensor_value = get_tensor_by_type(node, dtype)
    return np.reshape(tensor_value, tensor_shape)


def get_extra_embedding_net_variables_from_graph_def(
    graph_def: tf.GraphDef, suffix: str, extra_suffix: str, layer_size: int
):
    """Get extra embedding net variables from the given tf.GraphDef object.
    The "extra embedding net" means the embedding net with only type embeddings input,
    which occurs in "se_atten_v2" and "se_a_ebd_v2" descriptor.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix : str
        The "common" suffix in the descriptor
    extra_suffix : str
        This value depends on the value of "type_one_side".
        It should always be "_one_side_ebd" or "_two_side_ebd"
    layer_size : int
        The layer size of the embedding net

    Returns
    -------
    Dict
        The extra embedding net variables within the given tf.GraphDef object
    """
    extra_embedding_net_variables = {}
    for i in range(1, layer_size + 1):
        matrix_pattern = f"filter_type_all{suffix}/matrix_{i}{extra_suffix}"
        extra_embedding_net_variables[
            matrix_pattern
        ] = get_variables_from_graph_def_as_numpy_array(graph_def, matrix_pattern)
        bias_pattern = f"filter_type_all{suffix}/bias_{i}{extra_suffix}"
        extra_embedding_net_variables[
            bias_pattern
        ] = get_variables_from_graph_def_as_numpy_array(graph_def, bias_pattern)
    return extra_embedding_net_variables


def get_embedding_net_variables(model_file: str, suffix: str = "") -> Dict:
    """Get the embedding net variables with the given frozen model(model_file).

    Parameters
    ----------
    model_file
        The input frozen model path
    suffix : str, optional
        The suffix of the scope

    Returns
    -------
    Dict
        The embedding net variables within the given frozen model
    """
    _, graph_def = load_graph_def(model_file)
    return get_embedding_net_variables_from_graph_def(graph_def, suffix=suffix)


def get_fitting_net_nodes_from_graph_def(
    graph_def: tf.GraphDef, suffix: str = ""
) -> Dict:
    """Get the fitting net nodes with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix
        suffix of the scope

    Returns
    -------
    Dict
        The fitting net nodes within the given tf.GraphDef object
    """
    if suffix != "":
        fitting_net_pattern = (
            FITTING_NET_PATTERN.replace("/(idt)", suffix + "/(idt)")
            .replace("/(bias)", suffix + "/(bias)")
            .replace("/(matrix)", suffix + "/(matrix)")
        )
    else:
        fitting_net_pattern = FITTING_NET_PATTERN
    fitting_net_nodes = get_pattern_nodes_from_graph_def(graph_def, fitting_net_pattern)
    for key in fitting_net_nodes.keys():
        assert (
            key.find("bias") > 0 or key.find("matrix") > 0 or key.find("idt") > 0
        ), "currently, only support weight matrix, bias and idt at the model compression process!"
    return fitting_net_nodes


def get_fitting_net_nodes(model_file: str) -> Dict:
    """Get the fitting net nodes with the given frozen model(model_file).

    Parameters
    ----------
    model_file
        The input frozen model path

    Returns
    -------
    Dict
        The fitting net nodes with the given frozen model
    """
    _, graph_def = load_graph_def(model_file)
    return get_fitting_net_nodes_from_graph_def(graph_def)


def get_fitting_net_variables_from_graph_def(
    graph_def: tf.GraphDef, suffix: str = ""
) -> Dict:
    """Get the fitting net variables with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix
        suffix of the scope

    Returns
    -------
    Dict
        The fitting net variables within the given tf.GraphDef object
    """
    fitting_net_variables = {}
    fitting_net_nodes = get_fitting_net_nodes_from_graph_def(graph_def, suffix=suffix)
    for item in fitting_net_nodes:
        node = fitting_net_nodes[item]
        dtype = tf.as_dtype(node.dtype).as_numpy_dtype
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(
                node.tensor_content, dtype=tf.as_dtype(node.dtype).as_numpy_dtype
            )
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        fitting_net_variables[item] = np.reshape(tensor_value, tensor_shape)
    return fitting_net_variables


def get_fitting_net_variables(model_file: str, suffix: str = "") -> Dict:
    """Get the fitting net variables with the given frozen model(model_file).

    Parameters
    ----------
    model_file
        The input frozen model path
    suffix
        suffix of the scope

    Returns
    -------
    Dict
        The fitting net variables within the given frozen model
    """
    _, graph_def = load_graph_def(model_file)
    return get_fitting_net_variables_from_graph_def(graph_def, suffix=suffix)


def get_type_embedding_net_nodes_from_graph_def(
    graph_def: tf.GraphDef, suffix: str = ""
) -> Dict:
    """Get the type embedding net nodes with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix : str, optional
        The scope suffix

    Returns
    -------
    Dict
        The type embedding net nodes within the given tf.GraphDef object
    """
    if suffix != "":
        type_embedding_net_pattern = (
            TYPE_EMBEDDING_PATTERN.replace("/idt", suffix + "/idt")
            .replace("/bias", suffix + "/bias")
            .replace("/matrix", suffix + "/matrix")
        )
    else:
        type_embedding_net_pattern = TYPE_EMBEDDING_PATTERN

    type_embedding_net_nodes = get_pattern_nodes_from_graph_def(
        graph_def, type_embedding_net_pattern
    )
    return type_embedding_net_nodes


def get_type_embedding_net_variables_from_graph_def(
    graph_def: tf.GraphDef, suffix: str = ""
) -> Dict:
    """Get the type embedding net variables with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def : tf.GraphDef
        The input tf.GraphDef object
    suffix : str, optional
        The suffix of the scope

    Returns
    -------
    Dict
        The embedding net variables within the given tf.GraphDef object
    """
    type_embedding_net_variables = {}
    type_embedding_net_nodes = get_type_embedding_net_nodes_from_graph_def(
        graph_def, suffix=suffix
    )
    for item in type_embedding_net_nodes:
        node = type_embedding_net_nodes[item]
        dtype = tf.as_dtype(node.dtype).as_numpy_dtype
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(
                node.tensor_content, dtype=tf.as_dtype(node.dtype).as_numpy_dtype
            )
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        type_embedding_net_variables[item] = np.reshape(tensor_value, tensor_shape)
    return type_embedding_net_variables


def get_attention_layer_nodes_from_graph_def(
    graph_def: tf.GraphDef, suffix: str = ""
) -> Dict:
    """Get the attention layer nodes with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix : str, optional
        The scope suffix

    Returns
    -------
    Dict
        The attention layer nodes within the given tf.GraphDef object
    """
    if suffix != "":
        attention_layer_pattern = (
            ATTENTION_LAYER_PATTERN.replace("/c_query", suffix + "/c_query")
            .replace("/c_key", suffix + "/c_key")
            .replace("/c_value", suffix + "/c_value")
            .replace("/c_out", suffix + "/c_out")
            .replace("/layer_normalization", suffix + "/layer_normalization")
        )
    else:
        attention_layer_pattern = ATTENTION_LAYER_PATTERN

    attention_layer_nodes = get_pattern_nodes_from_graph_def(
        graph_def, attention_layer_pattern
    )
    return attention_layer_nodes


def get_attention_layer_variables_from_graph_def(
    graph_def: tf.GraphDef, suffix: str = ""
) -> Dict:
    """Get the attention layer variables with the given tf.GraphDef object.

    Parameters
    ----------
    graph_def : tf.GraphDef
        The input tf.GraphDef object
    suffix : str, optional
        The suffix of the scope

    Returns
    -------
    Dict
        The attention layer variables within the given tf.GraphDef object
    """
    attention_layer_variables = {}
    attention_layer_net_nodes = get_attention_layer_nodes_from_graph_def(
        graph_def, suffix=suffix
    )
    for item in attention_layer_net_nodes:
        node = attention_layer_net_nodes[item]
        dtype = tf.as_dtype(node.dtype).as_numpy_dtype
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(
                node.tensor_content, dtype=tf.as_dtype(node.dtype).as_numpy_dtype
            )
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        attention_layer_variables[item] = np.reshape(tensor_value, tensor_shape)
    return attention_layer_variables
