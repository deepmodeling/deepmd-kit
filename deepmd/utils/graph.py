import re
import numpy as np
from typing import Tuple, Dict
from deepmd.env import tf, EMBEDDING_NET_PATTERN, FITTING_NET_PATTERN
from deepmd.utils.sess import run_sess
from deepmd.utils.errors import GraphWithoutTensorError

def load_graph_def(model_file: str) -> Tuple[tf.Graph, tf.GraphDef]:
    """
    Load graph as well as the graph_def from the frozen model(model_file)

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
        tf.import_graph_def(graph_def, name = "")
    return graph, graph_def


def get_tensor_by_name_from_graph(graph: tf.Graph,
                                  tensor_name: str) -> tf.Tensor:
    """
    Load tensor value from the given tf.Graph object

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
        tensor = graph.get_tensor_by_name(tensor_name + ':0')
    except KeyError as e:
        raise GraphWithoutTensorError() from e
    with tf.Session(graph=graph) as sess:
        tensor = run_sess(sess, tensor)
    return tensor


def get_tensor_by_name(model_file: str,
                       tensor_name: str) -> tf.Tensor:
    """
    Load tensor value from the frozen model(model_file)

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


def get_tensor_by_type(node,
                       data_type : np.dtype) -> tf.Tensor:
    """
    Get the tensor value within the given node according to the input data_type

    Parameters
    ----------
    node
        The given tensorflow graph node
    data_type
        The data type of the node
    
    Returns
    ----------
    tf.Tensor
        The tensor value of the given node
    """
    if data_type == np.float64:
        tensor = np.array(node.double_val)
    elif data_type == np.float32:
        tensor = np.array(node.float_val)
    else:
        raise RuntimeError('model compression does not support the half precision')
    return tensor


def get_pattern_nodes_from_graph_def(graph_def: tf.GraphDef, pattern: str) -> Dict:
    """
    Get the pattern nodes with the given tf.GraphDef object

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    pattern
        The node pattern within the graph_def
    
    Returns
    ----------
    Dict
        The fitting net nodes within the given tf.GraphDef object
    """
    nodes = {}
    pattern = re.compile(pattern)
    for node in graph_def.node:
        if re.fullmatch(pattern, node.name) != None:
            nodes[node.name] = node.attr["value"].tensor
    return nodes


def get_embedding_net_nodes_from_graph_def(graph_def: tf.GraphDef, suffix: str = "") -> Dict:
    """
    Get the embedding net nodes with the given tf.GraphDef object

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix : str, optional
        The scope suffix
    
    Returns
    ----------
    Dict
        The embedding net nodes within the given tf.GraphDef object
    """
    # embedding_net_pattern = f"filter_type_\d+{suffix}/matrix_\d+_\d+|filter_type_\d+{suffix}/bias_\d+_\d+|filter_type_\d+{suffix}/idt_\d+_\d+|filter_type_all{suffix}/matrix_\d+_\d+|filter_type_all{suffix}/matrix_\d+_\d+_\d+|filter_type_all{suffix}/bias_\d+_\d+|filter_type_all{suffix}/bias_\d+_\d+_\d+|filter_type_all{suffix}/idt_\d+_\d+"
    if suffix != "":
        embedding_net_pattern = EMBEDDING_NET_PATTERN\
            .replace('/idt',    suffix + '/idt')\
            .replace('/bias',   suffix + '/bias')\
            .replace('/matrix', suffix + '/matrix')
    else:
        embedding_net_pattern = EMBEDDING_NET_PATTERN

    embedding_net_nodes = get_pattern_nodes_from_graph_def(graph_def, embedding_net_pattern)
    for key in embedding_net_nodes.keys():
        assert key.find('bias') > 0 or key.find(
            'matrix') > 0, "currently, only support weight matrix and bias matrix at the tabulation op!"
    return embedding_net_nodes


def get_embedding_net_nodes(model_file: str, suffix: str = "") -> Dict:
    """
    Get the embedding net nodes with the given frozen model(model_file)

    Parameters
    ----------
    model_file
        The input frozen model path
    suffix : str, optional
        The suffix of the scope
   
    Returns
    ----------
    Dict
        The embedding net nodes with the given frozen model
    """
    _, graph_def = load_graph_def(model_file)
    return get_embedding_net_nodes_from_graph_def(graph_def, suffix=suffix)


def get_embedding_net_variables_from_graph_def(graph_def : tf.GraphDef, suffix: str = "") -> Dict:
    """
    Get the embedding net variables with the given tf.GraphDef object

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    suffix : str, optional
        The suffix of the scope
    
    Returns
    ----------
    Dict
        The embedding net variables within the given tf.GraphDef object 
    """
    embedding_net_variables = {}
    embedding_net_nodes = get_embedding_net_nodes_from_graph_def(graph_def, suffix=suffix)
    for item in embedding_net_nodes:
        node = embedding_net_nodes[item]
        dtype = tf.as_dtype(node.dtype).as_numpy_dtype
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(node.tensor_content, dtype = tf.as_dtype(node.dtype).as_numpy_dtype)
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        embedding_net_variables[item] = np.reshape(tensor_value, tensor_shape)
    return embedding_net_variables

def get_embedding_net_variables(model_file : str, suffix: str = "") -> Dict:
    """
    Get the embedding net variables with the given frozen model(model_file)

    Parameters
    ----------
    model_file
        The input frozen model path
    suffix : str, optional
        The suffix of the scope
    
    Returns
    ----------
    Dict
        The embedding net variables within the given frozen model
    """
    _, graph_def = load_graph_def(model_file)
    return get_embedding_net_variables_from_graph_def(graph_def, suffix=suffix)


def get_fitting_net_nodes_from_graph_def(graph_def: tf.GraphDef) -> Dict:
    """
    Get the fitting net nodes with the given tf.GraphDef object

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    
    Returns
    ----------
    Dict
        The fitting net nodes within the given tf.GraphDef object
    """
    fitting_net_nodes = get_pattern_nodes_from_graph_def(graph_def, FITTING_NET_PATTERN)
    for key in fitting_net_nodes.keys():
        assert key.find('bias') > 0 or key.find('matrix') > 0 or key.find(
            'idt') > 0, "currently, only support weight matrix, bias and idt at the model compression process!"
    return fitting_net_nodes


def get_fitting_net_nodes(model_file : str) -> Dict:
    """
    Get the fitting net nodes with the given frozen model(model_file)

    Parameters
    ----------
    model_file
        The input frozen model path
   
    Returns
    ----------
    Dict
        The fitting net nodes with the given frozen model
    """
    _, graph_def = load_graph_def(model_file)
    return get_fitting_net_nodes_from_graph_def(graph_def)


def get_fitting_net_variables_from_graph_def(graph_def : tf.GraphDef) -> Dict:
    """
    Get the fitting net variables with the given tf.GraphDef object

    Parameters
    ----------
    graph_def
        The input tf.GraphDef object
    
    Returns
    ----------
    Dict
        The fitting net variables within the given tf.GraphDef object 
    """
    fitting_net_variables = {}
    fitting_net_nodes = get_fitting_net_nodes_from_graph_def(graph_def)
    for item in fitting_net_nodes:
        node = fitting_net_nodes[item]
        dtype= tf.as_dtype(node.dtype).as_numpy_dtype
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(node.tensor_content, dtype = tf.as_dtype(node.dtype).as_numpy_dtype)
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        fitting_net_variables[item] = np.reshape(tensor_value, tensor_shape)
    return fitting_net_variables

def get_fitting_net_variables(model_file : str) -> Dict:
    """
    Get the fitting net variables with the given frozen model(model_file)

    Parameters
    ----------
    model_file
        The input frozen model path
    
    Returns
    ----------
    Dict
        The fitting net variables within the given frozen model
    """
    _, graph_def = load_graph_def(model_file)
    return get_fitting_net_variables_from_graph_def(graph_def)
