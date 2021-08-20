import re
import numpy as np
from deepmd.env import tf
from deepmd.common import PRECISION_MAPPING
from deepmd.utils.sess import run_sess
from deepmd.utils.errors import GraphWithoutTensorError

def load_graph_def(model_file: str):
    """
    Load graph as well as the graph_def from the frozen model(model_file)

    Parameters
    ----------
    model_file : str
        The input frozen model.

    Returns
    -------
    graph
        The graph loaded from the frozen model.
    graph_def
        The graph_def loaded from the frozen model.
    """
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name = "")
    return graph, graph_def


def get_tensor_by_name(model_file: str,
                       tensor_name: str) -> tf.Tensor:
    """
    Load tensor value from the frozen model(model_file)

    Parameters
    ----------
    model_file : str
        The input frozen model.
    tensor_name : str
        Indicates which tensor which will be loaded from the frozen model.

    Returns
    -------
    tf.Tensor
        The tensor which was loaded from the frozen model.

    Raises
    ------
    GraphWithoutTensorError
        Whether the tensor_name is within the frozen model.
    """
    graph, _ = load_graph_def(model_file)
    try:
        tensor = graph.get_tensor_by_name(tensor_name + ':0')
    except KeyError as e:
        raise GraphWithoutTensorError() from e
    with tf.Session(graph=graph) as sess:
        tensor = run_sess(sess, tensor)
    return tensor


def get_tensor_by_type(node,
                       data_type : np.dtype):
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
    tensor
        The tensor value of the given node
    """
    if data_type == np.float64:
        tensor = np.array(node.double_val)
    elif data_type == np.float32:
        tensor = np.array(node.float_val)
    else:
        raise RunTimeError('model compression does not support the half precision')
    return tensor


def get_embedding_net_nodes(model_file: str):
    """
    Get the embedding net nodes with the given frozen model(model_file)

    Parameters
    ----------
    model_file
        The input frozen model.
    
    Returns
    ----------
    embedding_net_nodes
        The embedding net nodes with the given frozen model. 
    """
    _, graph_def = load_graph_def(model_file)
    embedding_net_nodes = {}
    embedding_net_pattern = "filter_type_\d+/matrix_\d+_\d+|filter_type_\d+/bias_\d+_\d+|filter_type_\d+/idt_\d+_\d+|filter_type_all/matrix_\d+_\d+|filter_type_all/bias_\d+_\d+|filter_type_all/idt_\d+_\d"
    for node in graph_def.node:
        if re.fullmatch(embedding_net_pattern, node.name) != None:
            embedding_net_nodes[node.name] = node.attr["value"].tensor
    for key in embedding_net_nodes.keys():
        assert key.find('bias') > 0 or key.find(
            'matrix') > 0, "currently, only support weight matrix and bias matrix at the tabulation op!"
    return embedding_net_nodes


def get_embedding_net_variables(model_file : str):
    """
    Get the embedding net variables with the given frozen model(model_file)

    Parameters
    ----------
    model_file
        The input frozen model.
    
    Returns
    ----------
        The embedding net variables within the given frozen model. 
    """
    embedding_net_variables = {}
    embedding_net_nodes = get_embedding_net_nodes(model_file)
    for item in embedding_net_nodes:
        node = embedding_net_nodes[item]
        dtype = PRECISION_MAPPING[node.dtype]
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(node.tensor_content)
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        embedding_net_variables[item] = np.reshape(tensor_value, tensor_shape)
    return embedding_net_variables


def get_fitting_net_nodes(model_file : str):
    """
    Get the fitting net nodes with the given frozen model(model_file)

    Parameters
    ----------
    model_file
        The input frozen model.
    
    Returns
    ----------
    fitting_net_nodes
        The fitting net nodes with the given frozen model. 
    """
    _, graph_def = load_graph_def(model_file)
    fitting_net_nodes = {}
    fitting_net_pattern = "layer_\d+_type_\d+/matrix+|layer_\d+_type_\d+/bias+|layer_\d+_type_\d+/idt+|final_layer_type_\d+/matrix+|final_layer_type_\d+/bias"
    for node in graph_def.node:
        if re.fullmatch(fitting_net_pattern, node.name) != None:
            fitting_net_nodes[node.name] = node.attr["value"].tensor
    for key in fitting_net_nodes.keys():
        assert key.find('bias') > 0 or key.find('matrix') > 0 or key.find(
            'idt') > 0, "currently, only support weight matrix, bias and idt at the model compression process!"
    return fitting_net_nodes


def get_fitting_net_variables(model_file : str):
    """
    Get the fitting net variables with the given frozen model(model_file)

    Parameters
    ----------
    model_file
        The input frozen model.
    
    Returns
    ----------
        The fitting net variables within the given frozen model. 
    """
    fitting_net_variables = {}
    fitting_net_nodes = get_fitting_net_nodes(model_file)
    for item in fitting_net_nodes:
        node = fitting_net_nodes[item]
        dtype= PRECISION_MAPPING[node.dtype]
        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
        if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
            tensor_value = np.frombuffer(node.tensor_content)
        else:
            tensor_value = get_tensor_by_type(node, dtype)
        fitting_net_variables[item] = np.reshape(tensor_value, tensor_shape)
    return fitting_net_variables