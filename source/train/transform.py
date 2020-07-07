from deepmd.env import tf
import re
import numpy as np

def convertNumber(number):
    binary = bin(number).replace("0b", "").zfill(16)
    sign = int(binary[0]) * (-2) + 1
    exp = int(binary[1:6], 2)
    frac = (int(binary[6:], 2) + 2 ** 10) * (2 ** -10)
    return sign * (2 ** (exp - 15)) * frac


def convertMatrix(matrix, shape):
    matrix = matrix.flatten()
    tmp = np.array([convertNumber(matrix[i]) for i in range(len(matrix))])
    return tmp.reshape(shape)


def transform(args):
    raw_graph = load_graph(args.raw_model)
    old_graph = load_graph(args.old_model)
    print("%d ops in the raw graph\n%d ops in the old graph" %(len(raw_graph.as_graph_def().node),len(old_graph.as_graph_def().node)))
    new_graph_def = transform_graph(raw_graph,old_graph)
    with tf.gfile.GFile(args.output, mode='wb') as f:
        f.write(new_graph_def.SerializeToString())
    print("the output model is saved in %s" % args.output)

def load_graph(graphName):
    graph_def = tf.GraphDef()
    with open(graphName,"rb") as f:
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name = "")
    return graph

def transform_graph(raw_graph,old_graph):
    precision_dict = {\
    1:(np.float32, "float32"),\
    2:(np.float64, "float64"),\
    19:(np.float16, "float16")\
    }
    old_graph_def = old_graph.as_graph_def()
    raw_graph_def = raw_graph.as_graph_def()
    raw_graph_node = load_transform_node(raw_graph_def)
    old_graph_node = load_transform_node(old_graph_def)

    if len(raw_graph_node) != len(old_graph_node):
        raise RuntimeError("raw graph and old graph has different network structure")

    for node in raw_graph_def.node:
        if node.name in raw_graph_node.keys():

            check_dim(raw_graph_node, old_graph_node, node.name)
            tensor_shape = [dim.size for dim in raw_graph_node[node.name].tensor_shape.dim]
            old_graph_dtype = precision_dict[old_graph_node[node.name].dtype]
            raw_graph_dtype = precision_dict[raw_graph_node[node.name].dtype]
            print("%s is passed from old graph(%s) to raw graph(%s)" % (node.name, old_graph_dtype[1],raw_graph_dtype[1]))
            
            if raw_graph_dtype[1] == "float16":
                if old_graph_dtype[1] == "float64" or old_graph_dtype[1] == "float32":
                    if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
                        tensor_value = np.frombuffer(old_graph_node[node.name].tensor_content, dtype=old_graph_dtype[0])
                        tensor_value = tensor_value.astype(np.float16)
                        node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value, tf.float16, tensor_shape)))

                    else:
                        if old_graph_dtype[1] == "float64":
                            tensor_value = (np.array(old_graph_node[node.name].double_val)).astype(np.float16)
                            node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,tf.float16, [1])))

                        elif old_graph_dtype[1] == "float32":
                            tensor_value = (np.array(old_graph_node[node.name].float_val)).astype(np.float16)
                            node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,tf.float16, [1])))

                elif old_graph_dtype[1] == "float16":
                    tensor_value = convertMatrix(np.array(old_graph_node[node.name].half_val), tensor_shape)
                    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value, tf.float16, tensor_value.shape)))
            
            elif raw_graph_dtype[1] == "float64" or raw_graph_dtype[1] == "float32":
                if old_graph_dtype[1] == "float64" or old_graph_dtype[1] == "float32":
                    if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
                        tensor_value = np.frombuffer(old_graph_node[node.name].tensor_content,dtype = old_graph_dtype[0])
                        tensor_value = tensor_value.astype(dtype=raw_graph_dtype[0])
                        node.attr["value"].tensor.tensor_content = tensor_value.tostring()

                    else:
                        if old_graph_dtype[1] == "float64":
                            tensor_value = (np.array(old_graph_node[node.name].double_val)).astype(raw_graph_dtype[0])
                            node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,raw_graph_dtype[0], [1])))

                        elif old_graph_dtype[1] == "float32": 
                            tensor_value = (np.array(old_graph_node[node.name].float_val)).astype(raw_graph_dtype[0])
                            node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,raw_graph_dtype[0], [1])))
                
                elif old_graph_dtype[1] == "float16":
                    if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
                        tensor_value = convertMatrix(np.array(old_graph_node[node.name].half_val), tensor_shape)
                        tensor_value = tensor_value.astype(raw_graph_dtype[0])
                        node.attr["value"].tensor.tensor_content = tensor_value.tostring()
                    else:
                        tensor_value = convertMatrix(np.array(old_graph_node[node.name].half_val), tensor_shape)
                        tensor_value = tensor_value.astype(raw_graph_dtype[0])
                        node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,raw_graph_dtype[0], tensor_value.shape)))

    return raw_graph_def

def check_dim(raw_graph_node, old_graph_node, node_name):
    raw_graph_dim = raw_graph_node[node_name].tensor_shape
    old_graph_dim = old_graph_node[node_name].tensor_shape
    if raw_graph_dim != old_graph_dim:
        raise RuntimeError("old graph and raw graph has different"+node_name+" dim")


def load_transform_node(graph):
    transform_node = {}
    transform_node_pattern = "\
filter_type_\d+/matrix_\d+_\d+|\
filter_type_\d+/bias_\d+_\d+|\
filter_type_\d+/idt_\d+_\d+|\
layer_\d+_type_\d+/matrix|\
layer_\d+_type_\d+/bias|\
layer_\d+_type_\d+/idt|\
final_layer_type_\d+/matrix|\
descrpt_attr/t_avg|\
descrpt_attr/t_std|\
final_layer_type_\d+/bias|\
fitting_attr/t_fparam_avg|\
fitting_attr/t_fparam_istd|\
fitting_attr/t_aparam_avg|\
fitting_attr/t_aparam_istd|\
model_attr/t_tab_info|\
model_attr/t_tab_data|\
"
    for node in graph.node:
        if re.fullmatch(transform_node_pattern,node.name) != None:
            transform_node[node.name] = node.attr["value"].tensor
    return transform_node
