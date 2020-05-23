from deepmd.env import tf
import re
import numpy as np
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
            if precision_dict[old_graph_node[node.name].dtype][1] == "float16" or precision_dict[raw_graph_node[node.name].dtype][1] == "float16":
                raise RuntimeError("float16 conversions not currently supported")

            check_dim(raw_graph_node, old_graph_node, node.name)

            if re.fullmatch("final_layer_type_\d+/bias",node.name) == None:
                tensor_value = np.frombuffer(old_graph_node[node.name].tensor_content,dtype = precision_dict[old_graph_node[node.name].dtype][0])
                tensor_value = tensor_value.astype(dtype=precision_dict[raw_graph_node[node.name].dtype][0])
                node.attr["value"].tensor.tensor_content = tensor_value.tostring()

            else:
                if precision_dict[old_graph_node[node.name].dtype][1] == "float64":
                    tensor_value = (np.array(old_graph_node[node.name].double_val)).astype(precision_dict[raw_graph_node[node.name].dtype][0])
                    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,precision_dict[raw_graph_node[node.name].dtype][0], [1])))
                
                elif precision_dict[old_graph_node[node.name].dtype][1] == "float32":
                    tensor_value = (np.array(old_graph_node[node.name].float_val)).astype(precision_dict[raw_graph_node[node.name].dtype][0])
                    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value, precision_dict[raw_graph_node[node.name].dtype][0], [1])))
                
                elif precision_dict[old_graph_node[node.name].dtype][1] == "float16":
                    tensor_value = (np.array(old_graph_node[node.name].half_val)).astype(precision_dict[raw_graph_node[node.name].dtype][0])
                    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value, precision_dict[raw_graph_node[node.name].dtype][0], [1])))
            
            print("%s is passed from old graph(%s) to raw graph(%s)" % (node.name,precision_dict[old_graph_node[node.name].dtype][1],precision_dict[raw_graph_node[node.name].dtype][1]))
    
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
final_layer_type_\d+/bias|\
final_layer_type_\d+/matrix\
"
    for node in graph.node:
        if re.fullmatch(transform_node_pattern,node.name) != None:
            transform_node[node.name] = node.attr["value"].tensor
    return transform_node
