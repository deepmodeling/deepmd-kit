from deepmd.env import tf
def transform(args):
    new_graph = load_graph(args.raw_model)
    old_graph = load_graph(args.old_model)
    print("%d ops in the raw graph\n%d ops in the old graph" %(len(new_graph.node),len(old_graph.node)))
    transform_node = load_data(new_graph,old_graph)
    for node in new_graph.node:
        if node.name in transform_node:
            print("%s is passed from old graph to raw graph" % node.name)
            node.attr["value"].tensor.CopyFrom(transform_node[node.name].attr["value"].tensor)
    with tf.gfile.GFile(args.output, mode='wb') as f:
        f.write(new_graph.SerializeToString())
    print("the output model is saved in %s" % args.output)

def load_graph(graphName):
    graph_def = tf.GraphDef()
    with open(graphName,"rb") as f:
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name = "")
    return graph_def

def load_data(new_graph,old_graph):
    new_graph_node = load_transform_node(new_graph)
    old_graph_node = load_transform_node(old_graph)
    if len(new_graph_node) != len(old_graph_node):
        raise RuntimeError("New graph and original graph has different network structure\n")
    for nodeName in old_graph_node.keys():
        check_dim(new_graph_node, old_graph_node, nodeName)
        check_precision(new_graph_node, old_graph_node, nodeName)
    return old_graph_node
        

def check_precision(new_graph_node, old_graph_node, nodeName):
    new_graph_precision = new_graph_node[nodeName].attr["value"].tensor.dtype
    old_graph_precision = old_graph_node[nodeName].attr["value"].tensor.dtype
    if new_graph_precision != old_graph_precision:
        raise RuntimeError("New graph and original graph has different"+nodeName+" precision\n")

def check_dim(new_graph_node, old_graph_node, nodeName):
    new_graph_dim = new_graph_node[nodeName].attr["value"].tensor.tensor_shape
    old_graph_dim = old_graph_node[nodeName].attr["value"].tensor.tensor_shape
    if new_graph_dim != old_graph_dim:
        raise RuntimeError("New graph and original graph has different"+nodeName+" dim\n")


def load_transform_node(graph):
    transform_node = {}
    filter_w = ["filter_type_0/matrix_{}_0".format(i) for i in range(1,10)]
    filter_b = ["filter_type_0/bias_{}_0".format(i) for i in range(1,10)]
    fitting_w = ["layer_{}_type_0/matrix".format(i) for i in range(0,10)]
    fitting_b = ["layer_{}_type_0/bias".format(i) for i in range(0,10)]
    fitting_idt = ["layer_{}_type_0/idt".format(i) for i in range(0,10)]
    final_layer = ["final_layer_type_0/bias","final_layer_type_0/matrix"]
    transform_node_list = filter_w + filter_b + fitting_w + fitting_b + fitting_idt + final_layer
    for node in graph.node:
        if node.name in transform_node_list:
            transform_node[node.name] = node
    return transform_node
