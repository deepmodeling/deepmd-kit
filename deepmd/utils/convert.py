import os
from deepmd.env import tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile


def convert_13_to_21(input_model: str, output_model: str):
    """Convert DP 1.3 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_13_to_21(input_model: str, output_model: str):
    """Convert DP 1.3 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_12_to_21(input_model: str, output_model: str):
    """Convert DP 1.2 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp12_to_dp13('frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_10_to_21(input_model: str, output_model: str):
    """Convert DP 1.0 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp10_to_dp11('frozen_model.pbtxt')
    convert_dp12_to_dp13('frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_012_to_21(input_model: str, output_model: str):
    """Convert DP 0.12 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp012_to_dp10('frozen_model.pbtxt')
    convert_dp10_to_dp11('frozen_model.pbtxt')
    convert_dp12_to_dp13('frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_20_to_21(input_model: str, output_model: str):
    """Convert DP 2.0 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)

def convert_pb_to_pbtxt(pbfile: str, pbtxtfile: str):
    """Convert DP graph to graph text.
    
    Parameters
    ----------
    pbfile : str
        filename of the input graph
    pbtxtfile : str
        filename of the output graph text
    """
    with gfile.FastGFile(pbfile, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', pbtxtfile, as_text=True)

def convert_pbtxt_to_pb(pbtxtfile: str, pbfile: str):
    """Convert DP graph text to graph.
    
    Parameters
    ----------
    pbtxtfile : str
        filename of the input graph text
    pbfile : str
        filename of the output graph
    """
    with tf.gfile.FastGFile(pbtxtfile, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', pbfile, as_text=False)


def convert_dp012_to_dp10(file: str):
    """Convert DP 1.0 graph text to 1.1 graph text.
    
    Parameters
    ----------
    file : str
        filename of the graph text
    """
    with open(file) as fp:
        file_content = fp.read()
    file_content = file_content\
                   .replace('DescrptNorot', 'DescrptSeA') \
                   .replace('ProdForceNorot', 'ProdForceSeA') \
                   .replace('ProdVirialNorot', 'ProdVirialSeA')
    with open(file, 'w') as fp:
        fp.write(file_content)


def convert_dp10_to_dp11(file: str):
    """Convert DP 1.0 graph text to 1.1 graph text.
    
    Parameters
    ----------
    file : str
        filename of the graph text
    """
    with open(file, 'a') as f:
        f.write("""
node {
  name: "fitting_attr/daparam"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }                                                                                                                                                 }
}
""")


def convert_dp12_to_dp13(file: str):
    """Convert DP 1.2 graph text to 1.3 graph text.
    
    Parameters
    ----------
    file : str
        filename of the graph text
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        ii = 0
        lines = f.readlines()
        while (ii < len(lines)):
            line = lines[ii]
            file_data += line
            ii+=1
            if 'name' in line and ('DescrptSeA' in line or 'ProdForceSeA' in line or 'ProdVirialSeA' in line):
                while not('attr' in lines[ii] and '{' in lines[ii]):
                    file_data += lines[ii]
                    ii+=1
                file_data += '  attr {\n'
                file_data += '    key: \"T\"\n'
                file_data += '    value {\n'
                file_data += '      type: DT_DOUBLE\n'
                file_data += '    }\n'
                file_data += '  }\n'
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def convert_dp13_to_dp20(fname: str):
    """Convert DP 1.3 graph text to 2.0 graph text.
    
    Parameters
    ----------
    file : str
        filename of the graph text
    """
    with open(fname) as fp:
        file_content = fp.read()
    file_content += """
node {
  name: "model_attr/model_version"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "1.0"
      }
    }
  }
}
"""
    file_content = file_content\
                   .replace('DescrptSeA', 'ProdEnvMatA')\
                   .replace('DescrptSeR', 'ProdEnvMatR')
    with open(fname, 'w') as fp:
        fp.write(file_content)

def convert_dp20_to_dp21(fname: str):
    with open(fname) as fp:
        file_content = fp.read()
    old_model_version_node = """
node {
  name: "model_attr/model_version"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "1.0"
      }
    }
  }
}
"""
    new_model_version_node = """
node {
  name: "model_attr/model_version"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "1.1"
      }
    }
  }
}
"""
    file_content = file_content\
                   .replace(old_model_version_node, new_model_version_node)\
                   .replace('TabulateFusion', 'TabulateFusionSeA')\
                   .replace('TabulateFusionGrad', 'TabulateFusionSeAGrad')\
                   .replace('TabulateFusionGradGrad', 'TabulateFusionSeAGradGrad')
    with open(fname, 'w') as fp:
        fp.write(file_content)