import os
from deepmd.env import tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util

def convert_13_to_20(args):
    convert_pb_to_pbtxt(args.input_model, 'frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', args.output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model(2.0 support) is saved in %s" % args.output_model)

def convert_pb_to_pbtxt(pbfile, pbtxtfile):
    with gfile.FastGFile(pbfile, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', pbtxtfile, as_text=True)

def convert_pbtxt_to_pb(pbtxtfile, pbfile):
    with tf.gfile.FastGFile(pbtxtfile, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', pbfile, as_text=False)

def convert_dp13_to_dp20(fname):
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
