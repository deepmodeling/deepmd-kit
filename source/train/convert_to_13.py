from deepmd.env import tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util

def convert_to_13(args):
    convert_pb_to_pbtxt(args.input_model, 'frozen_model.pbtxt')
    convert_to_dp13('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', args.output_model)
    print("the converted output model(1.3 support) is saved in %s" % args.output_model)

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

def convert_to_dp13(file):
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
