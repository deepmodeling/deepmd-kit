from google.protobuf import (
    text_format,
)
from tensorflow.python import (
    pywrap_tensorflow,
)
from tensorflow.python.framework import (
    graph_util,
)
from tensorflow.python.platform import (
    gfile,
)

from deepmd.env import (
    tf,
)


def convert_pbtxt_to_pb(pbtxtfile, pbfile):
    with tf.gfile.FastGFile(pbtxtfile, "r") as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, "./", pbfile, as_text=False)
