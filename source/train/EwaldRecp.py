import platform
import os
import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

if platform.system() == "Windows":
    ext = "dll"
elif platform.system() == "Darwin":
    ext = "dylib"
else:
    ext = "so"

module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.{}".format(ext) )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.{}".format(ext))

