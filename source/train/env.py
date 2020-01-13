import os
import logging
import platform
import numpy as np
from imp import reload

# import tensorflow v1 compatability
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

def set_env_if_empty(key, value):
    if os.environ.get(key) is None:
        os.environ[key] = value
        logging.warn("Environment variable {} is empty. Use the default value {}".format(key, value))

def set_mkl():
    """Tuning MKL for the best performance
    https://www.tensorflow.org/guide/performance/overview
    
    Fixing an issue in numpy built by MKL. See
    https://github.com/ContinuumIO/anaconda-issues/issues/11367
    https://github.com/numpy/numpy/issues/12374
    """

    # check whether the numpy is built by mkl, see
    # https://github.com/numpy/numpy/issues/14751
    if 'mkl_rt' in np.__config__.get_info("blas_mkl_info").get('libraries', []):
        set_env_if_empty("KMP_BLOCKTIME", "0")
        set_env_if_empty("KMP_AFFINITY", "granularity=fine,verbose,compact,1,0")
        reload(np)

def get_module(module_name):
    """Load force module."""
    if platform.system() == "Windows":
        ext = "dll"
    elif platform.system() == "Darwin":
        ext = "dylib"
    else:
        ext = "so"
    module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    assert (os.path.isfile (module_path  + "{}.{}".format(module_name, ext) )), "module %s does not exist" % module_name
    module = tf.load_op_library(module_path + "{}.{}".format(module_name, ext))
    return module

op_module = get_module("libop_abi")
op_grads_module = get_module("libop_grads")