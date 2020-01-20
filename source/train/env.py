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

def set_env_if_empty(key, value, verbose=True):
    if os.environ.get(key) is None:
        os.environ[key] = value
        if verbose:
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

def set_tf_default_nthreads():
    set_env_if_empty("TF_INTRA_OP_PARALLELISM_THREADS", "0", verbose=False)
    set_env_if_empty("TF_INTER_OP_PARALLELISM_THREADS", "0", verbose=False)

def get_tf_default_nthreads():
    return int(os.environ.get('TF_INTRA_OP_PARALLELISM_THREADS')), int(os.environ.get('TF_INTER_OP_PARALLELISM_THREADS'))
    
def get_tf_session_config():
    set_tf_default_nthreads()    
    intra, inter = get_tf_default_nthreads()
    return tf.ConfigProto(intra_op_parallelism_threads=intra, inter_op_parallelism_threads=inter)

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
default_tf_session_config = get_tf_session_config()
