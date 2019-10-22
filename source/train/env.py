import os
import logging
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

    if 'mkl_info' in np.__config__.__dict__ or 'mkl_rt' in np.__config__.blas_mkl_info['libraries']:
        set_env_if_empty("KMP_BLOCKTIME", "0")
        set_env_if_empty("KMP_AFFINITY", "granularity=fine,verbose,compact,1,0")
        reload(np)
