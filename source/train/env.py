import os
import logging

def set_env_if_empty(key, value):
    if os.environ.get(key) is None:
        os.environ[key] = value
        logging.warn("Environment variable {} is empty. Use the default value {}".format(key, value))

def set_mkl():
    """Tuning MKL for the best performance
    https://www.tensorflow.org/guide/performance/overview
    """

    set_env_if_empty("KMP_BLOCKTIME", "0")
    set_env_if_empty("KMP_AFFINITY", "granularity=fine,verbose,compact,1,0")