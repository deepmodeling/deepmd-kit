"""Module that sets tensorflow working environment and exports inportant constants."""

import os
from pathlib import Path
import logging
import platform
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any
import numpy as np
from imp import reload
from configparser import ConfigParser

if TYPE_CHECKING:
    from types import ModuleType

# import tensorflow v1 compatability
try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

__all__ = [
    "GLOBAL_CONFIG",
    "GLOBAL_TF_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "global_float_prec",
    "global_cvt_2_tf_float",
    "global_cvt_2_ener_float",
    "MODEL_VERSION",
    "SHARED_LIB_MODULE",
    "default_tf_session_config",
    "op_module",
    "op_grads_module",
]

SHARED_LIB_MODULE = "op"

def set_env_if_empty(key: str, value: str, verbose: bool = True):
    """Set environment variable only if it is empty.

    Parameters
    ----------
    key : str
        env variable name
    value : str
        env variable value
    verbose : bool, optional
        if True action will be logged, by default True
    """
    if os.environ.get(key) is None:
        os.environ[key] = value
        if verbose:
            logging.warn(
                f"Environment variable {key} is empty. Use the default value {value}"
            )


def set_mkl():
    """Tuning MKL for the best performance.

    References
    ----------
    TF overview
    https://www.tensorflow.org/guide/performance/overview

    Fixing an issue in numpy built by MKL
    https://github.com/ContinuumIO/anaconda-issues/issues/11367
    https://github.com/numpy/numpy/issues/12374

    check whether the numpy is built by mkl, see
    https://github.com/numpy/numpy/issues/14751
    """
    if "mkl_rt" in np.__config__.get_info("blas_mkl_info").get("libraries", []):
        set_env_if_empty("KMP_BLOCKTIME", "0")
        set_env_if_empty("KMP_AFFINITY", "granularity=fine,verbose,compact,1,0")
        reload(np)


def set_tf_default_nthreads():
    """Set TF internal number of threads to default=automatic selection.

    Notes
    -----
    `TF_INTRA_OP_PARALLELISM_THREADS` and `TF_INTER_OP_PARALLELISM_THREADS`
    control TF configuration of multithreading.
    """
    set_env_if_empty("TF_INTRA_OP_PARALLELISM_THREADS", "0", verbose=False)
    set_env_if_empty("TF_INTER_OP_PARALLELISM_THREADS", "0", verbose=False)


def get_tf_default_nthreads() -> Tuple[int, int]:
    """Get TF paralellism settings.

    Returns
    -------
    Tuple[int, int]
        number of `TF_INTRA_OP_PARALLELISM_THREADS` and
        `TF_INTER_OP_PARALLELISM_THREADS`
    """
    return int(os.environ.get("TF_INTRA_OP_PARALLELISM_THREADS", "0")), int(
        os.environ.get("TF_INTER_OP_PARALLELISM_THREADS", "0")
    )


def get_tf_session_config() -> Any:
    """Configure tensorflow session.

    Returns
    -------
    Any
        session configure object
    """
    set_tf_default_nthreads()
    intra, inter = get_tf_default_nthreads()
    return tf.ConfigProto(
        intra_op_parallelism_threads=intra, inter_op_parallelism_threads=inter
    )

default_tf_session_config = get_tf_session_config()

def get_module(module_name: str) -> "ModuleType":
    """Load force module.

    Returns
    -------
    ModuleType
        loaded force module

    Raises
    ------
    FileNotFoundError
        if module is not found in directory
    """
    if platform.system() == "Windows":
        ext = ".dll"
    elif platform.system() == "Darwin":
        ext = ".dylib"
    else:
        ext = ".so"

    module_file = (
        (Path(__file__).parent / SHARED_LIB_MODULE / module_name)
        .with_suffix(ext)
        .resolve()
    )

    if not module_file.is_file():
        raise FileNotFoundError(f"module {module_name} does not exist")
    else:
        module = tf.load_op_library(str(module_file))
        return module


op_module = get_module("libop_abi")
op_grads_module = get_module("libop_grads")


def _get_package_constants(
    config_file: Path = Path(__file__).parent / "pkg_config/run_config.ini",
) -> Dict[str, str]:
    """Read package constants set at compile time by CMake to dictionary.

    Parameters
    ----------
    config_file : str, optional
        path to CONFIG file, by default "config/run_config.ini"

    Returns
    -------
    Dict[str, str]
        dictionary with package constants
    """
    config = ConfigParser()
    config.read(config_file)
    return dict(config.items("CONFIG"))

GLOBAL_CONFIG = _get_package_constants()
MODEL_VERSION = GLOBAL_CONFIG["model_version"]

if GLOBAL_CONFIG["precision"] == "-DHIGH_PREC":
    GLOBAL_TF_FLOAT_PRECISION = tf.float64
    GLOBAL_NP_FLOAT_PRECISION = np.float64
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "double"
else:
    GLOBAL_TF_FLOAT_PRECISION = tf.float32
    GLOBAL_NP_FLOAT_PRECISION = np.float32
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "float"


def global_cvt_2_tf_float(xx: tf.Tensor) -> tf.Tensor:
    """Cast tensor to globally set TF precision.

    Parameters
    ----------
    xx : tf.Tensor
        input tensor

    Returns
    -------
    tf.Tensor
        output tensor cast to `GLOBAL_TF_FLOAT_PRECISION`
    """
    return tf.cast(xx, GLOBAL_TF_FLOAT_PRECISION)


def global_cvt_2_ener_float(xx: tf.Tensor) -> tf.Tensor:
    """Cast tensor to globally set energy precision.

    Parameters
    ----------
    xx : tf.Tensor
        input tensor

    Returns
    -------
    tf.Tensor
        output tensor cast to `GLOBAL_ENER_FLOAT_PRECISION`
    """
    return tf.cast(xx, GLOBAL_ENER_FLOAT_PRECISION)


