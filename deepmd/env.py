"""Module that sets tensorflow working environment and exports inportant constants."""

import logging
import os
import platform
from configparser import ConfigParser
from imp import reload
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

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
    "reset_default_tf_session_config",
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
        set_env_if_empty(
            "KMP_AFFINITY", "granularity=fine,verbose,compact,1,0")
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
    config = tf.ConfigProto(
        intra_op_parallelism_threads=intra, inter_op_parallelism_threads=inter
    )
    return config


default_tf_session_config = get_tf_session_config()


def reset_default_tf_session_config(cpu_only: bool):
    """Limit tensorflow session to CPU or not.

    Parameters
    ----------
    cpu_only : bool
        If enabled, no GPU device is visible to the TensorFlow Session.
    """
    global default_tf_session_config
    if cpu_only:
        default_tf_session_config.device_count['GPU'] = 0
    else:
        if 'GPU' in default_tf_session_config.device_count:
            del default_tf_session_config.device_count['GPU']


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
        try:
            module = tf.load_op_library(str(module_file))
        except tf.errors.NotFoundError as e:
            # check CXX11_ABI_FLAG is compatiblity
            # see https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
            # ABI should be the same
            if 'CXX11_ABI_FLAG' in tf.__dict__:
                tf_cxx11_abi_flag = tf.CXX11_ABI_FLAG
            else:
                tf_cxx11_abi_flag = tf.sysconfig.CXX11_ABI_FLAG
            if TF_CXX11_ABI_FLAG != tf_cxx11_abi_flag:
                raise RuntimeError(
                    "This deepmd-kit package was compiled with "
                    "CXX11_ABI_FLAG=%d, but TensorFlow runtime was compiled "
                    "with CXX11_ABI_FLAG=%d. These two library ABIs are "
                    "incompatible and thus an error is raised when loading %s."
                    "You need to rebuild deepmd-kit against this TensorFlow "
                    "runtime." % (
                        TF_CXX11_ABI_FLAG,
                        tf_cxx11_abi_flag,
                        module_name,
                    )) from e

            # different versions may cause incompatibility
            # see #406, #447, #557, #774, and #796 for example
            # throw a message if versions are different
            if TF_VERSION != tf.version.VERSION:
                raise RuntimeError(
                    "The version of TensorFlow used to compile this "
                    "deepmd-kit package is %s, but the version of TensorFlow "
                    "runtime you are using is %s. These two versions are "
                    "incompatible and thus an error is raised when loading %s. "
                    "You need to install TensorFlow %s, or rebuild deepmd-kit "
                    "against TensorFlow %s.\nIf you are using a wheel from "
                    "pypi, you may consider to install deepmd-kit execuating "
                    "`pip install deepmd-kit --no-binary deepmd-kit` "
                    "instead." % (
                        TF_VERSION,
                        tf.version.VERSION,
                        module_name,
                        TF_VERSION,
                        tf.version.VERSION,
                    )) from e
            raise RuntimeError(
                "This deepmd-kit package is inconsitent with TensorFlow"
                "Runtime, thus an error is raised when loading %s."
                "You need to rebuild deepmd-kit against this TensorFlow"
                "runtime." % (
                    module_name,
                )) from e
        return module


def _get_package_constants(
    config_file: Path = Path(__file__).parent / "pkg_config/run_config.ini",
) -> Dict[str, str]:
    """Read package constants set at compile time by CMake to dictionary.

    Parameters
    ----------
    config_file : str, optional
        path to CONFIG file, by default "pkg_config/run_config.ini"

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
TF_VERSION = GLOBAL_CONFIG["tf_version"]
TF_CXX11_ABI_FLAG = int(GLOBAL_CONFIG["tf_cxx11_abi_flag"])

op_module = get_module("libop_abi")
op_grads_module = get_module("libop_grads")

# FLOAT_PREC
dp_float_prec = os.environ.get("DP_INTERFACE_PREC", "high").lower()
if dp_float_prec in ("high", ""):
    # default is high
    GLOBAL_TF_FLOAT_PRECISION = tf.float64
    GLOBAL_NP_FLOAT_PRECISION = np.float64
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "double"
elif dp_float_prec == "low":
    GLOBAL_TF_FLOAT_PRECISION = tf.float32
    GLOBAL_NP_FLOAT_PRECISION = np.float32
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "float"
else:
    raise RuntimeError(
        "Unsupported float precision option: %s. Supported: high,"
        "low. Please set precision with environmental variable "
        "DP_INTERFACE_PREC." % dp_float_prec
    )


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
