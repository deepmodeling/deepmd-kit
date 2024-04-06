# SPDX-License-Identifier: LGPL-3.0-or-later
"""Module that sets tensorflow working environment and exports inportant constants."""

import ctypes
import logging
import os
import platform
from configparser import (
    ConfigParser,
)
from importlib import (
    import_module,
    reload,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Tuple,
)

import numpy as np
from packaging.version import (
    Version,
)

import deepmd.lib
from deepmd_utils.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    global_float_prec,
)

if TYPE_CHECKING:
    from types import (
        ModuleType,
    )


def dlopen_library(module: str, filename: str):
    """Dlopen a library from a module.

    Parameters
    ----------
    module : str
        The module name.
    filename : str
        The library filename pattern.
    """
    try:
        m = import_module(module)
    except ModuleNotFoundError:
        pass
    else:
        libs = sorted(Path(m.__path__[0]).glob(filename))
        # hope that there is only one version installed...
        if len(libs):
            ctypes.CDLL(str(libs[0].absolute()))


# dlopen pip cuda library before tensorflow
if platform.system() == "Linux":
    dlopen_library("nvidia.cuda_runtime.lib", "libcudart.so*")
    dlopen_library("nvidia.cublas.lib", "libcublasLt.so*")
    dlopen_library("nvidia.cublas.lib", "libcublas.so*")
    dlopen_library("nvidia.cufft.lib", "libcufft.so*")
    dlopen_library("nvidia.curand.lib", "libcurand.so*")
    dlopen_library("nvidia.cusolver.lib", "libcusolver.so*")
    dlopen_library("nvidia.cusparse.lib", "libcusparse.so*")
    dlopen_library("nvidia.cudnn.lib", "libcudnn.so*")


# keras 3 is incompatible with tf.compat.v1
# https://keras.io/getting_started/#tensorflow--keras-2-backwards-compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# import tensorflow v1 compatability
try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
try:
    import tensorflow.compat.v2 as tfv2
except ImportError:
    tfv2 = None

__all__ = [
    "GLOBAL_CONFIG",
    "GLOBAL_TF_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "global_float_prec",
    "global_cvt_2_tf_float",
    "global_cvt_2_ener_float",
    "MODEL_VERSION",
    "SHARED_LIB_DIR",
    "SHARED_LIB_MODULE",
    "default_tf_session_config",
    "reset_default_tf_session_config",
    "op_module",
    "op_grads_module",
    "TRANSFER_PATTERN",
    "FITTING_NET_PATTERN",
    "EMBEDDING_NET_PATTERN",
    "TYPE_EMBEDDING_PATTERN",
    "ATTENTION_LAYER_PATTERN",
    "REMOVE_SUFFIX_DICT",
    "TF_VERSION",
]

SHARED_LIB_MODULE = "lib"
SHARED_LIB_DIR = Path(deepmd.lib.__path__[0])
CONFIG_FILE = SHARED_LIB_DIR / "run_config.ini"

# Python library version
try:
    tf_py_version = tf.version.VERSION
except AttributeError:
    tf_py_version = tf.__version__

EMBEDDING_NET_PATTERN = str(
    r"filter_type_\d+/matrix_\d+_\d+|"
    r"filter_type_\d+/bias_\d+_\d+|"
    r"filter_type_\d+/idt_\d+_\d+|"
    r"filter_type_all/matrix_\d+|"
    r"filter_type_all/matrix_\d+_\d+|"
    r"filter_type_all/matrix_\d+_\d+_\d+|"
    r"filter_type_all/bias_\d+|"
    r"filter_type_all/bias_\d+_\d+|"
    r"filter_type_all/bias_\d+_\d+_\d+|"
    r"filter_type_all/idt_\d+|"
    r"filter_type_all/idt_\d+_\d+|"
)

FITTING_NET_PATTERN = str(
    r"layer_\d+/matrix|"
    r"layer_\d+_type_\d+/matrix|"
    r"layer_\d+/bias|"
    r"layer_\d+_type_\d+/bias|"
    r"layer_\d+/idt|"
    r"layer_\d+_type_\d+/idt|"
    r"final_layer/matrix|"
    r"final_layer_type_\d+/matrix|"
    r"final_layer/bias|"
    r"final_layer_type_\d+/bias|"
    # layer_name
    r"share_.+_type_\d/matrix|"
    r"share_.+_type_\d/bias|"
    r"share_.+_type_\d/idt|"
    r"share_.+/matrix|"
    r"share_.+/bias|"
    r"share_.+/idt|"
)

TYPE_EMBEDDING_PATTERN = str(
    r"type_embed_net+/matrix_\d+|"
    r"type_embed_net+/bias_\d+|"
    r"type_embed_net+/idt_\d+|"
)

ATTENTION_LAYER_PATTERN = str(
    r"attention_layer_\d+/c_query/matrix|"
    r"attention_layer_\d+/c_query/bias|"
    r"attention_layer_\d+/c_key/matrix|"
    r"attention_layer_\d+/c_key/bias|"
    r"attention_layer_\d+/c_value/matrix|"
    r"attention_layer_\d+/c_value/bias|"
    r"attention_layer_\d+/c_out/matrix|"
    r"attention_layer_\d+/c_out/bias|"
    r"attention_layer_\d+/layer_normalization/beta|"
    r"attention_layer_\d+/layer_normalization/gamma|"
    r"attention_layer_\d+/layer_normalization_\d+/beta|"
    r"attention_layer_\d+/layer_normalization_\d+/gamma|"
)

TRANSFER_PATTERN = (
    EMBEDDING_NET_PATTERN
    + FITTING_NET_PATTERN
    + TYPE_EMBEDDING_PATTERN
    + str(
        r"descrpt_attr/t_avg|"
        r"descrpt_attr/t_std|"
        r"fitting_attr/t_fparam_avg|"
        r"fitting_attr/t_fparam_istd|"
        r"fitting_attr/t_aparam_avg|"
        r"fitting_attr/t_aparam_istd|"
        r"model_attr/t_tab_info|"
        r"model_attr/t_tab_data|"
    )
)

REMOVE_SUFFIX_DICT = {
    "model_attr/sel_type_{}": "model_attr/sel_type",
    "model_attr/output_dim_{}": "model_attr/output_dim",
    "_{}/": "/",
    # when atom_ener is set
    "_{}_1/": "_1/",
    "o_energy_{}": "o_energy",
    "o_force_{}": "o_force",
    "o_virial_{}": "o_virial",
    "o_atom_energy_{}": "o_atom_energy",
    "o_atom_virial_{}": "o_atom_virial",
    "o_dipole_{}": "o_dipole",
    "o_global_dipole_{}": "o_global_dipole",
    "o_polar_{}": "o_polar",
    "o_global_polar_{}": "o_global_polar",
    "o_rmat_{}": "o_rmat",
    "o_rmat_deriv_{}": "o_rmat_deriv",
    "o_nlist_{}": "o_nlist",
    "o_rij_{}": "o_rij",
    "o_dm_force_{}": "o_dm_force",
    "o_dm_virial_{}": "o_dm_virial",
    "o_dm_av_{}": "o_dm_av",
    "o_wfc_{}": "o_wfc",
}


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
            logging.warning(
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
    try:
        is_mkl = (
            np.show_config("dicts")
            .get("Build Dependencies", {})
            .get("blas", {})
            .get("name", "")
            .lower()
            .startswith("mkl")
        )
    except TypeError:
        is_mkl = "mkl_rt" in np.__config__.get_info("blas_mkl_info").get(
            "libraries", []
        )
    if is_mkl:
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
    if (
        "OMP_NUM_THREADS" not in os.environ
        or "TF_INTRA_OP_PARALLELISM_THREADS" not in os.environ
        or "TF_INTER_OP_PARALLELISM_THREADS" not in os.environ
    ):
        logging.warning(
            "To get the best performance, it is recommended to adjust "
            "the number of threads by setting the environment variables "
            "OMP_NUM_THREADS, TF_INTRA_OP_PARALLELISM_THREADS, and "
            "TF_INTER_OP_PARALLELISM_THREADS. See "
            "https://deepmd.rtfd.io/parallelism/ for more information."
        )
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
    if int(os.environ.get("DP_JIT", 0)):
        set_env_if_empty("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
        # pip cuda package
        if platform.system() == "Linux":
            try:
                m = import_module("nvidia.cuda_nvcc")
            except ModuleNotFoundError:
                pass
            else:
                cuda_data_dir = str(Path(m.__file__).parent.absolute())
                set_env_if_empty(
                    "XLA_FLAGS", "--xla_gpu_cuda_data_dir=" + cuda_data_dir
                )
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        intra_op_parallelism_threads=intra,
        inter_op_parallelism_threads=inter,
    )
    if Version(tf_py_version) >= Version("1.15") and int(
        os.environ.get("DP_AUTO_PARALLELIZATION", 0)
    ):
        config.graph_options.rewrite_options.custom_optimizers.add().name = "dpparallel"
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
        default_tf_session_config.device_count["GPU"] = 0
    else:
        if "GPU" in default_tf_session_config.device_count:
            del default_tf_session_config.device_count["GPU"]


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
        prefix = ""
    # elif platform.system() == "Darwin":
    #    ext = ".dylib"
    else:
        ext = ".so"
        prefix = "lib"

    module_file = (SHARED_LIB_DIR / (prefix + module_name)).with_suffix(ext).resolve()

    if not module_file.is_file():
        raise FileNotFoundError(f"module {module_name} does not exist")
    else:
        try:
            module = tf.load_op_library(str(module_file))
        except tf.errors.NotFoundError as e:
            # check CXX11_ABI_FLAG is compatiblity
            # see https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
            # ABI should be the same
            if "CXX11_ABI_FLAG" in tf.__dict__:
                tf_cxx11_abi_flag = tf.CXX11_ABI_FLAG
            else:
                tf_cxx11_abi_flag = tf.sysconfig.CXX11_ABI_FLAG
            if TF_CXX11_ABI_FLAG != tf_cxx11_abi_flag:
                raise RuntimeError(
                    "This deepmd-kit package was compiled with "
                    "CXX11_ABI_FLAG=%d, but TensorFlow runtime was compiled "
                    "with CXX11_ABI_FLAG=%d. These two library ABIs are "
                    "incompatible and thus an error is raised when loading %s. "
                    "You need to rebuild deepmd-kit against this TensorFlow "
                    "runtime."
                    % (
                        TF_CXX11_ABI_FLAG,
                        tf_cxx11_abi_flag,
                        module_name,
                    )
                ) from e

            # different versions may cause incompatibility
            # see #406, #447, #557, #774, and #796 for example
            # throw a message if versions are different
            if TF_VERSION != tf_py_version:
                raise RuntimeError(
                    "The version of TensorFlow used to compile this "
                    f"deepmd-kit package is {TF_VERSION}, but the version of TensorFlow "
                    f"runtime you are using is {tf_py_version}. These two versions are "
                    f"incompatible and thus an error is raised when loading {module_name}. "
                    f"You need to install TensorFlow {TF_VERSION}, or rebuild deepmd-kit "
                    f"against TensorFlow {tf_py_version}.\nIf you are using a wheel from "
                    "pypi, you may consider to install deepmd-kit execuating "
                    "`pip install deepmd-kit --no-binary deepmd-kit` "
                    "instead."
                ) from e
            error_message = (
                "This deepmd-kit package is inconsitent with TensorFlow "
                f"Runtime, thus an error is raised when loading {module_name}. "
                "You need to rebuild deepmd-kit against this TensorFlow "
                "runtime."
            )
            if TF_CXX11_ABI_FLAG == 1:
                # #1791
                error_message += (
                    "\nWARNING: devtoolset on RHEL6 and RHEL7 does not support _GLIBCXX_USE_CXX11_ABI=1. "
                    "See https://bugzilla.redhat.com/show_bug.cgi?id=1546704"
                )
            raise RuntimeError(error_message) from e
        return module


def _get_package_constants(
    config_file: Path = CONFIG_FILE,
) -> Dict[str, str]:
    """Read package constants set at compile time by CMake to dictionary.

    Parameters
    ----------
    config_file : str, optional
        path to CONFIG file, by default "run_config.ini"

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

op_module = get_module("deepmd_op")
op_grads_module = get_module("op_grads")

# FLOAT_PREC
GLOBAL_TF_FLOAT_PRECISION = tf.dtypes.as_dtype(GLOBAL_NP_FLOAT_PRECISION)


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
