# SPDX-License-Identifier: LGPL-3.0-or-later
"""Read environment variables to configure the build."""

import os
from functools import (
    lru_cache,
)
from typing import (
    Tuple,
)

from packaging.version import (
    Version,
)

from .find_tensorflow import (
    find_tensorflow,
    get_tf_version,
)


@lru_cache
def get_argument_from_env() -> Tuple[str, list, list, dict, str]:
    """Get the arguments from environment variables.

    The environment variables are assumed to be not changed during the build.

    Returns
    -------
    str
        The minimum required CMake version.
    list of str
        The CMake arguments.
    list of str
        The requirements for the build.
    dict
        The extra scripts to be installed.
    str
        The TensorFlow version.
    """
    cmake_args = []
    extra_scripts = {}
    # get variant option from the environment varibles, available: cpu, cuda, rocm
    dp_variant = os.environ.get("DP_VARIANT", "cpu").lower()
    if dp_variant == "cpu" or dp_variant == "":
        cmake_minimum_required_version = "3.16"
    elif dp_variant == "cuda":
        cmake_minimum_required_version = "3.23"
        cmake_args.append("-DUSE_CUDA_TOOLKIT:BOOL=TRUE")
        cuda_root = os.environ.get("CUDAToolkit_ROOT")
        if cuda_root:
            cmake_args.append(f"-DCUDAToolkit_ROOT:STRING={cuda_root}")
    elif dp_variant == "rocm":
        cmake_minimum_required_version = "3.21"
        cmake_args.append("-DUSE_ROCM_TOOLKIT:BOOL=TRUE")
        rocm_root = os.environ.get("ROCM_ROOT")
        if rocm_root:
            cmake_args.append(f"-DCMAKE_HIP_COMPILER_ROCM_ROOT:STRING={rocm_root}")
        hipcc_flags = os.environ.get("HIP_HIPCC_FLAGS")
        if hipcc_flags is not None:
            os.environ["HIPFLAGS"] = os.environ.get("HIPFLAGS", "") + " " + hipcc_flags
    else:
        raise RuntimeError("Unsupported DP_VARIANT option: %s" % dp_variant)

    if os.environ.get("DP_BUILD_TESTING", "0") == "1":
        cmake_args.append("-DBUILD_TESTING:BOOL=TRUE")
    if os.environ.get("DP_ENABLE_NATIVE_OPTIMIZATION", "0") == "1":
        cmake_args.append("-DENABLE_NATIVE_OPTIMIZATION:BOOL=TRUE")
    dp_lammps_version = os.environ.get("DP_LAMMPS_VERSION", "")
    dp_ipi = os.environ.get("DP_ENABLE_IPI", "0")
    if dp_lammps_version != "" or dp_ipi == "1":
        cmake_args.append("-DBUILD_CPP_IF:BOOL=TRUE")
        cmake_args.append("-DUSE_TF_PYTHON_LIBS:BOOL=TRUE")
    else:
        cmake_args.append("-DBUILD_CPP_IF:BOOL=FALSE")

    if dp_lammps_version != "":
        cmake_args.append(f"-DLAMMPS_VERSION={dp_lammps_version}")
    if dp_ipi == "1":
        cmake_args.append("-DENABLE_IPI:BOOL=TRUE")
        extra_scripts["dp_ipi"] = "deepmd.tf.entrypoints.ipi:dp_ipi"

    if os.environ.get("DP_ENABLE_TENSORFLOW", "1") == "1":
        tf_install_dir, _ = find_tensorflow()
        tf_version = get_tf_version(tf_install_dir)
        if tf_version == "" or Version(tf_version) >= Version("2.12"):
            find_libpython_requires = []
        else:
            find_libpython_requires = ["find_libpython"]
        cmake_args.extend(
            [
                "-DENABLE_TENSORFLOW=ON",
                f"-DTENSORFLOW_VERSION={tf_version}",
                f"-DTENSORFLOW_ROOT:PATH={tf_install_dir}",
            ]
        )
    else:
        find_libpython_requires = []
        cmake_args.append("-DENABLE_TENSORFLOW=OFF")
        tf_version = None

    cmake_args = [
        "-DBUILD_PY_IF:BOOL=TRUE",
        *cmake_args,
    ]
    return (
        cmake_minimum_required_version,
        cmake_args,
        find_libpython_requires,
        extra_scripts,
        tf_version,
    )


def set_scikit_build_env():
    """Set scikit-build environment variables before executing scikit-build."""
    cmake_minimum_required_version, cmake_args, _, _, _ = get_argument_from_env()
    os.environ["SKBUILD_CMAKE_MINIMUM_VERSION"] = cmake_minimum_required_version
    os.environ["SKBUILD_CMAKE_ARGS"] = ";".join(cmake_args)
