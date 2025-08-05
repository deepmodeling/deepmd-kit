# SPDX-License-Identifier: LGPL-3.0-or-later
import platform

import torch
from packaging.version import (
    Version,
)

from deepmd.env import (
    GLOBAL_CONFIG,
    SHARED_LIB_DIR,
)


def load_library(module_name: str) -> bool:
    """Load OP library.

    Parameters
    ----------
    module_name : str
        Name of the module

    Returns
    -------
    bool
        Whether the library is loaded successfully
    """
    if platform.system() == "Windows":
        ext = ".dll"
        prefix = ""
    else:
        ext = ".so"
        prefix = "lib"

    module_file = (SHARED_LIB_DIR / (prefix + module_name)).with_suffix(ext).resolve()

    if module_file.is_file():
        try:
            torch.ops.load_library(module_file)
        except OSError as e:
            # check: CXX11_ABI_FLAG; version
            # from our op
            PT_VERSION = GLOBAL_CONFIG["pt_version"]
            PT_CXX11_ABI_FLAG = int(GLOBAL_CONFIG["pt_cxx11_abi_flag"])
            # from torch
            # strip the local version
            pt_py_version = Version(torch.__version__).public
            pt_cxx11_abi_flag = int(torch.compiled_with_cxx11_abi())

            if PT_CXX11_ABI_FLAG != pt_cxx11_abi_flag:
                raise RuntimeError(
                    "This deepmd-kit package was compiled with "
                    f"CXX11_ABI_FLAG={PT_CXX11_ABI_FLAG}, but PyTorch runtime was compiled "
                    f"with CXX11_ABI_FLAG={pt_cxx11_abi_flag}. These two library ABIs are "
                    f"incompatible and thus an error is raised when loading {module_name}. "
                    "You need to rebuild deepmd-kit against this PyTorch "
                    "runtime."
                ) from e

            # different versions may cause incompatibility, see TF
            if PT_VERSION != pt_py_version:
                raise RuntimeError(
                    "The version of PyTorch used to compile this "
                    f"deepmd-kit package is {PT_VERSION}, but the version of PyTorch "
                    f"runtime you are using is {pt_py_version}. These two versions are "
                    f"incompatible and thus an error is raised when loading {module_name}. "
                    f"You need to install PyTorch {PT_VERSION}, or rebuild deepmd-kit "
                    f"against PyTorch {pt_py_version}.\nIf you are using a wheel from "
                    "PyPI, you may consider to install deepmd-kit execuating "
                    "`DP_ENABLE_PYTORCH=1 pip install deepmd-kit --no-binary deepmd-kit` "
                    "instead."
                ) from e
            error_message = (
                "This deepmd-kit package is inconsistent with PyTorch "
                f"Runtime, thus an error is raised when loading {module_name}. "
                "You need to rebuild deepmd-kit against this PyTorch "
                "runtime."
            )
            if PT_CXX11_ABI_FLAG == 1:
                # #1791
                error_message += (
                    "\nWARNING: devtoolset on RHEL6 and RHEL7 does not support _GLIBCXX_USE_CXX11_ABI=1. "
                    "See https://bugzilla.redhat.com/show_bug.cgi?id=1546704"
                )
            raise RuntimeError(error_message) from e
        return True
    return False


if GLOBAL_CONFIG.get("CIBUILDWHEEL", "0") == "1":
    if platform.system() == "Linux":
        lib_env = "LD_LIBRARY_PATH"
        extension = ".so"
    elif platform.system() == "Darwin":
        lib_env = "DYLD_FALLBACK_LIBRARY_PATH"
        exition = ".dylib"
    else:
        # windows
        pass

    if platform.system() in ("Linux", "Darwin"):
        # try to find MPI directory and add to LD_LIBRARY_PATH
        try:
            MPI_ROOT = (
                [p for p in metadata.files("mpich") if f"libmpi{extension}" in str(p)][
                    0
                ]
                .locate()
                .parent
            )
            # insert MPI_ROOT to LD_LIBRARY_PATH
            current_ld_path = os.environ.get(lib_env, "")
            if current_ld_path:
                os.environ[lib_env] = f"{current_ld_path}:{MPI_ROOT}"
            else:
                os.environ[lib_env] = str(MPI_ROOT)
        except Exception:
            # MPI not found or not available, continue without it
            pass

ENABLE_CUSTOMIZED_OP = load_library("deepmd_op_pt")

__all__ = [
    "ENABLE_CUSTOMIZED_OP",
]
