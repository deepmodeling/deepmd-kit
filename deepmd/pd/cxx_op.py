# SPDX-License-Identifier: LGPL-3.0-or-later
import platform

import paddle
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
            paddle.utils.cpp_extension.load(module_file)
        except OSError as e:
            # check: CXX11_ABI_FLAG; version
            # from our op
            PD_VERSION = GLOBAL_CONFIG["pd_version"]
            PD_CXX11_ABI_FLAG = int(GLOBAL_CONFIG["pd_cxx11_abi_flag"])
            # from paddle
            # strip the local version
            pd_py_version = Version(paddle.__version__).public
            # pd_cxx11_abi_flag = int(paddle.compiled_with_cxx11_abi())
            pd_cxx11_abi_flag = 0

            if PD_CXX11_ABI_FLAG != pd_cxx11_abi_flag:
                raise RuntimeError(
                    "This deepmd-kit package was compiled with "
                    "CXX11_ABI_FLAG=%d, but Paddle runtime was compiled "
                    "with CXX11_ABI_FLAG=%d. These two library ABIs are "
                    "incompatible and thus an error is raised when loading %s. "
                    "You need to rebuild deepmd-kit against this Paddle "
                    "runtime."
                    % (
                        PD_CXX11_ABI_FLAG,
                        pd_cxx11_abi_flag,
                        module_name,
                    )
                ) from e

            # different versions may cause incompatibility, see TF
            if PD_VERSION != pd_py_version:
                raise RuntimeError(
                    "The version of Paddle used to compile this "
                    f"deepmd-kit package is {PD_VERSION}, but the version of Paddle "
                    f"runtime you are using is {pd_py_version}. These two versions are "
                    f"incompatible and thus an error is raised when loading {module_name}. "
                    f"You need to install Paddle {PD_VERSION}, or rebuild deepmd-kit "
                    f"against Paddle {pd_py_version}.\nIf you are using a wheel from "
                    "PyPI, you may consider to install deepmd-kit execuating "
                    "`DP_ENABLE_Paddle=1 pip install deepmd-kit --no-binary deepmd-kit` "
                    "instead."
                ) from e
            error_message = (
                "This deepmd-kit package is inconsitent with Paddle "
                f"Runtime, thus an error is raised when loading {module_name}. "
                "You need to rebuild deepmd-kit against this Paddle "
                "runtime."
            )
            if PD_CXX11_ABI_FLAG == 1:
                # #1791
                error_message += (
                    "\nWARNING: devtoolset on RHEL6 and RHEL7 does not support _GLIBCXX_USE_CXX11_ABI=1. "
                    "See https://bugzilla.redhat.com/show_bug.cgi?id=1546704"
                )
            raise RuntimeError(error_message) from e
        return True
    return False


ENABLE_CUSTOMIZED_OP = load_library("deepmd_op_pd")

__all__ = [
    "ENABLE_CUSTOMIZED_OP",
]
