# SPDX-License-Identifier: LGPL-3.0-or-later
"""Register entry points for lammps-wheel."""

import os
import platform
from importlib import (
    import_module,
)
from pathlib import (
    Path,
)

from packaging.version import (
    Version,
)

from deepmd.env import (
    SHARED_LIB_DIR,
)


def get_env(paths: list[str | None]) -> str:
    """Get the environment variable from given paths."""
    return ":".join(p for p in paths if p is not None)


def get_library_path(module: str, filename: str) -> list[str]:
    """Get library path from a module.

    Parameters
    ----------
    module : str
        The module name.
    filename : str
        The library filename pattern.

    Returns
    -------
    list[str]
        The library path.
    """
    try:
        m = import_module(module)
    except ModuleNotFoundError:
        return []
    else:
        libs = sorted(Path(m.__path__[0]).glob(filename))
        return [str(lib) for lib in libs]


def _get_tensorflow_library_paths() -> tuple[list[str], list[str]]:
    """Get TensorFlow library and preload paths when TensorFlow is installed."""
    try:
        tf_env = import_module("deepmd.tf.env")
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            return [], []
        raise

    tf_dir = tf_env.tf.sysconfig.get_lib()
    preload_paths = []
    if Version(tf_env.TF_VERSION) < Version("2.12"):
        find_libpython = import_module("find_libpython").find_libpython
        libpython = find_libpython()
        if libpython is not None:
            preload_paths.append(libpython)
    return [tf_dir, os.path.join(tf_dir, "python")], preload_paths


def _get_pytorch_library_paths() -> list[str]:
    """Get PyTorch library paths when PyTorch is installed."""
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return []
        raise
    return [os.path.join(torch.__path__[0], "lib")]


if platform.system() == "Linux":
    lib_env = "LD_LIBRARY_PATH"
elif platform.system() == "Darwin":
    lib_env = "DYLD_FALLBACK_LIBRARY_PATH"
else:
    raise RuntimeError("Unsupported platform")

if platform.system() == "Linux":
    preload_env = "LD_PRELOAD"
elif platform.system() == "Darwin":
    preload_env = "DYLD_INSERT_LIBRARIES"
else:
    raise RuntimeError("Unsupported platform")

op_dir = str(SHARED_LIB_DIR)


def _configure_lammps_environment() -> None:
    """Configure library paths for the installed LAMMPS backends."""
    cuda_library_paths = []
    if platform.system() == "Linux":
        cuda_library_paths.extend(
            [
                *get_library_path("nvidia.cuda_runtime.lib", "libcudart.so*"),
                *get_library_path("nvidia.cublas.lib", "libcublasLt.so*"),
                *get_library_path("nvidia.cublas.lib", "libcublas.so*"),
                *get_library_path("nvidia.cufft.lib", "libcufft.so*"),
                *get_library_path("nvidia.curand.lib", "libcurand.so*"),
                *get_library_path("nvidia.cusolver.lib", "libcusolver.so*"),
                *get_library_path("nvidia.cusparse.lib", "libcusparse.so*"),
                *get_library_path("nvidia.cudnn.lib", "libcudnn.so*"),
            ]
        )

    tf_library_paths, tf_preload_paths = _get_tensorflow_library_paths()
    pt_library_paths = _get_pytorch_library_paths()

    os.environ[preload_env] = get_env(
        [
            os.environ.get(preload_env),
            *cuda_library_paths,
            *tf_preload_paths,
        ]
    )

    # set LD_LIBRARY_PATH
    os.environ[lib_env] = get_env(
        [
            os.environ.get(lib_env),
            *tf_library_paths,
            *pt_library_paths,
            op_dir,
        ]
    )


_configure_lammps_environment()


def get_op_dir() -> str:
    """Get the directory of the deepmd-kit OP library."""
    return op_dir
