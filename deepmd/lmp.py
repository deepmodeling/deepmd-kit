"""Register entry points for lammps-wheel."""
import os
import platform
from importlib import (
    import_module,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
)

from packaging.version import (
    Version,
)

from deepmd.env import (
    TF_VERSION,
    tf,
)

if Version(TF_VERSION) < Version("2.12"):
    from find_libpython import (
        find_libpython,
    )
else:
    find_libpython = None


def get_env(paths: List[Optional[str]]) -> str:
    """Get the environment variable from given paths."""
    return ":".join(p for p in paths if p is not None)


def get_library_path(module: str) -> List[str]:
    """Get library path from a module.

    Parameters
    ----------
    module : str
        The module name.

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
        return [str(Path(m.__file__).parent)]


if platform.system() == "Linux":
    lib_env = "LD_LIBRARY_PATH"
elif platform.system() == "Darwin":
    lib_env = "DYLD_FALLBACK_LIBRARY_PATH"
else:
    raise RuntimeError("Unsupported platform")

tf_dir = tf.sysconfig.get_lib()
op_dir = str((Path(__file__).parent / "op").absolute())


cuda_library_paths = []
if platform.system() == "Linux":
    cuda_library_paths.extend(
        [
            *get_library_path("nvidia.cuda_runtime.lib"),
            *get_library_path("nvidia.cublas.lib"),
            *get_library_path("nvidia.cublas.lib"),
            *get_library_path("nvidia.cufft.lib"),
            *get_library_path("nvidia.curand.lib"),
            *get_library_path("nvidia.cusolver.lib"),
            *get_library_path("nvidia.cusparse.lib"),
            *get_library_path("nvidia.cudnn.lib"),
        ]
    )

# set LD_LIBRARY_PATH
os.environ[lib_env] = get_env(
    [
        os.environ.get(lib_env),
        tf_dir,
        os.path.join(tf_dir, "python"),
        op_dir,
        *cuda_library_paths,
    ]
)

# preload python library, only for TF<2.12
if find_libpython is not None:
    libpython = find_libpython()
    if platform.system() == "Linux":
        preload_env = "LD_PRELOAD"
    elif platform.system() == "Darwin":
        preload_env = "DYLD_INSERT_LIBRARIES"
    else:
        raise RuntimeError("Unsupported platform")
    os.environ[preload_env] = get_env(
        [
            os.environ.get(preload_env),
            libpython,
        ]
    )


def get_op_dir() -> str:
    """Get the directory of the deepmd-kit OP library."""
    return op_dir
