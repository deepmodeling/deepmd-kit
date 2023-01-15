"""Register entry points for lammps-wheel."""
import os
import platform
from pathlib import Path
from typing import List, Optional

from find_libpython import find_libpython

from deepmd.env import tf


def get_env(paths: List[Optional[str]]) -> str:
    """Get the environment variable from given paths"""
    return ":".join((p for p in paths if p is not None))


if platform.system() == "Linux":
    lib_env = "LD_LIBRARY_PATH"
elif platform.system() == "Darwin":
    lib_env = "DYLD_FALLBACK_LIBRARY_PATH"
else:
    raise RuntimeError("Unsupported platform")

tf_dir = tf.sysconfig.get_lib()
op_dir = str((Path(__file__).parent / "op").absolute())
# set LD_LIBRARY_PATH
os.environ[lib_env] = get_env([
    os.environ.get(lib_env),
    tf_dir,
    os.path.join(tf_dir, "python"),
    op_dir,
])

# preload python library
libpython = find_libpython()
if platform.system() == "Linux":
    preload_env = "LD_PRELOAD"
elif platform.system() == "Darwin":
    preload_env = "DYLD_INSERT_LIBRARIES"
else:
    raise RuntimeError("Unsupported platform")
os.environ[preload_env] = get_env([
    os.environ.get(preload_env),
    libpython,
])

def get_op_dir() -> str:
    """Get the directory of the deepmd-kit OP library"""
    return op_dir
