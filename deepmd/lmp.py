"""Register entry points for lammps-wheel."""
import os
import platform
from pathlib import Path

from deepmd.env import tf

if platform.system() == "Linux":
    lib_env = "LD_LIBRARY_PATH"
elif platform.system() == "Darwin":
    lib_env = "DYLD_FALLBACK_LIBRARY_PATH"
else:
    raise RuntimeError("Unsupported platform")

tf_dir = tf.sysconfig.get_lib()
op_dir = str((Path(__file__).parent / "op").absolute())
# set LD_LIBRARY_PATH
os.environ[lib_env] = ":".join((
    os.environ.get(lib_env, ""),
    tf_dir,
    os.path.join(tf_dir, "python"),
    op_dir,
))


def get_op_dir() -> str:
    """Get the directory of the deepmd-kit OP library"""
    return op_dir
