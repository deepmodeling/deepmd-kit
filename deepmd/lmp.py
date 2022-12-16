"""Register entry points for lammps-wheel."""
import os
from pathlib import Path

from deepmd.env import tf

tf_dir = tf.sysconfig.get_lib()
# set LD_LIBRARY_PATH
os.environ["LD_LIBRARY_PATH"] = ":".join((
    os.environ.get("LD_LIBRARY_PATH", ""),
    tf_dir,
    os.path.join(tf_dir, "python"),
))


def get_op_dir() -> str:
    """Get the directory of the deepmd-kit OP library"""
    return str((Path(__file__).parent / "op").absolute())
