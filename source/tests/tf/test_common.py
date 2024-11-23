# SPDX-License-Identifier: LGPL-3.0-or-later
import shutil
import unittest
from pathlib import (
    Path,
)

from deepmd.tf.common import (
    GLOBAL_TF_FLOAT_PRECISION,
    cast_precision,
    expand_sys_str,
)
from deepmd.tf.env import (
    tf,
)


# compute relative path
# https://stackoverflow.com/questions/38083555/using-pathlibs-relative-to-for-directories-on-the-same-level
def relpath(path_to, path_from):
    path_to = Path(path_to).resolve()
    path_from = Path(path_from).resolve()
    try:
        for p in (*reversed(path_from.parents), path_from):
            head, tail = p, path_to.relative_to(p)
    except ValueError:  # Stop when the paths diverge.
        pass
    return Path("../" * (len(path_from.parents) - len(head.parents))).joinpath(tail)


class TestCommonExpandSysDir(unittest.TestCase):
    def setUp(self) -> None:
        self.match_file = Path("type.raw")
        Path("test_sys").mkdir()
        self.dir = Path("test_sys")
        self.dira = Path("test_sys/a")
        self.dirb = Path("test_sys/a/b")
        self.dirc = Path("test_sys/c")
        self.dird = Path("test_sys/c/d")
        self.dire = Path("test_sys/c/type.raw")
        self.dira.mkdir()
        self.dirb.mkdir()
        self.dirc.mkdir()
        for ii in [self.dir, self.dira, self.dirb]:
            (ii / self.match_file).touch()
        relb = relpath(self.dirb, self.dirc)
        absb = self.dirb.resolve()
        self.dird.symlink_to(relb)
        self.dire.symlink_to(absb)
        self.expected_out = [
            "test_sys",
            "test_sys/a",
            "test_sys/a/b",
            "test_sys/c/d",
            "test_sys/c/type.raw",
        ]
        self.expected_out.sort()

    def tearDown(self) -> None:
        shutil.rmtree("test_sys")

    def test_expand(self) -> None:
        ret = expand_sys_str("test_sys")
        ret.sort()
        self.assertEqual(ret, self.expected_out)


class TestCastPrecision(unittest.TestCase):
    """This class tests `deepmd.tf.common.cast_precision`."""

    @property
    def precision(self):
        if GLOBAL_TF_FLOAT_PRECISION == tf.float32:
            return tf.float64
        return tf.float32

    def test_cast_precision(self) -> None:
        x = tf.zeros(1, dtype=GLOBAL_TF_FLOAT_PRECISION)
        y = tf.zeros(1, dtype=tf.int64)
        self.assertEqual(x.dtype, GLOBAL_TF_FLOAT_PRECISION)
        self.assertEqual(y.dtype, tf.int64)
        x, y, z = self._inner_method(x, y)
        self.assertEqual(x.dtype, GLOBAL_TF_FLOAT_PRECISION)
        self.assertEqual(y.dtype, tf.int64)
        self.assertIsInstance(z, bool)

    @cast_precision
    def _inner_method(self, x: tf.Tensor, y: tf.Tensor, z: bool = False) -> tf.Tensor:
        # y and z should not be cast here
        self.assertEqual(x.dtype, self.precision)
        self.assertEqual(y.dtype, tf.int64)
        self.assertIsInstance(z, bool)
        return x, y, z
