# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

from deepmd.pt.entrypoints import (
    compress,
)


class TestCompressEntrypoint(unittest.TestCase):
    def test_customized_op_guard_raises_when_op_is_missing(self) -> None:
        # The command used to continue far enough to save a scripted model whose
        # compressed descriptor path only failed later at inference time.  Patch
        # the module-level flag so this regression test is independent of how the
        # local test wheel was built.
        with patch.object(compress, "ENABLE_CUSTOMIZED_OP", False):
            with self.assertRaisesRegex(
                RuntimeError,
                "libdeepmd_op_pt.*`dp --pt compress`",
            ):
                compress.assert_customized_op_available_for_compression()

    def test_customized_op_guard_allows_available_op(self) -> None:
        with patch.object(compress, "ENABLE_CUSTOMIZED_OP", True):
            compress.assert_customized_op_available_for_compression()


if __name__ == "__main__":
    unittest.main()
