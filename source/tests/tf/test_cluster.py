# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest import (
    mock,
)

from deepmd.tf.cluster import (
    local,
)

kHostName = "compute-b24-1"


class FakePopen:
    def __init__(self, stdout=b"", stderr=b"", returncode=0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode

    def communicate(self):
        return self._stdout, self._stderr

    @property
    def returncode(self):
        return self._returncode


class TestGPU(unittest.TestCase):
    @mock.patch("tensorflow.compat.v1.test.is_built_with_cuda")
    @mock.patch("subprocess.Popen")
    def test_none(self, mock_Popen, mock_is_built_with_cuda) -> None:
        mock_Popen.return_value.__enter__.return_value = FakePopen(b"0", b"")
        mock_is_built_with_cuda.return_value = True
        gpus = local.get_gpus()
        self.assertIsNone(gpus)

    @mock.patch("tensorflow.compat.v1.test.is_built_with_cuda")
    @mock.patch("subprocess.Popen")
    def test_valid(self, mock_Popen, mock_is_built_with_cuda) -> None:
        mock_Popen.return_value.__enter__.return_value = FakePopen(b"2", b"")
        mock_is_built_with_cuda.return_value = True
        gpus = local.get_gpus()
        self.assertEqual(gpus, [0, 1])

    @mock.patch("tensorflow.compat.v1.test.is_built_with_cuda")
    @mock.patch("subprocess.Popen")
    def test_error(self, mock_Popen, mock_is_built_with_cuda) -> None:
        mock_Popen.return_value.__enter__.return_value = FakePopen(
            stderr=b"!", returncode=1
        )
        mock_is_built_with_cuda.return_value = True
        with self.assertRaises(RuntimeError) as cm:
            _ = local.get_gpus()
            self.assertIn("Failed to detect", str(cm.exception))

    @mock.patch("tensorflow.compat.v1.test.is_built_with_rocm", create=True)
    @mock.patch("tensorflow.compat.v1.test.is_built_with_cuda")
    def test_cpu(self, mock_is_built_with_cuda, mock_is_built_with_rocm) -> None:
        mock_is_built_with_cuda.return_value = False
        mock_is_built_with_rocm.return_value = False
        gpus = local.get_gpus()
        self.assertIsNone(gpus)


class TestLocal(unittest.TestCase):
    @mock.patch("socket.gethostname")
    def test_resource(self, mock_gethostname) -> None:
        mock_gethostname.return_value = kHostName
        nodename, nodelist, _ = local.get_resource()
        self.assertEqual(nodename, kHostName)
        self.assertEqual(nodelist, [kHostName])
