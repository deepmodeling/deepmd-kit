# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import array_api_compat
import ml_dtypes
import numpy as np

from deepmd.dpmodel.common import (
    get_xp_precision,
    safe_cast_array,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)


def _torch_available() -> bool:
    """Return True if PyTorch is importable."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class TestGetXPPrecision(unittest.TestCase):
    def test_numpy(self) -> None:
        """Test get_xp_precision with NumPy namespace.

        NumPy does not have a native bfloat16, so the fallback to
        ml_dtypes.bfloat16 is expected.
        """
        aa = np.zeros(3)
        xp = array_api_compat.array_namespace(aa)
        self.assertEqual(get_xp_precision(xp, "float16"), xp.float16)
        self.assertEqual(get_xp_precision(xp, "float32"), xp.float32)
        self.assertEqual(get_xp_precision(xp, "float64"), xp.float64)
        self.assertEqual(get_xp_precision(xp, "single"), xp.float32)
        self.assertEqual(get_xp_precision(xp, "double"), xp.float64)
        self.assertEqual(get_xp_precision(xp, "global"), GLOBAL_NP_FLOAT_PRECISION)
        self.assertEqual(get_xp_precision(xp, "default"), GLOBAL_NP_FLOAT_PRECISION)
        # NumPy array namespace does not expose bfloat16 -- falls back to ml_dtypes
        self.assertEqual(get_xp_precision(xp, "bfloat16"), ml_dtypes.bfloat16)

        # Test invalid input
        with self.assertRaises(ValueError):
            get_xp_precision(xp, "invalid_precision")

    @unittest.skipIf(not _torch_available(), "PyTorch is not installed")
    def test_torch_bfloat16(self) -> None:
        """Test get_xp_precision returns torch.bfloat16 for PyTorch namespace.

        Fixes Code scan #5638: PyTorch's native bfloat16 dtype must be returned
        instead of ml_dtypes.bfloat16, because torch tensors cannot be cast to
        ml_dtypes.bfloat16 via xp.astype().
        """
        import torch

        dev = torch.device("cpu")
        xp = array_api_compat.array_namespace(torch.zeros(3, device=dev))
        result = get_xp_precision(xp, "bfloat16")
        self.assertIs(result, torch.bfloat16)
        # Verify safe_cast_array works with bfloat16 on PyTorch backend
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=dev)
        casted = safe_cast_array(t, "float32", "bfloat16")
        self.assertEqual(casted.dtype, torch.bfloat16)

    @unittest.skipIf(not _torch_available(), "PyTorch is not installed")
    def test_torch_bfloat16_roundtrip(self) -> None:
        """Test round-trip cast float32 -> bfloat16 -> float32 on PyTorch backend.

        This verifies that the backend-native bfloat16 from get_xp_precision
        can be used for real casts without raising TypeError.
        """
        import torch

        dev = torch.device("cpu")
        xp = array_api_compat.array_namespace(torch.zeros(3, device=dev))
        bf16 = get_xp_precision(xp, "bfloat16")
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=dev)
        t_bf16 = xp.astype(t, bf16)
        self.assertEqual(t_bf16.dtype, torch.bfloat16)
        # Round-trip back to float32
        t_back = xp.astype(t_bf16, xp.float32)
        self.assertEqual(t_back.dtype, torch.float32)
