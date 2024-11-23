# SPDX-License-Identifier: LGPL-3.0-or-later
import array_api_compat
from array_api_strict import (
    set_array_api_strict_flags,
)


class ArrayAPITest:
    """Utils for array API tests."""

    def set_array_api_version(self, func) -> None:
        """Set the array API version for a function."""
        set_array_api_strict_flags(api_version=func.array_api_version)

    def assert_namespace_equal(self, a, b) -> None:
        """Assert two array has the same namespace."""
        self.assertEqual(
            array_api_compat.array_namespace(a), array_api_compat.array_namespace(b)
        )

    def assert_dtype_equal(self, a, b) -> None:
        """Assert two array has the same dtype."""
        self.assertEqual(a.dtype, b.dtype)

    def assert_device_equal(self, a, b) -> None:
        """Assert two array has the same device."""
        self.assertEqual(array_api_compat.device(a), array_api_compat.device(b))
