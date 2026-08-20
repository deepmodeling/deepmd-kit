# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared version guard for pt_expt ``torch.compile`` tests."""

import unittest

from deepmd.pt.utils.compile_compat import (
    check_compile_torch_version,
)

try:
    check_compile_torch_version()
except RuntimeError as error:
    _COMPILE_SUPPORT_ERROR = str(error)
else:
    _COMPILE_SUPPORT_ERROR = ""

REQUIRES_SUPPORTED_COMPILE = unittest.skipIf(
    bool(_COMPILE_SUPPORT_ERROR),
    _COMPILE_SUPPORT_ERROR,
)
