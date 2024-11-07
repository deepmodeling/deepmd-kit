# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest

from ...utils import (
    DP_TEST_TF2_ONLY,
)

pytest.mark.skipif(
    not DP_TEST_TF2_ONLY, reason="TF2 conflicts with TF1", allow_module_level=True
)
