# SPDX-License-Identifier: LGPL-3.0-or-later
import gc

import pytest


@pytest.fixture(autouse=True)
def ensure_gc():
    """Wordaround to resolve OOM killed issue in the GitHub Action."""
    # https://github.com/pytest-dev/pytest/discussions/8153#discussioncomment-214812
    gc.collect()
