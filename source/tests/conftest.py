# SPDX-License-Identifier: LGPL-3.0-or-later
import gc

import pytest


@pytest.fixture(scope="package", autouse=True)
def automatic_memory_release():
    """Release memory after each package."""
    # pre
    yield
    # post
    gc.collect()
