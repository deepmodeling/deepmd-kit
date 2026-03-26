# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conftest for pt_expt tests.

Safety net: pops any leaked ``torch.utils._device.DeviceContext`` modes
from the torch function mode stack before each test.

The primary leak source was ``source/tests/pt/__init__.py`` which calls
``torch.set_default_device("cuda:9999999")``; pt_expt tests previously
imported shared mixins from ``tests.pt.model``, triggering that init.
This was fixed by moving the shared mixins to ``tests.common.test_mixins``
so pt_expt tests no longer import from the ``tests.pt`` package.
"""

import pytest
import torch.utils._device as _device
from torch.overrides import (
    _get_current_function_mode_stack,
)


def _pop_device_contexts() -> list:
    """Pop all stale DeviceContext modes from the torch function mode stack."""
    popped = []
    while True:
        modes = _get_current_function_mode_stack()
        if not modes:
            break
        top = modes[-1]
        if isinstance(top, _device.DeviceContext):
            top.__exit__(None, None, None)
            popped.append(top)
        else:
            break
    return popped


@pytest.fixture(autouse=True)
def _clear_leaked_device_context():
    """Pop any stale ``DeviceContext`` before each test (safety net)."""
    _pop_device_contexts()
    yield
