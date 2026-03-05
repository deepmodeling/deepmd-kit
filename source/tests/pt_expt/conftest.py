# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conftest for pt_expt tests.

Clears any leaked ``torch.utils._device.DeviceContext`` modes that may
have been left on the torch function mode stack by ``make_fx`` or other
tracing utilities during test collection.  A stale ``DeviceContext``
silently reroutes ``torch.tensor(...)`` calls (without an explicit
``device=``) to a fake CUDA device, causing spurious "no NVIDIA driver"
errors on CPU-only machines.

The leak is triggered when pytest collects descriptor test modules that
import ``make_fx``.  A ``DeviceContext(cuda:127)`` ends up on the
``torch.overrides`` function mode stack and is never popped.

Our own code (``display_if_exist`` in ``deepmd/dpmodel/loss/loss.py``)
is already fixed to pass ``device=`` explicitly.  However, PyTorch's
``Adam._init_group`` (``torch/optim/adam.py``) contains::

    torch.tensor(0.0, dtype=_get_scalar_dtype())  # no device=

on the ``capturable=False, fused=False`` path (the default).  This is
a PyTorch bug — the ``capturable=True`` branch correctly uses
``device=p.device`` but the default branch omits it.  We cannot fix
PyTorch internals, so this fixture works around the issue by popping
leaked ``DeviceContext`` modes before each test.
"""

import pytest
import torch.utils._device as _device
from torch.overrides import (
    _get_current_function_mode_stack,
)


@pytest.fixture(autouse=True)
def _clear_leaked_device_context():
    """Pop any stale ``DeviceContext`` before each test, restore after."""
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
    yield
    # Restore in reverse order so the stack is back to its original state.
    for ctx in reversed(popped):
        ctx.__enter__()
