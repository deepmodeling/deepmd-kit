# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for data-dependent array sizes."""

import numpy as np
import pytest
import torch
from packaging.version import (
    Version,
)
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.array_api import (
    xp_hint_dynamic_size,
)

pytestmark = pytest.mark.filterwarnings("error:_check_is_size:FutureWarning")


@pytest.mark.parametrize("size", [0, 1, 3])
def test_hint_dynamic_size_eager(size: int) -> None:
    for x in (np.empty((size, 2)), torch.empty((size, 2))):
        assert xp_hint_dynamic_size(x) is None


@pytest.mark.parametrize("operation", ["nonzero", "repeat"])
@pytest.mark.skipif(
    Version(torch.__version__.split("+")[0]) < Version("2.6"),
    reason="Compact graph export requires torch >= 2.6",
)
def test_hint_dynamic_size_symbolic(operation: str) -> None:
    def fn(repeats):
        if operation == "nonzero":
            values = torch.nonzero(repeats).flatten()
        else:
            values = torch.repeat_interleave(torch.arange(repeats.shape[0]), repeats)
        xp_hint_dynamic_size(values)
        return values.reshape(values.shape[0], 1).expand(-1, 2).clone()

    inputs = (torch.tensor([2, 0, 1]),)
    traced = make_fx(fn, tracing_mode="symbolic")(*inputs)
    exported = torch.export.export(traced, inputs, strict=False).module()
    # Reuse the same graphs for empty, singleton and larger unbacked sizes.
    for repeats in (torch.tensor([0, 0, 0]), torch.tensor([0, 1, 0]), inputs[0]):
        expected = fn(repeats)
        torch.testing.assert_close(traced(repeats), expected)
        torch.testing.assert_close(exported(repeats), expected)
