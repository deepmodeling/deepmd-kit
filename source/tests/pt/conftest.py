# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import pytest
import torch


@pytest.fixture(scope="package", autouse=True)
def clear_cuda_memory(request):
    yield
    torch.cuda.empty_cache()
