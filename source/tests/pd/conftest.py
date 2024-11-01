# SPDX-License-Identifier: LGPL-3.0-or-later
import paddle
import pytest


@pytest.fixture(scope="package", autouse=True)
def clear_cuda_memory(request):
    yield
    paddle.device.cuda.empty_cache()
