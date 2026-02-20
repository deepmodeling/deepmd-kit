# SPDX-License-Identifier: LGPL-3.0-or-later
import paddle
import pytest


@pytest.fixture(scope="package", autouse=True)
def clear_cuda_memory(request):
    yield
    if paddle.device.get_device() != "cpu":
        paddle.device.empty_cache()
