# SPDX-License-Identifier: LGPL-3.0-or-later
import subprocess

import pytest


@pytest.fixture(autouse=True)
def log_gpu_memory(request):
    yield  # Run the test
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        memory_used = result.stdout.strip()
        print(f"\nGPU Memory after test '{request.node.name}': {memory_used} MB")  # noqa:T201
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nFailed to get GPU memory: {e}")  # noqa:T201
