# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from pathlib import (
    Path,
)

import pytest

if os.environ.get("DP_CI_IMPORT_PADDLE_BEFORE_TF", "0") == "1":
    # Paddle must be loaded before TensorFlow in the CI test process.
    import paddle  # noqa: F401
    import tensorflow  # noqa: F401


# A test module is "AOTI-compiling" if it freezes a ``.pt2`` at runtime, i.e. it
# references one of the pt_expt AOTInductor freeze entry points below. Those
# tests are CPU-bound (minutes of inductor / g++ / ptxas code generation) while
# the GPU sits ~idle (measured ~98% idle, <250 MiB peak), so CI can run them
# concurrently with the GPU-bound tests that do NOT compile a ``.pt2`` -- the
# compile CPU-time then overlaps the GPU unit tests instead of serializing in
# front of them (see ``.github/workflows/test_cuda.yml``). Detecting the set by
# source scan keeps the ``aoti_compile`` partition correct as tests are added,
# without a hand-maintained file list.
_AOTI_FREEZE_SYMBOLS = (
    "deserialize_to_file",
    "_trace_and_export",
    "aoti_compile_and_package",
)
_aoti_module_cache: dict[str, bool] = {}


def _module_compiles_pt2(path: str) -> bool:
    hit = _aoti_module_cache.get(path)
    if hit is None:
        try:
            src = Path(path).read_text(encoding="utf-8")
        except OSError:
            src = ""
        hit = _aoti_module_cache[path] = any(s in src for s in _AOTI_FREEZE_SYMBOLS)
    return hit


def pytest_collection_modifyitems(items) -> None:
    """Auto-tag every test whose module freezes a ``.pt2`` with ``aoti_compile``.

    Lets the CUDA CI split the suite into a CPU-bound compile group
    (``-m aoti_compile``) and a GPU-bound group (``-m "not aoti_compile"``)
    that run concurrently on the same GPU.
    """
    for item in items:
        path = getattr(item, "path", None) or getattr(item, "fspath", None)
        if path is not None and _module_compiles_pt2(str(path)):
            item.add_marker(pytest.mark.aoti_compile)
