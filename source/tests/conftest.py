# SPDX-License-Identifier: LGPL-3.0-or-later

import os

import pytest

if os.environ.get("DP_CI_IMPORT_PADDLE_BEFORE_TF", "0") == "1":
    # Paddle must be loaded before TensorFlow in the CI test process.
    import paddle  # noqa: F401
    import tensorflow  # noqa: F401


# --- aoti_compile partition: explicit producer list + runtime guard ----------
#
# Tests that freeze a ``.pt2`` (torch.export / make_fx trace + AOTInductor
# compile) are CPU-bound: the GPU sits ~idle (measured ~98% idle, <250 MiB)
# while inductor / g++ / ptxas generate code. The CUDA CI marks them
# ``aoti_compile`` and runs them in a CPU lane concurrent with the GPU-bound
# tests (see ``.github/workflows/test_cuda.yml``).
#
# The producer set is listed EXPLICITLY below rather than inferred from test
# source: a source grep both MISSES ``freeze(...)``-driven compiles (e.g.
# ``test_dp_freeze`` / ``test_finetune`` never name the low-level entry points)
# and OVER-tags files that only mock the compiler or produce ``.pth`` / ``.pte``.
# The list was derived by running the suite under a detector that wraps the real
# compile entry points; the always-on ``pytest_sessionfinish`` guard below
# re-checks it every run and fails if any test freezes a ``.pt2`` without the
# marker, so the list cannot silently drift out of date.
_AOTI_COMPILE_MODULES = frozenset(
    {
        # direct AOTInductor / _trace_and_export API
        "pt_expt/descriptor/test_dpa1_cuda.py",
        "pt_expt/model/test_model_compression.py",
        "pt_expt/model/test_dpa4_export.py",
        "pt_expt/model/test_export_pipeline.py",
        "pt_expt/model/test_export_with_comm.py",
        "pt_expt/model/test_dpa1_graph_lower.py",
        "pt_expt/model/test_graph_export.py",
        "pt_expt/model/test_graph_export_with_comm.py",
        "pt_expt/utils/test_graph_pt2_metadata.py",
        "pt_expt/infer/test_deep_eval_metadata_only.py",
        "pt_expt/infer/test_deep_eval_pt_checkpoint.py",
        "pt_expt/infer/test_deep_eval_spin.py",
        "pt_expt/infer/test_deep_eval.py",
        "pt_expt/infer/test_deep_eval_property.py",
        "pt_expt/infer/test_dpa4_deep_eval.py",
        "pt_expt/infer/test_graph_deepeval.py",
        # freeze(...) / finetune / multitask driven (a grep on the low-level
        # entry points misses these; confirmed by the runtime detector)
        "pt_expt/test_dp_freeze.py",
        "pt_expt/test_finetune.py",
        "pt_expt/test_multitask.py",
        "pt_expt/test_dp_test.py",
        "pt_expt/test_change_bias.py",
        "infer/test_models.py",
        # pt / jax producers
        "pt/model/test_sezm_export.py",
        "pt/model/test_sezm_spin_model.py",
        "jax/test_deep_dos.py",
        "jax/test_training.py",
    }
)


def _tests_relpath(path: object) -> str:
    """Normalize an item path to ``<dir>/.../test_x.py`` under ``source/tests``."""
    p = str(path).replace(os.sep, "/")
    marker = "source/tests/"
    i = p.rfind(marker)
    return p[i + len(marker) :] if i >= 0 else p


def pytest_collection_modifyitems(items) -> None:
    """Tag tests in the explicit producer modules with ``aoti_compile``."""
    for item in items:
        path = getattr(item, "path", None) or getattr(item, "fspath", None)
        if path is not None and _tests_relpath(path) in _AOTI_COMPILE_MODULES:
            item.add_marker(pytest.mark.aoti_compile)


# --- guard: fail-closed if a .pt2 compile escapes the aoti_compile marker -----
_UNMARKED_COMPILERS: set[str] = set()
_current_item: dict = {"item": None}


def pytest_configure(config) -> None:
    """Wrap the real freeze/compile entry points to record any test that
    triggers one without the ``aoti_compile`` marker (drift detection).
    """
    try:
        import torch._inductor as ti

        import deepmd.pt_expt.utils.serialization as ser
    except Exception:
        return  # pt_expt / inductor unavailable -> nothing can compile a .pt2

    def wrap(mod, name):
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "_dp_aoti_guard", False):
            return

        def probe(*args, **kwargs):
            item = _current_item["item"]
            if item is not None and item.get_closest_marker("aoti_compile") is None:
                _UNMARKED_COMPILERS.add(item.nodeid)
            return orig(*args, **kwargs)

        probe._dp_aoti_guard = True
        setattr(mod, name, probe)

    # aoti_compile_and_package covers every real .pt2 compile (incl. freeze /
    # finetune driven, re-imported from torch._inductor at call time);
    # _trace_and_export covers the trace-only export tests.
    wrap(ti, "aoti_compile_and_package")
    wrap(ser, "_trace_and_export")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _current_item["item"] = item
    try:
        yield
    finally:
        _current_item["item"] = None


def pytest_sessionfinish(session, exitstatus) -> None:
    if _UNMARKED_COMPILERS and session.exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not _UNMARKED_COMPILERS:
        return
    terminalreporter.section("aoti_compile marker drift")
    terminalreporter.write_line(
        "These tests froze a .pt2 but lack the 'aoti_compile' marker, so the "
        "CUDA CI would run their compile in the GPU lane. Add each one's module "
        "to _AOTI_COMPILE_MODULES in source/tests/conftest.py:"
    )
    for nodeid in sorted(_UNMARKED_COMPILERS):
        terminalreporter.write_line(f"  {nodeid}")
