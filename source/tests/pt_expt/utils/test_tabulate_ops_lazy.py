# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for lazy fake-op registration in ``pt_expt.utils``.

Two failure modes, both surfaced when running ``dp test`` on the plain pt
(torch.jit) backend in an environment WITHOUT the C++ custom op library
(``libdeepmd_op_pt.so``):

1. ``deepmd.pt.infer.deep_eval`` imports the vesin neighbor list from
   ``deepmd.pt_expt.utils``. If that package eagerly imported ``tabulate_ops``
   (which registers fake custom ops at import time), plain pt inference would
   drag custom-op registration onto its path.

2. When the C++ op library is absent, the pt descriptor fallbacks monkeypatch a
   plain Python function onto ``torch.ops.deepmd.<op>`` (see e.g.
   ``deepmd/pt/model/descriptor/se_a.py``). A bare ``hasattr`` guard then
   returns True even though no real dispatcher op exists, and
   ``register_fake`` raises ``RuntimeError: operator deepmd::... does not
   exist``, crashing the import.
"""

import subprocess
import sys
import textwrap

import torch

from deepmd.pt_expt.utils import (
    tabulate_ops,
)


def test_pt_deep_eval_does_not_eager_import_tabulate_ops() -> None:
    """Importing the plain pt inference entry must not pull in tabulate_ops.

    Run in a fresh interpreter so ``sys.modules`` is not polluted by the test
    session. Guards against re-introducing the eager
    ``from deepmd.pt_expt.utils import tabulate_ops`` in the package ``__init__``.
    """
    code = textwrap.dedent(
        """
        import sys
        import deepmd.pt.infer.deep_eval  # noqa: F401

        leaked = [
            m
            for m in (
                "deepmd.pt_expt.utils.tabulate_ops",
                "deepmd.pt_expt.utils.comm",
            )
            if m in sys.modules
        ]
        assert not leaked, f"eagerly imported custom-op modules: {leaked}"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "OK" in result.stdout


def test_ensure_fake_registered_skips_monkeypatched_fallback() -> None:
    """``ensure_fake_registered`` must skip a monkeypatched plain-function op.

    Simulates the no-C++-op-library state by installing a plain Python function
    on ``torch.ops.deepmd.tabulate_fusion_se_a`` (exactly what the pt descriptor
    fallback does). With the old bare-``hasattr`` guard this raised
    ``RuntimeError: operator ... does not exist``; the fix must detect that it is
    not a real ``OpOverloadPacket`` and skip it without raising.
    """
    op_name = "tabulate_fusion_se_a"
    qualname = "deepmd::" + op_name
    ns = torch.ops.deepmd

    # Snapshot any existing (possibly cached real op) attribute so we can restore.
    had_attr = op_name in ns.__dict__
    saved = ns.__dict__.get(op_name)
    # ensure_fake_registered() may touch several op names; snapshot the whole set.
    saved_registered = set(tabulate_ops._registered)

    def _fallback(*args, **kwargs):
        raise NotImplementedError

    try:
        # Install the plain-function fallback (mimics the no-op-lib descriptor hack).
        setattr(ns, op_name, _fallback)
        # It must NOT be recognised as a real dispatcher op.
        assert not tabulate_ops._op_exists(op_name)

        # Force a registration attempt for this op.
        tabulate_ops._registered.discard(qualname)

        # The crash repro: must complete without raising.
        tabulate_ops.ensure_fake_registered()

        # The monkeypatched fallback must have been skipped, not registered.
        assert qualname not in tabulate_ops._registered
    finally:
        if had_attr:
            setattr(ns, op_name, saved)
        else:
            ns.__dict__.pop(op_name, None)
        tabulate_ops._registered.clear()
        tabulate_ops._registered.update(saved_registered)
