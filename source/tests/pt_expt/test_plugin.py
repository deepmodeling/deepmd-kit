# SPDX-License-Identifier: LGPL-3.0-or-later

import importlib
import importlib.metadata
import sys


class _FakeEntryPoint:
    def __init__(self, calls):
        self.calls = calls

    def load(self):
        self.calls.append("load")


def test_pt_expt_loads_plugin_entry_points(monkeypatch):
    groups = []
    calls = []

    def fake_entry_points(*, group=None):
        groups.append(group)
        return [_FakeEntryPoint(calls)]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    # Snapshot the deepmd.pt_expt module tree BEFORE re-importing. Just popping
    # "deepmd.pt_expt" and leaving its submodules cached poisons sys.modules
    # for the rest of the pytest process: a later import of a cached submodule
    # (e.g. deepmd.pt_expt.infer.deep_eval) re-creates a BARE parent package
    # whose submodule attributes (utils/infer/...) are never rebound, and
    # mock.patch("deepmd.pt_expt.utils...") then fails with AttributeError on
    # py3.10 (shard-order dependent CI failure).
    saved = {
        k: v
        for k, v in sys.modules.items()
        if k == "deepmd.pt_expt" or k.startswith("deepmd.pt_expt.")
    }
    deepmd_pkg = sys.modules.get("deepmd")
    sys.modules.pop("deepmd.pt_expt", None)

    try:
        importlib.import_module("deepmd.pt_expt")
    finally:
        # drop everything the fresh import created, then restore the snapshot
        # (including the parent-package attribute binding).
        for k in [
            m
            for m in list(sys.modules)
            if m == "deepmd.pt_expt" or m.startswith("deepmd.pt_expt.")
        ]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        if deepmd_pkg is not None and "deepmd.pt_expt" in saved:
            deepmd_pkg.pt_expt = saved["deepmd.pt_expt"]

    assert groups == ["deepmd.pt_expt"]
    assert calls == ["load"]
