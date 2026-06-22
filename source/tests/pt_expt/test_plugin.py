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
    sys.modules.pop("deepmd.pt_expt", None)

    try:
        importlib.import_module("deepmd.pt_expt")
    finally:
        sys.modules.pop("deepmd.pt_expt", None)

    assert groups == ["deepmd.pt_expt"]
    assert calls == ["load"]
