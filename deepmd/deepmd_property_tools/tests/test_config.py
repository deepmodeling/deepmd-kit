# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from deepmd_property_tools.config import (
    ConfigHandler,
)


def test_merge_deep_updates_nested_dicts() -> None:
    base = {"training": {"numb_steps": 10, "data": {"batch_size": 1}}, "loss": "mae"}
    updates = {"training": {"data": {"batch_size": 4}}}

    merged = ConfigHandler.merge(base, updates)

    assert merged["training"]["numb_steps"] == 10
    assert merged["training"]["data"]["batch_size"] == 4
    assert base["training"]["data"]["batch_size"] == 1
