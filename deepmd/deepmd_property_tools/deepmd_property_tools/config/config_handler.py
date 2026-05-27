# SPDX-License-Identifier: LGPL-3.0-or-later
"""JSON config handler."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ConfigHandler:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else Path(__file__).with_name("default.json")

    def read(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def write(self, data: dict[str, Any], out_file_path: str | Path) -> None:
        Path(out_file_path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def merge(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
        result = copy.deepcopy(base)
        if updates:
            _deep_update(result, updates)
        return result


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
