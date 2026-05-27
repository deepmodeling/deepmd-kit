# SPDX-License-Identifier: LGPL-3.0-or-later
"""Property inference model wrapper."""

from __future__ import annotations

from pathlib import Path


class PropertyModel:
    def __init__(self, model_path: str | Path) -> None:
        from deepmd.infer.deep_property import DeepProperty

        self.model = DeepProperty(str(model_path), no_jit=True)

    def eval(self, *args: object, **kwargs: object) -> object:
        return self.model.eval(*args, **kwargs)
