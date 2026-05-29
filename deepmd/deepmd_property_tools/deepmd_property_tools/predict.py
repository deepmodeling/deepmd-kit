# SPDX-License-Identifier: LGPL-3.0-or-later
"""High-level property prediction interface."""

from __future__ import (
    annotations,
)

import json
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
from deepmd_property_tools.data import (
    DataHub,
)
from deepmd_property_tools.predictor import (
    Predictor,
)


class PropertyPredict:
    def __init__(
        self,
        load_model: str | Path,
        type_map: list[str] | None = None,
        property_name: str | None = None,
    ) -> None:
        if not load_model:
            raise ValueError("load_model is empty")
        load_model_path = Path(load_model)
        if load_model_path.is_dir():
            self.model_dir = load_model_path
            frozen_model = load_model_path / "frozen_model.pth"
            self.load_model = (
                frozen_model
                if frozen_model.exists()
                else self._latest_checkpoint(load_model_path)
            )
        else:
            self.load_model = load_model_path
            self.model_dir = load_model_path.parent
        config = self._load_config()
        self.type_map = type_map or config.get("type_map")
        if self.type_map is None:
            raise ValueError(
                "type_map is required when property_tools_config.json is absent"
            )
        self.property_name = property_name or config.get("property_name", "Property")
        self.datahub: DataHub | None = None

    def predict(
        self,
        data: dict[str, Any] | str | Path,
        save_path: str | Path | None = None,
        metrics: str = "none",
    ) -> np.ndarray:
        del metrics
        self.datahub = DataHub(
            data=data,
            is_train=False,
            save_path=self.load_model.parent,
            property_name=self.property_name,
            property_col=None,
        )
        prefix = Path(data).stem if isinstance(data, (str, Path)) else "test"
        predictor = Predictor(
            model_path=self.load_model,
            type_map=self.type_map,
            property_name=self.property_name,
        )
        return predictor.predict(
            self.datahub.atoms,
            self.datahub.coordinates,
            self.datahub.raw_data,
            save_path=save_path,
            prefix=prefix,
        )

    def _load_config(self) -> dict[str, Any]:
        candidates = [
            self.model_dir / "property_tools_config.json",
        ]
        for path in candidates:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _latest_checkpoint(model_dir: Path) -> Path:
        candidates = sorted(
            model_dir.glob("model.ckpt-*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.append(model_dir / "model.ckpt.pt")
        for checkpoint in candidates:
            if checkpoint.exists():
                return checkpoint
        raise FileNotFoundError(
            f"No frozen_model.pth or model.ckpt*.pt checkpoint found in {model_dir}"
        )
