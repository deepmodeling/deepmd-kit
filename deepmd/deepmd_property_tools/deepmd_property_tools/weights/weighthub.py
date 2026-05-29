# SPDX-License-Identifier: LGPL-3.0-or-later
"""Local pretrained-weight path helper."""

from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)
from typing import (
    Any,
)


class WeightHub:
    def __init__(
        self, root: str | Path = ".", cache_dir: str | Path | None = None
    ) -> None:
        self.root = Path(root)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else self.root / "pretrained_models"
        )

    def get(self, name_or_path: str | Path) -> str:
        path = Path(name_or_path)
        if path.exists():
            print(f"Using local pretrained model: {path.resolve()}")
            return str(path)
        candidate = self.root / path
        if candidate.exists():
            print(f"Using local pretrained model: {candidate.resolve()}")
            return str(candidate)
        model_registry = self._model_registry()
        model_name = self._resolve_model_name(path, model_registry)
        if model_name is not None:
            from deepmd.pretrained.download import (
                resolve_model_path,
            )

            filename = str(model_registry[model_name]["filename"])
            expected_path = self.cache_dir / filename
            was_cached = expected_path.exists()
            resolved_path = resolve_model_path(model_name, cache_dir=self.cache_dir)
            action = "Using cached" if was_cached else "Downloaded"
            print(f"{action} pretrained model: {resolved_path}")
            return str(resolved_path)
        available = ", ".join(sorted(model_registry))
        raise FileNotFoundError(
            f"Pretrained model not found: {name_or_path}. Available built-in models: {available}"
        )

    @staticmethod
    def _model_registry() -> dict[str, dict[str, Any]]:
        from deepmd.pretrained.registry import (
            MODEL_REGISTRY,
        )

        return MODEL_REGISTRY

    @staticmethod
    def _resolve_model_name(
        path: Path, model_registry: dict[str, dict[str, Any]]
    ) -> str | None:
        alias = path.name
        if alias in model_registry:
            return alias
        lowered = alias.lower()
        for model_name, model_info in model_registry.items():
            if lowered in {model_name.lower(), str(model_info["filename"]).lower()}:
                return model_name
        return None
