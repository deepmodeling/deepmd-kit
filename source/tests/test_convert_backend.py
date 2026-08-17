# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest

from deepmd.backend.backend import (
    Backend,
)
from deepmd.entrypoints.convert_backend import (
    convert_backend,
)


def test_convert_backend_uses_auto_for_unannotated_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class InputBackend:
        name = "input"

        @staticmethod
        def serialize_hook(path: str) -> dict[str, str]:
            return {"path": path}

    class OutputBackend:
        name = "output"

        @staticmethod
        def deserialize_hook(
            path: str,
            data: dict[str, str],
            *,
            lower_kind: str = "nlist",
            do_atomic_virial: bool = False,
        ) -> None:
            captured.update(
                path=path,
                data=data,
                lower_kind=lower_kind,
                do_atomic_virial=do_atomic_virial,
            )

    def detect_backend(path: str) -> type[InputBackend] | type[OutputBackend]:
        return InputBackend if path.endswith(".input") else OutputBackend

    monkeypatch.setattr(Backend, "detect_backend_by_model", detect_backend)

    convert_backend(INPUT="model.input", OUTPUT="model.output")

    assert captured["lower_kind"] == "auto"
    assert captured["do_atomic_virial"] is False


@pytest.mark.parametrize(
    "lower_input_kind",
    ["nlist", "graph", "dpa1_canonical", "dpa4c_canonical", "edge_vec"],
)
def test_convert_backend_preserves_explicit_lower_kind(
    monkeypatch: pytest.MonkeyPatch,
    lower_input_kind: str,
) -> None:
    captured: dict[str, object] = {}

    class InputBackend:
        name = "input"

        @staticmethod
        def serialize_hook(path: str) -> dict[str, str]:
            return {"path": path, "lower_input_kind": lower_input_kind}

    class OutputBackend:
        name = "output"

        @staticmethod
        def deserialize_hook(
            path: str,
            data: dict[str, str],
            *,
            lower_kind: str = "nlist",
        ) -> None:
            captured.update(path=path, data=data, lower_kind=lower_kind)

    def detect_backend(path: str) -> type[InputBackend] | type[OutputBackend]:
        return InputBackend if path.endswith(".input") else OutputBackend

    monkeypatch.setattr(Backend, "detect_backend_by_model", detect_backend)

    convert_backend(INPUT="model.input", OUTPUT="model.output")

    assert captured["lower_kind"] == lower_input_kind


@pytest.mark.parametrize("lower_input_kind", [None, "nlist"])
def test_convert_backend_allows_dense_compatible_source_for_plain_output(
    monkeypatch: pytest.MonkeyPatch,
    lower_input_kind: str | None,
) -> None:
    captured: dict[str, object] = {}

    class InputBackend:
        name = "input"

        @staticmethod
        def serialize_hook(path: str) -> dict[str, str]:
            data = {"path": path}
            if lower_input_kind is not None:
                data["lower_input_kind"] = lower_input_kind
            return data

    class OutputBackend:
        name = "output"

        @staticmethod
        def deserialize_hook(path: str, data: dict[str, str]) -> None:
            captured.update(path=path, data=data)

    def detect_backend(path: str) -> type[InputBackend] | type[OutputBackend]:
        return InputBackend if path.endswith(".input") else OutputBackend

    monkeypatch.setattr(Backend, "detect_backend_by_model", detect_backend)

    convert_backend(INPUT="model.input", OUTPUT="model.output")

    assert captured["path"] == "model.output"


def test_convert_backend_rejects_graph_for_dense_only_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InputBackend:
        name = "input"

        @staticmethod
        def serialize_hook(path: str) -> dict[str, str]:
            return {"path": path, "lower_input_kind": "graph"}

    class OutputBackend:
        name = "output"

        @staticmethod
        def deserialize_hook(path: str, data: dict[str, str]) -> None:
            raise AssertionError("dense-only output hook must not be called")

    def detect_backend(path: str) -> type[InputBackend] | type[OutputBackend]:
        return InputBackend if path.endswith(".input") else OutputBackend

    monkeypatch.setattr(Backend, "detect_backend_by_model", detect_backend)

    with pytest.raises(ValueError, match="Cannot preserve lower_input_kind 'graph'"):
        convert_backend(INPUT="model.input", OUTPUT="model.output")
