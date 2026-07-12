# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import pytest

from deepmd.backend.backend import (
    Backend,
)
from deepmd.entrypoints.convert_backend import (
    convert_backend,
)


@pytest.mark.parametrize(
    ("expected_lower_kind", "kwargs"),
    [
        ("nlist", {}),
        ("graph", {"lower_kind": "graph"}),
    ],
)
def test_convert_backend_lower_kind(
    monkeypatch: pytest.MonkeyPatch,
    expected_lower_kind: str,
    kwargs: dict[str, str],
) -> None:
    captured: dict[str, Any] = {}

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

    convert_backend(INPUT="model.input", OUTPUT="model.output", **kwargs)

    assert captured["lower_kind"] == expected_lower_kind
    assert captured["do_atomic_virial"] is False
