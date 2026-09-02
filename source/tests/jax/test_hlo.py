# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for metadata exposed by serialized JAX HLO models."""

from types import (
    SimpleNamespace,
)

import pytest
from typing_extensions import (
    Self,
)

from deepmd.jax.model.hlo import (
    HLO,
)
from deepmd.jax.utils import (
    serialization,
)


def test_hlo_get_nnei_uses_stored_selection() -> None:
    """Return the neighbor width encoded in an HLO model's selection metadata.

    Constructing a complete ``HLO`` instance requires valid serialized
    StableHLO artifacts.  This API only depends on the stored ``sel`` metadata,
    so bypass initialization to test that metadata contract directly.
    """
    model = HLO.__new__(HLO)
    model.sel = [6, 12, 1]

    assert model.get_nnei() == sum(model.sel)


def test_hlo_serialization_declares_dense_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JAX HLO artifact exposes its dense source execution semantics."""
    stored_data = {
        "model": {},
        "constants": {},
        "@variables": {"stablehlo": b"module"},
    }
    monkeypatch.setattr(serialization, "load_dp_model", lambda _path: stored_data)

    data = serialization.serialize_from_file("model.hlo")

    assert data["lower_input_kind"] == "nlist"


def test_checkpoint_serialization_declares_dense_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JAX training checkpoint exposes its dense source semantics."""

    class Checkpointer:
        def __init__(self, _handler: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def restore(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                state={},
                model_def_script={"model_dict": {}},
            )

    monkeypatch.setattr(serialization.ocp, "Checkpointer", Checkpointer)

    data = serialization.serialize_from_file("model.jax")

    assert data["lower_input_kind"] == "nlist"
