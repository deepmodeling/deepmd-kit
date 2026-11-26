# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checkpoint metadata and portable JAX array restoration."""

from pathlib import (
    Path,
)

import numpy as np
import orbax.checkpoint as ocp
import pytest
from typing_extensions import (
    Self,
)

from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.model.hlo import (
    HLO,
)
from deepmd.jax.utils import (
    serialization,
)
from deepmd.jax.utils.checkpoint_state import (
    read_checkpoint_state,
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
    tmp_path: Path,
) -> None:
    """A JAX HLO artifact exposes its dense source execution semantics."""
    stored_data = {
        "model": {},
        "constants": {},
        "@variables": {"stablehlo": b"module"},
    }
    monkeypatch.setattr(serialization, "load_dp_model", lambda _path: stored_data)

    path = tmp_path / "model.hlo"
    path.touch()
    data = serialization.serialize_from_file(str(path))

    assert data["lower_input_kind"] == "nlist"


def test_checkpoint_serialization_declares_dense_lower(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A JAX training checkpoint exposes its dense source semantics."""

    class Checkpointer:
        def __init__(self, _handler: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def restore(self, *_args: object, **_kwargs: object) -> dict[str, dict]:
            return {"model_dict": {}}

    monkeypatch.setattr(serialization.ocp, "Checkpointer", Checkpointer)
    monkeypatch.setattr(serialization, "read_checkpoint_state", lambda _path: {})

    path = tmp_path / "model.jax"
    path.mkdir()
    data = serialization.serialize_from_file(str(path))

    assert data["lower_input_kind"] == "nlist"


@pytest.mark.parametrize("standard", [False, True])
def test_checkpoint_arrays_restore_without_saved_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standard: bool,
) -> None:
    """Host conversion retains arrays and metadata when saved devices vanish."""
    state = {
        "matrix": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "clock": jnp.asarray(7, dtype=jnp.uint32),
        "empty": None,
    }
    if not standard:
        state["metadata"] = {"name": "HybridMuon", "step": 7, "enabled": True}
    handler = (
        ocp.StandardCheckpointHandler() if standard else ocp.PyTreeCheckpointHandler()
    )
    save_args = ocp.args.StandardSave(state) if standard else ocp.args.PyTreeSave(state)
    path = tmp_path / "state"
    with ocp.Checkpointer(handler) as checkpointer:
        checkpointer.save(path, args=save_args)
    monkeypatch.setattr(jax, "local_devices", lambda *args, **kwargs: [])
    restored = read_checkpoint_state(path)
    for name in ("matrix", "clock"):
        assert isinstance(restored[name], np.ndarray)
        np.testing.assert_array_equal(restored[name], state[name])
        assert restored[name].dtype == state[name].dtype
    assert restored["empty"] is None
    if not standard:
        assert restored["metadata"] == state["metadata"]
        for name, value in state["metadata"].items():
            assert type(restored["metadata"][name]) is type(value)


def test_native_checkpoint_preserves_hessian_output(tmp_path: Path) -> None:
    """A native checkpoint's Hessian flag selects real, conservative derivatives."""
    from deepmd.infer import (
        DeepPot,
    )
    from deepmd.jax.model.model import (
        get_model,
    )

    config = {
        "type_map": ["H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [2],
            "rcut": 3.0,
            "rcut_smth": 2.0,
            "neuron": [2],
            "axis_neuron": 1,
            "seed": 1,
        },
        "fitting_net": {"neuron": [2], "seed": 2},
    }
    model = get_model(config)
    path = tmp_path / "hessian.jax"
    serialization.deserialize_to_file(
        str(path),
        {"model": model.serialize(), "model_def_script": config},
        hessian=True,
    )
    coord = np.array([[[0.0, 0.0, 0.0], [0.8, 0.3, 0.2]]])
    prediction = DeepPot(str(path), auto_batch_size=False).eval(coord, None, [0, 0])
    hessian = prediction[-1]
    assert hessian.shape == (1, 6, 6)
    assert np.isfinite(hessian).all()
    assert np.linalg.norm(hessian) > 1e-8
    np.testing.assert_allclose(hessian, hessian.transpose(0, 2, 1), atol=1e-10)
    np.testing.assert_allclose(
        hessian.reshape(1, 2, 3, 2, 3).sum(axis=3), 0.0, atol=1e-10
    )
