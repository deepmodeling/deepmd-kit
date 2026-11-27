# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checkpoint and StableHLO inference parity for DPA4 model families."""

import sys
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from typing import (
    Any,
)

import numpy as np
import pytest
import torch

from deepmd.infer import (
    DeepPot,
)
from deepmd.jax.common import (
    as_pure,
    to_flat_state,
)
from deepmd.jax.entrypoints.freeze import (
    freeze,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.jax.utils.serialization import (
    deserialize_to_file,
)

from .test_dpa4_model import (
    _reference,
    model_config,
)


@pytest.mark.parametrize("suffix", [".jax", ".hlo"])
@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("bridging", [False, True])
def test_checkpoint_and_hlo_inference(
    tmp_path: Path,
    suffix: str,
    spin: bool,
    bridging: bool,
) -> None:
    """All model inputs and atom-resolved outputs survive export and reload."""
    config = model_config(spin, bridging)
    config["descriptor"]["add_chg_spin_ebd"] = True
    config["descriptor"]["default_chg_spin"] = [0.0, 1.0]
    reference = _reference(config)
    output = tmp_path / f"model{suffix}"
    deserialize_to_file(
        str(output),
        {
            "model": reference.serialize(),
            "model_def_script": config,
            "min_nbor_dist": 0.7,
        },
    )
    potential = DeepPot(str(output), auto_batch_size=False)
    assert potential.get_model().get_min_nbor_dist() == 0.7
    assert potential.deep_eval.get_has_spin() is spin
    assert potential.deep_eval.has_chg_spin_ebd()

    for nframes, natoms, periodic in ((1, 2, False), (2, 3, True)):
        generator = np.random.default_rng(20261023)
        coord = generator.uniform(0.0, 1.5, (nframes, natoms, 3))
        atype = np.tile(np.arange(natoms) % 2, (nframes, 1))
        spins = generator.normal(size=coord.shape)
        box = (
            np.tile((np.eye(3) * 3.0).reshape(1, 9), (nframes, 1)) if periodic else None
        )
        charge_spin = np.tile([[1.0, 2.0]], (nframes, 1))
        expected = reference(
            torch.tensor(coord),
            torch.tensor(atype),
            box=torch.tensor(box) if box is not None else None,
            charge_spin=torch.tensor(charge_spin),
            do_atomic_virial=True,
            **({"spin": torch.tensor(spins)} if spin else {}),
        )
        actual = potential.eval(
            coord,
            box,
            atype,
            atomic=True,
            mixed_type=True,
            charge_spin=charge_spin,
            **({"spin": spins} if spin else {}),
        )
        keys = ["energy", "force", "virial", "atom_energy", "atom_virial"]
        if spin:
            keys += ["force_mag", "mask_mag"]
        for value, key in zip(actual, keys, strict=True):
            np.testing.assert_allclose(
                value,
                expected[key].detach().numpy(),
                rtol=2e-8,
                atol=2e-9,
                err_msg=f"{suffix}: {key}",
            )


def test_neighbor_list_checkpoint_forwards_charge_spin(tmp_path: Path) -> None:
    """A checkpoint evaluated on a neighbor list reads the requested condition."""
    from deepmd.utils.argcheck import (
        model_args,
    )

    repflow = {
        "n_dim": 8,
        "e_dim": 6,
        "a_dim": 4,
        "nlayers": 1,
        "e_rcut": 4.0,
        "e_rcut_smth": 3.5,
        "e_sel": 8,
        "a_rcut": 3.0,
        "a_rcut_smth": 2.5,
        "a_sel": 6,
        "axis_neuron": 2,
    }
    config = model_args().normalize_value(
        {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa3",
                "repflow": repflow,
                "precision": "float64",
                "seed": 1,
                "add_chg_spin_ebd": True,
                "default_chg_spin": [0.0, 1.0],
            },
            "fitting_net": {"neuron": [6], "precision": "float64", "seed": 1},
        },
        trim_pattern="_*",
    )
    model = get_model(config)
    path = tmp_path / "model.jax"
    deserialize_to_file(
        str(path), {"model": model.serialize(), "model_def_script": config}
    )
    potential = DeepPot(str(path), auto_batch_size=False)
    coord = np.array([[0.0, 0.0, 0.0, 0.9, 0.3, 0.1, 1.5, 1.0, 0.4]])
    atype = [0, 1, 1]
    for condition in ([[0.0, 1.0]], [[2.0, 3.0]]):
        energy = potential.eval(coord, None, atype, charge_spin=np.array(condition))[0]
        expected = model(
            jnp.asarray(coord.reshape(1, 3, 3)),
            jnp.asarray([atype]),
            charge_spin=jnp.asarray(condition),
        )["energy"]
        np.testing.assert_allclose(
            energy.reshape(-1), np.asarray(expected).reshape(-1), rtol=1e-10
        )


def test_unconditioned_hlo_uses_original_abi(tmp_path: Path) -> None:
    """The six-argument lower ABI remains usable without conditioning inputs."""
    config = model_config()
    reference = _reference(config)
    path = tmp_path / "unconditioned.hlo"
    deserialize_to_file(
        str(path),
        {"model": reference.serialize(), "model_def_script": config},
    )
    coord = np.array([[[0.0, 0.0, 0.0], [0.8, 0.3, 0.2]]], dtype=np.float32)
    expected = reference(
        torch.tensor(coord, dtype=torch.float64), torch.tensor([[0, 1]])
    )
    actual = DeepPot(str(path), auto_batch_size=False).eval(coord, None, [0, 1])
    for value, key in zip(actual, ("energy", "force", "virial"), strict=True):
        np.testing.assert_allclose(
            value, expected[key].detach().numpy(), rtol=2e-8, atol=2e-9
        )


def test_hlo_preserves_bfloat16_inference_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symbolic export retains mixed-precision energy and force evaluation."""
    monkeypatch.setenv("DP_AMP_INFER", "1")
    config = model_config()
    config["descriptor"].update(precision="float32", mlp_bias=True)
    config["fitting_net"]["precision"] = "float32"
    from deepmd.jax.model.base_model import (
        BaseModel,
    )

    record = _reference(config).serialize()
    model = BaseModel.deserialize(record)
    model.eval()
    coord = jnp.asarray(
        [[[0.0, 0.0, 0.0], [0.8, 0.3, 0.2], [0.3, 1.2, 0.6]]], dtype=jnp.float64
    )
    atype = jnp.asarray([[0, 1, 0]], dtype=jnp.int32)
    nlist = jnp.asarray([[[1, 2], [0, 2], [0, 1]]], dtype=jnp.int64)
    mapping = jnp.asarray([[0, 1, 2]], dtype=jnp.int64)
    expected = jax.jit(model.call_common_lower)(coord, atype, nlist, mapping)
    path = tmp_path / "amp.hlo"
    deserialize_to_file(str(path), {"model": record, "model_def_script": config})
    frozen = DeepPot(str(path), auto_batch_size=False).deep_eval.dp
    actual = frozen.call_lower(coord, atype, nlist, mapping)
    for key in ("energy", "energy_redu", "energy_derv_r", "energy_derv_c_redu"):
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-5, atol=1e-6, err_msg=key
        )


def test_multitask_checkpoint_requires_and_selects_head(tmp_path: Path) -> None:
    """A named inference head retains its own checkpoint parameters."""
    configurations = {"first": model_config(), "second": model_config()}
    configurations["second"]["descriptor"]["seed"] = 19
    references = {name: _reference(config) for name, config in configurations.items()}
    configurations["first"]["_min_nbor_dist"] = 0.6
    configurations["second"]["_min_nbor_dist"] = 0.7
    path = tmp_path / "multitask.jax"
    deserialize_to_file(
        str(path),
        {
            "model": {
                "model_dict": {
                    name: model.serialize() for name, model in references.items()
                }
            },
            "model_def_script": {"model_dict": configurations},
        },
    )
    with pytest.raises(ValueError, match="requires a head name"):
        DeepPot(str(path), auto_batch_size=False)
    coord = np.array([[[0.0, 0.0, 0.0], [0.8, 0.3, 0.2]]])
    potential = DeepPot(str(path), auto_batch_size=False, head="second")
    assert potential.get_model().get_min_nbor_dist() == 0.7
    expected = references["second"](torch.tensor(coord), torch.tensor([[0, 1]]))
    actual = potential.eval(coord, None, [0, 1])
    np.testing.assert_allclose(
        actual[0], expected["energy"].detach().numpy(), rtol=2e-8, atol=2e-9
    )
    frozen = tmp_path / "second.hlo"
    for head in (None, "missing"):
        with pytest.raises(ValueError, match="valid --task"):
            freeze(checkpoint_folder=str(path), output=str(frozen), head=head)
    freeze(checkpoint_folder=str(path), output=str(frozen), head="second")
    frozen_potential = DeepPot(str(frozen), auto_batch_size=False)
    assert frozen_potential.get_model().get_min_nbor_dist() == 0.7
    frozen_output = frozen_potential.eval(coord, None, [0, 1])
    np.testing.assert_allclose(
        frozen_output[0], expected["energy"].detach().numpy(), rtol=2e-8, atol=2e-9
    )


@pytest.mark.parametrize("graph_native", [False, True])
def test_live_inference_keeps_state_after_allocation_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graph_native: bool,
) -> None:
    """A failed dispatched prediction cannot replace the live model arrays."""
    config = (
        model_config()
        if graph_native
        else {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [2, 2],
                "rcut": 3.0,
                "rcut_smth": 2.5,
                "neuron": [2, 4],
                "axis_neuron": 2,
                "seed": 1,
            },
            "fitting_net": {"neuron": [2], "seed": 2},
        }
    )
    source = get_model(config)
    checkpoint = tmp_path / "model.jax"
    deserialize_to_file(
        str(checkpoint),
        {"model": source.serialize(), "model_def_script": config},
    )
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [0.8, 0.3, 0.2]],
            [[0.0, 0.1, 0.0], [0.9, 0.3, 0.2]],
            [[0.0, 0.0, 0.1], [0.8, 0.4, 0.2]],
        ]
    )
    expected = DeepPot(str(checkpoint), auto_batch_size=False).eval(
        coordinates,
        None,
        [0, 1],
        atomic=True,
    )
    monkeypatch.delenv("DP_INFER_BATCH_SIZE", raising=False)
    potential = DeepPot(str(checkpoint), auto_batch_size=6)
    model = potential.get_model()
    before = {
        path: as_pure(variable) for path, variable in to_flat_state(nnx.state(model))
    }
    values = {path: np.asarray(value).copy() for path, value in before.items()}
    name = "_call_graph" if graph_native else "_call_lower"
    compiled = getattr(potential.deep_eval, name)
    attempted_frames = []

    def allocation_failure_after_dispatch(*args: Any, **kwargs: Any) -> Any:
        inputs = args[-1]
        attempted_frames.append(
            inputs["n_node" if graph_native else "extended_atype"].shape[0]
        )
        result = compiled(*args, **kwargs)
        if len(attempted_frames) == 1:
            jax.block_until_ready(result)
            raise RuntimeError(
                "RESOURCE_EXHAUSTED: simulated inference allocation failure"
            )
        return result

    monkeypatch.setattr(potential.deep_eval, name, allocation_failure_after_dispatch)
    actual = potential.eval(coordinates, None, [0, 1], atomic=True)
    assert attempted_frames[1] < attempted_frames[0]
    for reference, output in zip(expected, actual, strict=True):
        np.testing.assert_allclose(output, reference, rtol=1e-12, atol=1e-12)
    after = dict(to_flat_state(nnx.state(model)))
    for path, value in before.items():
        assert value is as_pure(after[path]), path
        np.testing.assert_array_equal(as_pure(after[path]), values[path])


@pytest.mark.parametrize("suffix", [".jax", ".hlo"])
def test_case_film_vacuum_checkpoint_and_hlo(tmp_path: Path, suffix: str) -> None:
    """Case FiLM and parameter-conditioned vacuum references survive export."""
    config = model_config()
    config["fitting_net"].update(
        {
            "dim_case_embd": 2,
            "case_film_embd": True,
            "numb_fparam": 1,
            "numb_aparam": 1,
            "default_fparam": [0.25],
            "vacuum_ref": True,
        }
    )
    reference = _reference(config)
    reference.set_case_embd(1)
    reference.cpu()
    path = tmp_path / f"case{suffix}"
    deserialize_to_file(
        str(path),
        {"model": reference.serialize(), "model_def_script": config},
    )
    coord = np.array([[[0.0, 0.0, 0.0], [0.8, 0.3, 0.2]]])
    aparam = np.array([[[0.3], [-0.5]]])
    expected = reference(
        torch.tensor(coord),
        torch.tensor([[0, 1]]),
        aparam=torch.tensor(aparam),
    )
    potential = DeepPot(str(path), auto_batch_size=False)
    actual = potential.eval(coord, None, [0, 1], aparam=aparam)
    for value, key in zip(actual, ("energy", "force", "virial"), strict=True):
        np.testing.assert_allclose(
            value, expected[key].detach().numpy(), rtol=2e-8, atol=2e-9
        )


def test_savedmodel_guard_preserves_standard_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph models fail before TensorFlow import; ordinary models keep their exporter."""
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "deepmd.jax.jax2tf.serialization",
        SimpleNamespace(
            deserialize_to_file=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
    )
    graph_config = model_config(spin=True)
    with pytest.raises(NotImplementedError, match="cannot preserve full-neighbor"):
        deserialize_to_file(
            str(tmp_path / "graph.savedmodel"),
            {
                "model": get_model(graph_config).serialize(),
                "model_def_script": graph_config,
            },
        )
    assert not calls
    standard_config = {
        "type_map": ["H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [2],
            "rcut": 4.0,
            "rcut_smth": 3.0,
            "neuron": [2],
            "axis_neuron": 1,
        },
        "fitting_net": {"neuron": [2]},
    }
    data = {
        "model": get_model(standard_config).serialize(),
        "model_def_script": standard_config,
    }
    path = str(tmp_path / "standard.savedmodel")
    deserialize_to_file(path, data)
    assert calls == [((path, data), {"hessian": False})]
