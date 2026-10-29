# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX DPA4 model and native-spin parity against the PyTorch implementation."""

import copy
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import pytest

from deepmd.jax.common import (
    to_flat_state,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.model import (
    get_model,
)


def model_config(spin: bool = False, bridging: bool = False) -> dict[str, Any]:
    config = {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "use_compile": False,
        "descriptor": {
            "type": "dpa4",
            "sel": 8,
            "rcut": 4.0,
            "channels": 4,
            "n_radial": 3,
            "lmax": 1,
            "mmax": 1,
            "n_blocks": 1,
            "grid_branch": 0,
            "random_gamma": False,
            "use_amp": False,
            "precision": "float64",
            "seed": 20261023,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [4],
            "precision": "float64",
            "seed": 20261023,
        },
    }
    if spin:
        config["spin"] = {"scheme": "native", "use_spin": [True, False]}
    if bridging:
        config["bridging_method"] = "zbl"
    return config


@pytest.mark.parametrize("kind", ["linear", "switched"])
def test_dense_pairtab_rejects_full_neighbor_descriptor(
    tmp_path: Path, kind: str
) -> None:
    """A dense-only child cannot silently truncate a full-neighbor descriptor."""
    table = tmp_path / "pair.txt"
    np.savetxt(
        table,
        np.array(
            [
                [0.1, 1.0, 2.0, 3.0],
                [0.2, 0.8, 1.6, 2.4],
                [0.3, 0.5, 1.0, 1.5],
                [0.4, 0.0, 0.0, 0.0],
            ]
        ),
    )
    learned = model_config()
    if kind == "linear":
        config = {
            "type": "linear_ener",
            "type_map": learned["type_map"],
            "models": [
                learned,
                {"type": "pairtab", "tab_file": str(table), "rcut": 0.4, "sel": 8},
            ],
            "weights": "sum",
        }
    else:
        learned["type"] = "standard"
        learned["fitting_net"]["type"] = "ener"
        config = dict(learned, use_srtab=str(table), sw_rmin=0.15, sw_rmax=0.3)
    with pytest.raises(NotImplementedError, match=r"full neighbor.*graph"):
        get_model(config)


def test_dense_hybrid_rejects_full_neighbor_descriptor() -> None:
    """Hybrid capacity slicing cannot represent an unbounded child descriptor."""
    config = model_config()
    config["type"] = "standard"
    config["descriptor"] = {"type": "hybrid", "list": [config["descriptor"]]}
    config["fitting_net"]["type"] = "ener"
    with pytest.raises(NotImplementedError, match=r"full neighbor.*graph"):
        get_model(config)


def test_full_neighbor_requirement_is_independent_of_spin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neighbor selection depends on graph semantics rather than spin support."""
    from deepmd.jax.atomic_model.dp_atomic_model import (
        DPAtomicModel,
    )

    monkeypatch.setattr(DPAtomicModel, "supports_native_spin", lambda self: False)
    model = get_model(model_config())
    assert model.uses_full_neighbor_list()
    assert BaseModel.deserialize(model.serialize()).uses_full_neighbor_list()


def _reference(config: dict[str, Any]) -> Any:
    import torch

    from deepmd.pt.model.model import get_model as get_pt_model

    model = get_pt_model(copy.deepcopy(config)).cpu().eval()
    generator = torch.Generator().manual_seed(20261023)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                0.05
                * torch.randn(
                    parameter.shape, dtype=parameter.dtype, generator=generator
                )
            )
    return model


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("bridging", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_pt_energy_force_and_magnetic_force(
    spin: bool,
    bridging: bool,
    periodic: bool,
) -> None:
    """Serialized weights preserve energies and both conservative derivatives."""
    import torch

    config = model_config(spin, bridging)
    reference = _reference(config)
    model = BaseModel.deserialize(reference.serialize())
    coord = np.array([[[0.0, 0.0, 0.0], [0.8, 0.3, 0.2], [0.4, 1.2, 0.7]]])
    atype = np.array([[0, 1, 0]], dtype=np.int64)
    spins = np.array([[[0.4, -0.2, 0.3], [0.2, 0.1, 0.3], [0.1, 0.5, 0.2]]])
    box = (np.eye(3) * 3.0).reshape(1, 9) if periodic else None
    kwargs = {"spin": torch.tensor(spins)} if spin else {}
    expected = reference(
        torch.tensor(coord),
        torch.tensor(atype),
        box=torch.tensor(box) if box is not None else None,
        do_atomic_virial=True,
        **kwargs,
    )
    kwargs = {"spin": jnp.asarray(spins)} if spin else {}
    actual = model(
        jnp.asarray(coord),
        jnp.asarray(atype),
        box=jnp.asarray(box) if box is not None else None,
        do_atomic_virial=True,
        **kwargs,
    )
    keys = ["energy", "atom_energy", "force", "virial", "atom_virial"]
    if spin:
        keys += ["force_mag", "mask_mag"]
    for key in keys:
        np.testing.assert_allclose(
            np.asarray(actual[key]),
            expected[key].detach().numpy(),
            rtol=2e-9,
            atol=2e-10,
            err_msg=key,
        )
    assert np.linalg.norm(np.asarray(actual["force"])) > 1e-8
    restored = BaseModel.deserialize(model.serialize())
    repeated = restored(
        jnp.asarray(coord),
        jnp.asarray(atype),
        box=jnp.asarray(box) if box is not None else None,
        do_atomic_virial=True,
        **kwargs,
    )
    for key in keys:
        np.testing.assert_allclose(actual[key], repeated[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("bridging", [False, True])
def test_factory_builds_trainable_model(spin: bool, bridging: bool) -> None:
    model = get_model(model_config(spin, bridging))
    assert model.has_spin() is spin
    assert to_flat_state(nnx.state(model, nnx.Param))


@pytest.mark.parametrize("spin", [False, True])
def test_lower_jit_and_force_loss_gradient(spin: bool) -> None:
    """The force-training path has finite mixed coordinate/parameter derivatives."""
    model = BaseModel.deserialize(_reference(model_config(spin)).serialize())
    coord = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.3, 0.2]]])
    atype = jnp.array([[0, 1]])
    nlist = jnp.array([[[1, -1], [0, -1]]])
    mapping = jnp.array([[0, 1]])
    spins = jnp.array([[[0.2, 0.3, -0.1], [0.0, 0.0, 0.0]]]) if spin else None

    def loss(model: Any, coord: jnp.ndarray) -> jnp.ndarray:
        outputs = model.call_common_lower(coord, atype, nlist, mapping, spin=spins)
        value = jnp.sum(outputs["energy_derv_r"] ** 2)
        if spin:
            value += jnp.sum(outputs["energy_derv_r_mag"] ** 2)
        return value

    value, gradients = nnx.jit(nnx.value_and_grad(loss))(model, coord)
    assert np.isfinite(value)
    leaves = jax.tree.leaves(gradients)
    assert leaves and all(np.isfinite(leaf).all() for leaf in leaves)
    assert sum(float(jnp.linalg.norm(leaf)) for leaf in leaves) > 1e-8


@pytest.mark.parametrize("spin", [False, True])
def test_graph_lower_jit_matches_dense(spin: bool) -> None:
    """Fixed graph capacities preserve forces, virials, and magnetic gradients."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        graph_from_dense_quartet,
    )

    model = BaseModel.deserialize(_reference(model_config(spin)).serialize())
    coord = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.3, 0.2]]])
    atype = jnp.array([[0, 1]])
    nlist = jnp.array([[[1, -1], [0, -1]]])
    mapping = jnp.array([[0, 1]])
    spins = jnp.array([[[0.2, 0.3, -0.1], [0.0, 0.0, 0.0]]]) if spin else None
    graph, local_types = graph_from_dense_quartet(coord, atype, nlist, mapping)

    def evaluate(model: Any, vectors: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return model.call_common_lower_graph(
            local_types,
            graph.n_node,
            graph.edge_index,
            vectors,
            graph.edge_mask,
            spin=spins.reshape(-1, 3) if spin else None,
        )

    actual = nnx.jit(evaluate)(model, graph.edge_vec)
    expected = model.call_common_lower(coord, atype, nlist, mapping, spin=spins)
    for key in ("energy", "energy_redu", "energy_derv_r", "energy_derv_c_redu"):
        np.testing.assert_allclose(
            np.asarray(actual[key]).reshape(expected[key].shape),
            expected[key],
            rtol=1e-9,
            atol=1e-10,
            err_msg=key,
        )
    if spin:
        np.testing.assert_allclose(
            actual["energy_derv_r_mag"].reshape(expected["energy_derv_r_mag"].shape),
            expected["energy_derv_r_mag"],
            rtol=1e-9,
            atol=1e-10,
        )


def test_explicit_hessian_output_is_rejected() -> None:
    """The output contract fails explicitly instead of returning a NaN placeholder."""
    model = get_model(model_config())
    model.enable_hessian()
    with pytest.raises(NotImplementedError, match="Explicit Hessian outputs"):
        model.call_common_lower(
            jnp.zeros((1, 1, 3)),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.full((1, 1, 1), -1, dtype=jnp.int64),
            jnp.zeros((1, 1), dtype=jnp.int64),
        )


def test_coordinate_correction_preserves_total_virial() -> None:
    """A common coordinate shift changes atomic attribution but not total virial."""
    model = BaseModel.deserialize(_reference(model_config()).serialize())
    coord = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.3, 0.2]]])
    atype = jnp.array([[0, 1]])
    nlist = jnp.array([[[1], [0]]])
    mapping = jnp.array([[0, 1]])
    correction = jnp.broadcast_to(jnp.array([0.2, -0.3, 0.1]), coord.shape)
    expected = model.call_common_lower(coord, atype, nlist, mapping)
    actual = model.call_common_lower(
        coord,
        atype,
        nlist,
        mapping,
        extended_coord_corr=correction,
    )
    np.testing.assert_allclose(
        actual["energy_derv_c_redu"],
        expected["energy_derv_c_redu"],
        rtol=1e-12,
        atol=1e-12,
    )
    delta = actual["energy_derv_c"] - expected["energy_derv_c"]
    force = expected["energy_derv_r"]
    np.testing.assert_allclose(
        delta.reshape(1, 2, 1, 3, 3),
        force[..., :, None] * correction[:, :, None, None, :],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.linalg.norm(delta) > 1e-8
