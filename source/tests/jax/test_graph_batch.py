# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sparse batch capacities preserve topology, loss values and derivatives."""

from typing import (
    Any,
)

import numpy as np
import pytest

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.ener_spin import (
    EnergySpinLoss,
)
from deepmd.dpmodel.loss.reduction import (
    absolute_error,
    safe_vector_norm,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    build_neighbor_graph,
)
from deepmd.dpmodel.utils.neighbor_graph.vesin_builder import (
    build_neighbor_graph_vesin,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.utils.graph_batch import (
    bucket_capacity,
    prepare_graph_batch,
    prepare_graph_batches,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 17, 63, 64, 65, 137, 10000])
@pytest.mark.parametrize("subdivisions", [1, 2, 8, 16])
def test_capacity_never_truncates_and_limits_padding(
    size: int, subdivisions: int
) -> None:
    capacity = bucket_capacity(size, subdivisions=subdivisions)
    assert capacity >= max(size, 1)
    assert capacity <= max(size, 1) * (1.0 + 1.0 / subdivisions)
    if subdivisions == 1:
        assert capacity & (capacity - 1) == 0
    assert bucket_capacity(size, minimum=16, subdivisions=subdivisions) >= 16


@pytest.mark.parametrize("subdivisions", [0, -1, 3, 6, 8.0])
def test_capacity_rejects_invalid_subdivisions(subdivisions: Any) -> None:
    with pytest.raises(ValueError, match="positive power-of-two integer"):
        bucket_capacity(17, subdivisions=subdivisions)


@pytest.mark.parametrize("periodic", [False, True])
def test_sparse_search_matches_uncapped_dense_search(periodic: bool) -> None:
    """Small periodic cells retain repeated images and each directed edge."""
    coord = np.array([[[0.2, 0.1, 0.3], [1.3, 0.6, 0.2], [0.6, 1.7, 0.8]]])
    atype = np.array([[0, 1, -1]])
    box = (
        np.array([[[2.1, 0, 0], [0.2, 2.2, 0], [0.3, -0.1, 2.3]]]) if periodic else None
    )
    sparse = build_neighbor_graph_vesin(
        coord, atype, box, 3.2, GraphLayout(min_edges=0)
    )
    dense = build_neighbor_graph(coord, atype, box, 3.2, GraphLayout(min_edges=0))

    def edges(graph: Any) -> np.ndarray:
        values = np.column_stack((graph.edge_index.T, graph.edge_vec))
        return np.array(sorted(map(tuple, np.round(values, 12))))

    np.testing.assert_allclose(edges(sparse), edges(dense), atol=1e-12)


def test_rectangular_and_ragged_batches_have_identical_compact_graphs() -> None:
    rng = np.random.default_rng(20261026)
    coord = rng.normal(size=(2, 11, 3))
    atype = np.ones((2, 11), dtype=np.int64)
    atype[0, 6:] = -1
    force = rng.normal(size=coord.shape)
    rectangular = {"coord": coord, "type": atype, "force": force, "find_force": 1.0}
    physical = atype.reshape(-1) >= 0
    ragged = {
        "coord": coord.reshape(-1, 3)[physical],
        "type": atype.reshape(-1)[physical],
        "force": force.reshape(-1, 3)[physical],
        "find_force": 1.0,
        "n_node": np.array([6, 11]),
    }
    labels, inputs = prepare_graph_batch(rectangular, rcut=4.0, ntypes=2)
    ragged_labels, ragged_inputs = prepare_graph_batch(ragged, rcut=4.0, ntypes=2)
    for first, second in ((labels, ragged_labels), (inputs, ragged_inputs)):
        assert first.keys() == second.keys()
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
    assert inputs["atype"].size == 18
    assert np.sum(inputs["atype"] >= 0) == 17
    assert inputs["edge_mask"].sum() > 8 * 17


def test_device_batches_merge_compact_graphs_and_label_availability() -> None:
    """Mixed periodicity, phantom nodes and missing labels preserve each group."""
    rng = np.random.default_rng(876)
    first_types = rng.integers(0, 2, size=(2, 11))
    first_types[0, 6:] = -1
    second_types = np.array([[0, 1, 0, 1, -1]])
    first = {
        "type": first_types.copy(),
        "coord": rng.uniform(0, 2.0, size=(2, 11, 3)),
        "box": np.tile(np.eye(3) * 2.5, (2, 1, 1)),
        "find_box": True,
        "energy": np.array([[1.0], [2.0]]),
        "find_energy": True,
        "force": rng.normal(size=(2, 11, 3)),
        "find_force": True,
        "atom_pref": np.ones((2, 11, 3)) * 2.0,
        "find_atom_pref": True,
        "find_property": False,
        "fparam": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    second = {
        "type": second_types.copy(),
        "coord": rng.uniform(0, 2.0, size=(1, 5, 3)),
        "find_box": False,
        "energy": np.array([[3.0]]),
        "find_energy": True,
        "find_force": False,
        "find_atom_pref": False,
        "property": np.array([[6.0, 7.0]]),
        "find_property": True,
        "fparam": np.array([[5.0, 6.0]]),
    }
    requirements = [
        DataRequirementItem("force", 3, atomic=True, default=0.125),
        DataRequirementItem("atom_pref", 3, atomic=True, default=1.0),
        DataRequirementItem("property", 2, default=-2.0),
    ]
    options = {
        "rcut": 1.8,
        "ntypes": 2,
        "data_requirements": requirements,
        "pair_exclude_types": [(0, 1)],
    }
    labels, inputs = prepare_graph_batches((first, second), **options)
    individual = [prepare_graph_batch(batch, **options) for batch in (first, second)]
    np.testing.assert_array_equal(inputs["n_node"][:3], [6, 11, 4])
    np.testing.assert_array_equal(inputs["n_node"][3:], 0)
    np.testing.assert_array_equal(labels["data_group"][:3], [0, 0, 1])
    np.testing.assert_array_equal(labels["data_group"][3:], -1)
    np.testing.assert_array_equal(labels["frame_mask"][:3], True)
    np.testing.assert_array_equal(labels["frame_mask"][3:], False)
    np.testing.assert_array_equal(labels["group_find_force"], [True, False])
    np.testing.assert_array_equal(labels["group_find_property"], [False, True])
    np.testing.assert_array_equal(labels["group_find_box"], [True, False])
    assert labels["find_force"] and labels["find_property"]
    assert inputs["atype"].shape == (22,)
    assert np.all(inputs["atype"][:21] >= 0) and inputs["atype"][21] == -1
    np.testing.assert_array_equal(labels["force"][17:21], 0.125)
    np.testing.assert_array_equal(labels["atom_pref"][17:21], 1.0)
    np.testing.assert_array_equal(labels["property"][:2], -2.0)
    np.testing.assert_array_equal(labels["property"][2], [6.0, 7.0])
    np.testing.assert_array_equal(inputs["fparam"][:3], [[1, 2], [3, 4], [5, 6]])
    expected_indices, expected_vectors = [], []
    for (_, graph), offset in zip(individual, (0, 17), strict=True):
        mask = graph["edge_mask"]
        expected_indices.append(graph["edge_index"][:, mask] + offset)
        expected_vectors.append(graph["edge_vec"][mask])
    physical_edges = inputs["edge_mask"]
    np.testing.assert_array_equal(
        inputs["edge_index"][:, physical_edges],
        np.concatenate(expected_indices, axis=1),
    )
    np.testing.assert_array_equal(
        inputs["edge_vec"][physical_edges], np.concatenate(expected_vectors)
    )
    node_groups = np.repeat([0, 0, 1], [6, 11, 4])
    edges = inputs["edge_index"][:, physical_edges]
    np.testing.assert_array_equal(node_groups[edges[0]], node_groups[edges[1]])
    assert "data_group" not in individual[0][0]
    np.testing.assert_array_equal(first["type"], first_types)
    np.testing.assert_array_equal(second["type"], second_types)


def test_empty_edge_groups_share_only_final_guards() -> None:
    batches = [
        {
            "type": np.array([[0]]),
            "coord": np.array([[[float(group), 0.0, 0.0]]]),
            "find_box": False,
        }
        for group in range(3)
    ]
    labels, inputs = prepare_graph_batches(batches, rcut=0.5, ntypes=1)
    np.testing.assert_array_equal(inputs["n_node"], [1, 1, 1, 0])
    np.testing.assert_array_equal(labels["data_group"], [0, 1, 2, -1])
    assert inputs["edge_vec"].shape == (2, 3)
    assert not inputs["edge_mask"].any()
    np.testing.assert_array_equal(inputs["edge_index"], 0)


def test_graph_group_sequence_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="at least one input batch"):
        prepare_graph_batches([], rcut=1.0, ntypes=1)


@pytest.mark.parametrize("subdivisions", [1, 8])
def test_frame_capacity_follows_node_capacity(subdivisions: int) -> None:
    """Regrouping the same node budget does not introduce a compilation shape."""
    raw = {
        "type": np.zeros(5, dtype=np.int64),
        "coord": np.arange(15, dtype=np.float64).reshape(5, 3) * 10,
    }
    inputs = [
        prepare_graph_batch(
            dict(raw, n_node=np.asarray(counts)),
            rcut=1.0,
            ntypes=1,
            bucket_subdivisions=subdivisions,
        )[1]
        for counts in ([2, 3], [1, 1, 1, 1, 1])
    ]
    assert {key: value.shape for key, value in inputs[0].items()} == {
        key: value.shape for key, value in inputs[1].items()
    }
    assert inputs[0]["atype"].size == bucket_capacity(5, subdivisions=subdivisions)


def test_fine_and_coarse_buckets_preserve_physical_fields() -> None:
    rng = np.random.default_rng(36)
    raw = {
        "type": np.zeros(17, dtype=np.int64),
        "coord": rng.normal(size=(17, 3)),
        "n_node": np.array([6, 11]),
        "force": rng.normal(size=(17, 3)),
        "find_force": True,
    }
    fine_labels, fine = prepare_graph_batch(raw, rcut=4.0, ntypes=1)
    coarse_labels, coarse = prepare_graph_batch(
        raw, rcut=4.0, ntypes=1, bucket_subdivisions=1
    )
    assert fine["atype"].size == 18 and coarse["atype"].size == 32
    assert np.sum(fine["atype"] >= 0) == np.sum(coarse["atype"] >= 0) == 17
    assert np.sum(fine_labels["frame_mask"]) == np.sum(coarse_labels["frame_mask"]) == 2
    np.testing.assert_array_equal(
        fine_labels["force"][:17], coarse_labels["force"][:17]
    )
    np.testing.assert_array_equal(coarse_labels["force"][17:], 0.0)
    np.testing.assert_array_equal(
        fine["edge_index"][:, fine["edge_mask"]],
        coarse["edge_index"][:, coarse["edge_mask"]],
    )
    np.testing.assert_array_equal(
        fine["edge_vec"][fine["edge_mask"]], coarse["edge_vec"][coarse["edge_mask"]]
    )
    np.testing.assert_array_equal(coarse["edge_vec"][~coarse["edge_mask"]], 0.0)


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("loss_func", ["mse", "mae"])
def test_frame_and_node_padding_preserve_loss_and_parameter_gradient(
    spin: bool, loss_func: str
) -> None:
    """Padded frames do not dilute per-frame energy/virial averages."""
    kwargs = {
        "starter_learning_rate": 1.0,
        "loss_func": loss_func,
        "start_pref_e": 1.0,
        "limit_pref_e": 1.0,
        "start_pref_v": 1.0,
        "limit_pref_v": 1.0,
    }
    if spin:
        loss = EnergySpinLoss(
            **kwargs,
            start_pref_fr=1.0,
            limit_pref_fr=1.0,
            start_pref_fm=1.0,
            limit_pref_fm=1.0,
        )
    else:
        loss = EnergyLoss(
            **kwargs,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            f_use_norm=loss_func == "mae",
        )
    rng = np.random.default_rng(20261026)
    labels = {
        "energy": rng.normal(size=(2, 1)),
        "force": rng.normal(size=(5, 3)),
        "force_mag": rng.normal(size=(5, 3)),
        "virial": rng.normal(size=(2, 9)),
        "find_energy": 1.0,
        "find_force": 1.0,
        "find_force_mag": 1.0,
        "find_virial": 1.0,
    }
    predictions = {
        key: value * 0.6 for key, value in labels.items() if not key.startswith("find_")
    }
    predictions.update(
        mask=np.ones(5), mask_mag=np.ones((5, 1), dtype=bool), n_node=np.array([2, 3])
    )
    padded_labels = dict(labels, frame_mask=np.array([True, True, False]))
    padded_predictions = dict(predictions)
    for key in ("energy", "force", "force_mag", "virial", "mask", "mask_mag"):
        pad = ((0, 1),) + ((0, 0),) * (predictions[key].ndim - 1)
        padded_predictions[key] = np.pad(predictions[key], pad)
        if key in labels:
            padded_labels[key] = np.pad(labels[key], pad)
    padded_predictions["n_node"] = np.array([2, 3, 0])

    def evaluate(scale: jax.Array, padded: bool) -> jax.Array:
        pred = padded_predictions if padded else predictions
        target = padded_labels if padded else labels
        values = {
            key: jnp.asarray(value) * scale
            if key in {"energy", "force", "force_mag", "virial"}
            else jnp.asarray(value)
            for key, value in pred.items()
        }
        return loss(1.0, 5, values, jax.tree.map(jnp.asarray, target))[0]

    original = jax.jit(jax.value_and_grad(lambda scale: evaluate(scale, False)))(1.1)
    padded = jax.jit(jax.value_and_grad(lambda scale: evaluate(scale, True)))(1.1)
    np.testing.assert_allclose(padded, original, rtol=1e-12, atol=1e-12)


def test_zero_force_error_has_finite_subgradient() -> None:
    gradient = jax.jit(jax.grad(lambda value: jnp.sum(safe_vector_norm(value))))
    np.testing.assert_array_equal(gradient(jnp.zeros((4, 3))), np.zeros((4, 3)))


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("capacity_mode", ["frames", "buckets"])
def test_frame_capacity_preserves_full_model_force_loss_gradient(
    spin: bool, capacity_mode: str
) -> None:
    """Frame and node/edge capacities preserve physical outputs and mixed derivatives."""
    from deepmd.jax.env import (
        nnx,
    )
    from deepmd.jax.model.base_model import (
        BaseModel,
    )
    from deepmd.jax.train.trainer import (
        _evaluate_inputs,
    )

    from .test_dpa4_model import (
        _reference,
        model_config,
    )

    config = model_config(spin, bridging=spin)
    config["fitting_net"]["vacuum_ref"] = True
    model = BaseModel.deserialize(_reference(config).serialize())
    rng = np.random.default_rng(52)
    raw = {
        "type": np.array([0, 1, 0, 0, 1]),
        "coord": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.3, 0.2],
                [0.4, 1.2, 0.7],
                [0.1, 0.2, 0.3],
                [1.0, 0.8, 0.6],
            ]
        ),
        "n_node": np.array([3, 2]),
        "energy": rng.normal(size=(2, 1)),
        "force": rng.normal(size=(5, 3)),
        "virial": rng.normal(size=(2, 9)),
        "find_energy": True,
        "find_force": True,
        "find_virial": True,
    }
    if spin:
        raw["spin"] = rng.normal(size=(5, 3))
    labels, inputs = prepare_graph_batch(raw, rcut=4.0, ntypes=2)
    if capacity_mode == "buckets":
        expanded_labels, expanded_inputs = prepare_graph_batch(
            raw,
            rcut=4.0,
            ntypes=2,
            bucket_subdivisions=1,
        )
    else:
        expanded_labels, expanded_inputs = dict(labels), dict(inputs)
        for mapping in (expanded_labels, expanded_inputs):
            for key in ("energy", "virial", "frame_mask", "n_node"):
                if key in mapping:
                    value = mapping[key]
                    padding = [(0, 32 - value.shape[0])] + [(0, 0)] * (value.ndim - 1)
                    mapping[key] = np.pad(value, padding)
    loss = EnergyLoss(
        starter_learning_rate=1.0,
        loss_func="mae",
        f_use_norm=True,
        start_pref_e=1.0,
        limit_pref_e=1.0,
        start_pref_f=1.0,
        limit_pref_f=1.0,
        start_pref_v=1.0,
        limit_pref_v=1.0,
    )

    def objective(
        model: BaseModel, labels: dict, inputs: dict
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        prediction = _evaluate_inputs(model, inputs)
        physical = {
            "energy": prediction["energy"][:2],
            "force": prediction["force"][:5],
            "virial": prediction["virial"][:2],
        }
        if spin:
            physical["force_mag"] = prediction["force_mag"][:5]
        return loss(1.0, inputs["atype"].size, prediction, labels)[0], physical

    evaluate = nnx.jit(nnx.value_and_grad(objective, has_aux=True))
    expected = evaluate(
        model, jax.tree.map(jnp.asarray, labels), jax.tree.map(jnp.asarray, inputs)
    )
    assert (
        sum(float(jnp.linalg.norm(value)) for value in jax.tree.leaves(expected[1]))
        > 1e-8
    )
    actual = evaluate(
        model,
        jax.tree.map(jnp.asarray, expanded_labels),
        jax.tree.map(jnp.asarray, expanded_inputs),
    )
    for reference, value in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_allclose(value, reference, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("spin", [False, True])
def test_exact_energy_fit_has_zero_mae_gradient(spin: bool) -> None:
    """An exactly fitted label does not drive an optimizer update."""
    options = {
        "starter_learning_rate": 1.0,
        "loss_func": "mae",
        "start_pref_e": 1.0,
        "limit_pref_e": 1.0,
    }
    if spin:
        loss = EnergySpinLoss(
            **options,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
        )
    else:
        loss = EnergyLoss(**options, start_pref_f=0.0, limit_pref_f=0.0)
    target = jnp.array([[0.0], [1.0]])

    def objective(scale: jax.Array) -> jax.Array:
        return loss(
            1.0,
            1,
            {"energy": scale * target},
            {"energy": target, "find_energy": True},
        )[0]

    assert jax.jit(jax.grad(objective))(1.0) == 0.0


def test_residual_reductions_preserve_nonfinite_values() -> None:
    """Zero-subgradient conventions must not turn invalid residuals into zeros."""
    values = jnp.array([0.0, jnp.nan, jnp.inf, -jnp.inf])
    absolute = jax.jit(absolute_error)(values)
    norms = jax.jit(safe_vector_norm)(values[:, None])
    for result in (absolute, norms):
        assert result[0] == 0.0
        assert np.isnan(result[1])
        assert np.isposinf(result[2:]).all()


def test_energy_loss_does_not_retain_jit_tracers() -> None:
    """Loss instances keep configuration only, so compiled traces can be released."""
    loss = EnergyLoss(
        starter_learning_rate=1.0,
        start_pref_e=1.0,
        limit_pref_e=1.0,
        start_pref_f=0.0,
        limit_pref_f=0.0,
    )

    def objective(energy: jax.Array) -> jax.Array:
        return loss(
            1.0,
            1,
            {"energy": energy.reshape(1, 1)},
            {"energy": jnp.zeros((1, 1)), "find_energy": True},
        )[0]

    with jax.checking_leaks():
        assert jax.jit(objective)(jnp.array(2.0)) == 4.0
