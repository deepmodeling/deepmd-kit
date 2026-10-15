# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-native isolated references must not allocate a dense sel-sized list."""

import dataclasses
from unittest import (
    mock,
)

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.dpmodel.descriptor.dpa4c import (
    DescrptDPA4C,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    graph_from_dense_quartet,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)

TYPE_MAP = ["O", "H"]
BIAS = [[-3.0], [0.5]]


def make_model(
    kind: str,
    charge: bool = False,
    spin: bool = False,
    precision: str = "float64",
) -> DPAtomicModel:
    """Activate learned condition heads so missing inputs cannot pass by chance."""
    if kind == "dpa1":
        descriptor = DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[2],
            ntypes=2,
            attn_layer=0,
            axis_neuron=2,
            neuron=[4, 8],
            precision=precision,
            seed=3,
        )
    else:
        cls = DescrptDPA4 if kind == "dpa4" else DescrptDPA4C
        kwargs = {
            "ntypes": 2,
            "rcut": 4.0,
            "channels": 8,
            "n_radial": 4,
            "lmax": 2,
            "precision": precision,
            "seed": 3,
            "add_chg_spin_ebd": charge,
            "default_chg_spin": [0, 1] if charge else None,
            "use_spin": [True, False] if spin else None,
        }
        if kind == "dpa4":
            kwargs.update(sel=8, mmax=1, n_blocks=1)
        descriptor = cls(**kwargs)
        descriptor = cls.deserialize(
            jitter_zero_arrays(descriptor.serialize(), np.random.default_rng(7))
        )
        if kind == "dpa4c" and spin:
            descriptor.spin.spin_gate[...] = 1.0
    fitting = InvarFitting(
        "energy",
        2,
        descriptor.get_dim_out(),
        1,
        neuron=[8],
        mixed_types=descriptor.mixed_types(),
        precision=precision,
        vacuum_ref=True,
        seed=1,
    )
    return DPAtomicModel(
        descriptor,
        fitting,
        type_map=TYPE_MAP,
        preset_out_bias={"energy": BIAS},
    )


def bounded_reference(model: DPAtomicModel):
    """Use two explicitly bounded dense slots, never the descriptor's sel.

    The native-spin interface is graph-only, so adapt the dense quartet and
    pass spin on its flat node axis. Spin-less models also exercise the
    descriptor's dense compatibility adapter directly.
    """
    dtype = model.fitting_net.bias_atom_e.dtype
    coord = np.zeros((2, 1, 3), dtype=dtype)
    atype = np.arange(2, dtype=np.int64).reshape(2, 1)
    nlist = np.full((2, 1, 2), -1, dtype=np.int64)
    mapping = np.zeros((2, 1), dtype=np.int64)
    graph, flat_atype = graph_from_dense_quartet(coord, atype, nlist, mapping)
    conditions = {
        key: np.asarray(value, dtype=dtype)
        for key, value in model.vacuum_conditions().items()
    }
    descriptor = model.descriptor
    expected = descriptor.call_graph(
        graph,
        flat_atype,
        type_embedding=descriptor.graph_type_embedding_table(),
        **conditions,
    )[0]
    if not model.add_spin_ebd:
        dense = descriptor(coord, atype, nlist, mapping=mapping, **conditions)[0]
        np.testing.assert_allclose(dense.reshape(expected.shape), expected, atol=2e-6)
    return expected, graph, flat_atype, conditions


@pytest.mark.parametrize("kind", ["dpa4", "dpa4c"])
@pytest.mark.parametrize(
    "charge,spin", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_vacuum_uses_bounded_graph(kind, charge, spin, precision) -> None:
    model = make_model(kind, charge, spin, precision)
    expected, _, _, conditions = bounded_reference(model)
    descriptor = model.descriptor
    assert descriptor.supports_graph_vacuum()
    if kind == "dpa4c":
        assert descriptor.get_sel() == [999999]
    call_graph = descriptor.call_graph

    def checked_graph(graph, atype, **kwargs):
        np.testing.assert_array_equal(graph.n_node, [1, 1])
        np.testing.assert_array_equal(atype, [0, 1])
        assert graph.edge_index.shape == (2, 2)
        assert graph.edge_vec.shape == (2, 3)
        assert graph.edge_mask.shape == (2,)
        assert not graph.edge_mask.any()
        assert graph.edge_vec.dtype == model.fitting_net.bias_atom_e.dtype
        assert set(kwargs) == {"type_embedding", *conditions}
        for name, value in conditions.items():
            np.testing.assert_array_equal(kwargs[name], value)
        return call_graph(graph, atype, **kwargs)

    # Fail before allocation if the implementation attempts the dense route.
    # Repeat with a larger sentinel to prove that the graph is sel-independent.
    with (
        mock.patch.object(
            descriptor, "get_nnei", side_effect=AssertionError("dense capacity queried")
        ),
        mock.patch.object(
            descriptor, "call", side_effect=AssertionError("dense adapter called")
        ),
        mock.patch.object(
            descriptor, "call_graph", side_effect=checked_graph
        ) as graph_call,
    ):
        for sentinel in (999999, 1999999):
            with mock.patch.object(descriptor, "get_sel", return_value=[sentinel]):
                actual = model.vacuum_descriptor()
            assert actual.shape == (2, descriptor.get_dim_out())
            assert actual.dtype == model.fitting_net.bias_atom_e.dtype
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        assert graph_call.call_count == 2
    assert np.max(np.abs(actual)) > 1e-5
    assert not np.allclose(actual[0], actual[1])


@pytest.mark.parametrize("kind", ["dpa4", "dpa4c"])
def test_empty_and_masked_edges_preserve_conditions(kind) -> None:
    model = make_model(kind, charge=True, spin=True)
    expected, dense_graph, atype, conditions = bounded_reference(model)
    np.testing.assert_allclose(model.vacuum_descriptor(), expected, atol=1e-12)
    for nedges in (0, 1, 7):
        graph = dataclasses.replace(
            dense_graph,
            edge_index=np.tile(np.array([[1], [0]], dtype=np.int64), (1, nedges)),
            edge_vec=np.ones((nedges, 3)),
            edge_mask=np.zeros(nedges, dtype=bool),
            destination_order=None,
            destination_row_ptr=None,
            source_order=None,
            source_row_ptr=None,
        )
        actual = model.descriptor.call_graph(graph, atype, **conditions)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    # The reference is the learned, ground-state-conditioned value, not zero
    # or a default-condition shortcut, even when no real edges exist.
    for name in conditions:
        other = {**conditions, name: np.zeros_like(conditions[name])}
        if name == "charge_spin":
            other[name][:, 1] = 1.0
        actual = model.descriptor.call_graph(dense_graph, atype, **other)[0]
        assert np.max(np.abs(actual - expected)) > 1e-7, name


def test_graph_descriptor_receives_its_type_embedding() -> None:
    """An opted-in descriptor receives its external type-embedding table."""
    model = make_model("dpa1")
    assert not model.descriptor.supports_graph_vacuum()
    expected, _, _, _ = bounded_reference(model)
    with (
        mock.patch.object(model.descriptor, "supports_graph_vacuum", return_value=True),
        mock.patch.object(
            model.descriptor, "call", side_effect=AssertionError("dense adapter called")
        ),
    ):
        actual = model.vacuum_descriptor()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_dense_forward_keeps_its_own_reference_semantics() -> None:
    """DPA1's legacy padding residual must cancel on its dense path too."""
    model = make_model("dpa1")
    model.descriptor.se_atten.mean[...] = 0.3
    coord = np.zeros((2, 1, 3))
    atype = np.arange(2, dtype=np.int64).reshape(2, 1)
    nlist = np.full((2, 1, 2), -1, dtype=np.int64)
    mapping = np.zeros((2, 1), dtype=np.int64)
    assert model.descriptor.uses_graph_lower()
    assert not model.descriptor.supports_graph_vacuum()
    graph, flat_atype = graph_from_dense_quartet(coord, atype, nlist, mapping)
    graph_reference = model.descriptor.call_graph(
        graph,
        flat_atype,
        type_embedding=model.descriptor.graph_type_embedding_table(),
    )[0]
    dense_reference = model.vacuum_descriptor()
    assert np.max(np.abs(graph_reference - dense_reference)) > 1e-7
    before = model.forward_atomic(coord, atype, nlist, mapping=mapping)["energy"]
    np.testing.assert_allclose(
        before[:, 0], model.fitting_net.bias_atom_e, rtol=1e-12, atol=1e-12
    )
    # Graph support alone must not silently change the legacy reference.
    model.fold_vacuum_reference()
    after = model.forward_atomic(coord, atype, nlist, mapping=mapping)["energy"]
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)


def test_disabled_graph_lower_keeps_dense_reference() -> None:
    """An explicit lower opt-out still selects the dense compatibility path."""
    model = make_model("dpa4", charge=True)
    expected = model.vacuum_descriptor()
    model.descriptor.disable_graph_lower()
    with mock.patch.object(
        model.descriptor, "call", wraps=model.descriptor.call
    ) as call:
        actual = model.vacuum_descriptor()
    call.assert_called_once()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kind,charge,spin",
    [
        ("dpa4", False, False),
        ("dpa4", True, True),
        ("dpa4c", False, False),
        ("dpa4c", True, True),
    ],
)
def test_folding_matches_appended_reference_frames(kind, charge, spin) -> None:
    """Folding must match reference nodes carried alongside a real cluster."""
    model = make_model(kind, charge, spin)
    coord = np.array([[[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]]])
    atype = np.array([[0, 1]], dtype=np.int64)
    nlist = np.array([[[1, -1], [0, -1]]], dtype=np.int64)
    graph, atype = graph_from_dense_quartet(coord, atype, nlist, None)
    conditions = {}
    if charge:
        conditions["charge_spin"] = np.array([[1.0, 2.0]])
    if spin:
        conditions["spin"] = np.array([[0.2, 0.4, 1.0], [0.0, 0.0, 0.0]])
    before = model.forward_atomic_graph(graph, atype, **conditions)["energy"]
    model.fold_vacuum_reference()
    assert not model.fitting_net.needs_vacuum_descriptor()
    after = model.forward_atomic_graph(graph, atype, **conditions)["energy"]
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)
