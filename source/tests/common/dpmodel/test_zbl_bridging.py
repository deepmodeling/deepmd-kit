# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel ZBL bridging as COMPOSITION (review 3638077323, redesigned).

``bridging_method: ZBL`` builds a
``LinearEnergyModel(LinearEnergyAtomicModel([dp, InterPotentialAtomicModel],
weights="sum"))`` -- the analytical term is its own atomic model summed
with the learned one, not a flag on it.
"""

import copy

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model.inter_potential import (
    InterPotentialAtomicModel,
)
from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.dp_linear_model import (
    LinearEnergyModel,
)
from deepmd.dpmodel.model.model import (
    get_model,
)

ZBL_CONFIG = {
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        "sel": 8,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 7,
        "random_gamma": False,
    },
    "fitting_net": {"type": "dpa4_ener", "neuron": [8, 8]},
    "bridging_method": "ZBL",
    "bridging_r_inner": 0.8,
    "bridging_r_outer": 1.2,
}


def _close_pair_inputs():
    rng = np.random.default_rng(5)
    coord = rng.uniform(1.5, 5.5, size=(1, 6, 3))
    coord[0, 1] = coord[0, 0] + np.array([0.9, 0.0, 0.0])  # close Ni-Ni pair
    atype = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int64)
    box = 8.0 * np.eye(3, dtype=np.float64)[None]
    return coord, atype, box


def test_builder_composes_linear_model():
    model = get_model(copy.deepcopy(ZBL_CONFIG))
    assert type(model) is LinearEnergyModel
    am = model.atomic_model
    assert isinstance(am, LinearEnergyAtomicModel)
    assert am.weights == "sum"
    kinds = [type(c).__name__ for c in am.models]
    assert (
        kinds == ["EnergyAtomicModel", "InterPotentialAtomicModel"]
        or kinds[1] == "InterPotentialAtomicModel"
    )
    # radii wired to the LEARNED child's descriptor InnerClamp
    dp_child = am.models[0]
    assert dp_child.descriptor.inner_clamp is not None
    assert float(dp_child.descriptor.inner_clamp.r_inner) == 0.8


def test_zbl_child_equals_composition_minus_learned():
    """Composition energy == learned child + analytical child (exact sum)."""
    model = get_model(copy.deepcopy(ZBL_CONFIG))
    coord, atype, box = _close_pair_inputs()
    e_sum = model.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    dp_child, zbl_child = model.atomic_model.models
    # learned child alone through its OWN model wrapper
    from deepmd.dpmodel.model.ener_model import (
        EnergyModel,
    )

    m_dp = EnergyModel(atomic_model_=dp_child)
    e_dp = m_dp.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    diff = float(np.sum(e_sum - e_dp))
    assert diff > 1e-3, f"ZBL contribution missing or non-positive: {diff:.3e}"
    # EXACT analytical check (gas phase, no box: a direct double loop over
    # pairs within rcut is the complete reference).
    import math

    e_gas_sum = model.call_common(coord, atype, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    e_gas_dp = m_dp.call_common(coord, atype, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    z_of = {0: 28.0, 1: 8.0}  # Ni, O
    total = 0.0
    for i in range(6):
        for j in range(i + 1, 6):
            r = float(np.linalg.norm(coord[0, i] - coord[0, j]))
            if r >= 4.0:
                continue
            zi, zj = z_of[int(atype[0, i])], z_of[int(atype[0, j])]
            a = 0.88534 * 0.5291772109 / (zi**0.23 + zj**0.23)
            phi = sum(
                ak * math.exp(-bk * (r / a))
                for ak, bk in zip(
                    (0.18175, 0.50986, 0.28022, 0.028171),
                    (3.1998, 0.94229, 0.4029, 0.20162),
                    strict=True,
                )
            )
            total += 14.3996 * zi * zj / r * phi
    np.testing.assert_allclose(
        float(np.sum(e_gas_sum - e_gas_dp)), total, rtol=1e-10, atol=1e-10
    )


def test_zbl_serialize_roundtrip_energy_identical():
    model = get_model(copy.deepcopy(ZBL_CONFIG))
    coord, atype, box = _close_pair_inputs()
    data = model.serialize()
    # the flat wire type is "linear" -- the SAME string pt/tf write, so a
    # composition round-trips across backends
    assert data["type"] == "linear"
    m2 = BaseModel.deserialize(data)
    assert type(m2) is LinearEnergyModel
    e1 = model.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    e2 = m2.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    np.testing.assert_allclose(e1, e2, rtol=1e-12)


def test_zbl_atomic_dense_route_raises():
    zbl = InterPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
    with pytest.raises(NotImplementedError, match="NeighborGraph route only"):
        zbl.forward_atomic(None, None, None)


def test_inter_potential_supports_graph_lower():
    """The analytical ZBL term is graph-capable (rides the NeighborGraph)."""
    zbl = InterPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
    assert zbl.uses_graph_lower() is True


def test_linear_graph_lower_requires_all_children():
    """A composition is graph-capable iff EVERY child supports the graph lower.

    Regression: a dense-only child (a bare-minimum stand-in for
    ``PairTabAtomicModel``, standard DP+ZBL) forces the whole linear model
    onto the dense route, even alongside a graph-capable child -- otherwise
    the graph route would call ``forward_atomic_graph`` on the dense-only
    child, which does not implement it.
    """

    class _GraphChild(InterPotentialAtomicModel):
        pass  # inherits uses_graph_lower() -> True

    class _DenseOnlyChild(InterPotentialAtomicModel):
        def uses_graph_lower(self) -> bool:
            return False

    graph_child = _GraphChild(type_map=["Ni", "O"], rcut=4.0, sel=[8])
    dense_child = _DenseOnlyChild(type_map=["Ni", "O"], rcut=4.0, sel=[8])

    all_graph = LinearEnergyAtomicModel(
        [graph_child, _GraphChild(type_map=["Ni", "O"], rcut=4.0, sel=[8])],
        type_map=["Ni", "O"],
        weights="sum",
    )
    assert all_graph.uses_graph_lower() is True

    mixed = LinearEnergyAtomicModel(
        [graph_child, dense_child],
        type_map=["Ni", "O"],
        weights="sum",
    )
    assert mixed.uses_graph_lower() is False


def test_zbl_atomic_graph_values():
    """Atomic-model wrapper reproduces the kernel's known values."""
    import math

    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )

    r = 0.8
    zbl = InterPotentialAtomicModel(type_map=["O"], rcut=4.0, sel=[8])
    graph = NeighborGraph(
        n_node=np.array([2], dtype=np.int64),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_vec=np.array([[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=np.float64),
        edge_mask=np.ones(2, dtype=bool),
    )
    out = zbl.forward_common_atomic_graph(graph, np.zeros(2, dtype=np.int64))
    a = 0.88534 * 0.5291772109 / (8.0**0.23 + 8.0**0.23)
    phi = sum(
        ak * math.exp(-bk * (r / a))
        for ak, bk in zip(
            (0.18175, 0.50986, 0.28022, 0.028171),
            (3.1998, 0.94229, 0.4029, 0.20162),
            strict=True,
        )
    )
    ref = 14.3996 * 64.0 / r * phi
    np.testing.assert_allclose(float(np.sum(out["energy"])), ref, atol=1e-5)


def _pair_energy(model, natoms=2, r=1.0):
    """Total ZBL energy of one pair at distance ``r``, all atoms of type 0."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )

    graph = NeighborGraph(
        n_node=np.array([natoms], dtype=np.int64),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_vec=np.array([[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=np.float64),
        edge_mask=np.ones(2, dtype=bool),
    )
    out = model.forward_common_atomic_graph(graph, np.zeros(natoms, dtype=np.int64))
    return float(np.sum(out["energy"]))


class TestInterPotentialChangeTypeMap:
    """``change_type_map`` must rebuild the ZBL element lookup.

    The generic ``BaseAtomicModel.change_type_map`` only rewrites the public
    map and the stat/exclusion state; the nuclear-charge table belongs to
    ``InterPotential`` and is rebuilt there (review 3649295675).  Without it
    the lookup keeps the ORIGINAL elements while ``atype`` values already mean
    the new ones -- silently wrong energies, or ``IndexError`` for a longer
    map.
    """

    def test_reorder_matches_a_freshly_built_model(self) -> None:
        model = InterPotentialAtomicModel(type_map=["H", "O"], rcut=4.0, sel=[8])
        e_hh = _pair_energy(model)
        model.change_type_map(["O", "H"])
        fresh = InterPotentialAtomicModel(type_map=["O", "H"], rcut=4.0, sel=[8])
        e_fresh = _pair_energy(fresh)
        # anti-vacuity: the two element pairs must be far apart, or a stale
        # lookup would be indistinguishable from a rebuilt one
        assert abs(e_fresh - e_hh) > 1.0
        np.testing.assert_allclose(_pair_energy(model), e_fresh, rtol=1e-12)
        assert list(model.potential.atomic_numbers) == [8.0, 1.0]
        assert model.potential.type_map == ["O", "H"]

    def test_added_element_extends_the_lookup(self) -> None:
        model = InterPotentialAtomicModel(type_map=["H", "O"], rcut=4.0, sel=[8])
        model.change_type_map(["H", "O", "Ni"])
        assert model.potential.ntypes_real == 3
        assert list(model.potential.atomic_numbers) == [1.0, 8.0, 28.0]
        # the new type is now addressable -- a stale (length-2) table raises
        # IndexError here
        graph_atype = np.full(2, 2, dtype=np.int64)
        from deepmd.dpmodel.utils.neighbor_graph import (
            NeighborGraph,
        )

        graph = NeighborGraph(
            n_node=np.array([2], dtype=np.int64),
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_vec=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64),
            edge_mask=np.ones(2, dtype=bool),
        )
        e_nini = float(
            np.sum(model.forward_common_atomic_graph(graph, graph_atype)["energy"])
        )
        fresh = InterPotentialAtomicModel(type_map=["Ni"], rcut=4.0, sel=[8])
        np.testing.assert_allclose(e_nini, _pair_energy(fresh), rtol=1e-12)

    def test_dropped_element_shrinks_the_lookup(self) -> None:
        model = InterPotentialAtomicModel(type_map=["H", "O", "Ni"], rcut=4.0, sel=[8])
        model.change_type_map(["Ni"])
        assert model.potential.ntypes_real == 1
        assert list(model.potential.atomic_numbers) == [28.0]

    def test_serialize_roundtrip_after_change_type_map(self) -> None:
        """Checkpoint continuity: the restored model must predict the same.

        Serialization records the NEW public map, so a stale in-memory lookup
        and its deserialized twin disagree -- the restart-time symptom of the
        same bug.
        """
        from deepmd.dpmodel.atomic_model.base_atomic_model import (
            BaseAtomicModel,
        )

        model = InterPotentialAtomicModel(type_map=["H", "O"], rcut=4.0, sel=[8])
        model.change_type_map(["O", "H"])
        data = model.serialize()
        restored = BaseAtomicModel.get_class_by_type(data["type"]).deserialize(data)
        assert restored.get_type_map() == ["O", "H"]
        np.testing.assert_allclose(
            _pair_energy(restored), _pair_energy(model), rtol=1e-12
        )


class TestNativeSpinCapabilityOnAtomicModel:
    """``supports_native_spin`` is answered by the ATOMIC MODEL.

    The model layer must not reach into an atomic model for a descriptor to
    decide spin eligibility: an analytical term has no descriptor at all, and
    a composition has several children.  Each atomic model answers from its
    own structure, exactly like ``uses_graph_lower``.
    """

    def test_analytical_term_is_not_spin_capable(self) -> None:
        zbl = InterPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
        # inherits the concrete base default -- no descriptor, no spin input
        assert zbl.supports_native_spin() is False

    def test_composition_is_capable_when_any_child_is(self) -> None:
        """ANY, not ALL: analytical children accept and ignore ``spin``."""
        learned = get_model(
            {
                **copy.deepcopy(ZBL_CONFIG),
                "spin": {"use_spin": [True, False], "scheme": "native"},
            }
        ).atomic_model
        assert learned.supports_native_spin() is True
        kinds = [type(c).__name__ for c in learned.models]
        assert kinds[1] == "InterPotentialAtomicModel", kinds
        # ... and the spin-free analytical child alone is not capable
        assert learned.models[1].supports_native_spin() is False

    def test_composition_without_a_spin_consumer_is_not_capable(self) -> None:
        """No consumer => the magnetic force would be identically zero."""
        zbl_a = InterPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
        zbl_b = InterPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
        composed = LinearEnergyAtomicModel(
            [zbl_a, zbl_b], type_map=["Ni", "O"], weights="sum"
        )
        assert composed.supports_native_spin() is False


class TestCompositionForwardsConditioningCapabilities:
    """A composition must FORWARD every capability its children own.

    ``get_dim_fparam``/``get_dim_aparam`` were forwarded, but the
    charge/spin FiLM and default-fparam accessors fell through to
    ``BaseAtomicModel``'s ``False``/``0``.  That is silently wrong rather
    than loudly broken: the eager forward still conditions on
    ``charge_spin`` (the learned child consumes it), while the FREEZE reads
    these accessors -- so a 0 dropped the charge_spin slot from the exported
    ABI and from the metadata the C++ feeder reads, and the artifact
    disagreed with its own eager model.
    """

    @staticmethod
    def _model(bridging: bool, chg_spin: bool = True):
        config = copy.deepcopy(ZBL_CONFIG)
        config["descriptor"]["add_chg_spin_ebd"] = chg_spin
        if not bridging:
            for key in ("bridging_method", "bridging_r_inner", "bridging_r_outer"):
                config.pop(key, None)
        return get_model(config)

    def test_charge_spin_survives_bridging(self) -> None:
        plain = self._model(bridging=False)
        bridged = self._model(bridging=True)
        # anti-vacuity: the unbridged model must actually declare the input
        assert plain.get_dim_chg_spin() > 0
        assert bridged.get_dim_chg_spin() == plain.get_dim_chg_spin()
        assert bridged.has_chg_spin_ebd() is True
        # ... and the composition really is a composition
        assert [type(c).__name__ for c in bridged.atomic_model.models][1] == (
            "InterPotentialAtomicModel"
        )

    def test_no_charge_spin_stays_zero(self) -> None:
        """The other branch: nothing is invented when no child declares it."""
        bridged = self._model(bridging=True, chg_spin=False)
        assert bridged.get_dim_chg_spin() == 0
        assert bridged.has_chg_spin_ebd() is False

    def test_default_conditioning_accessors_are_forwarded(self) -> None:
        """``has_default_*`` must not fall through to the base either."""
        bridged = self._model(bridging=True)
        plain = self._model(bridging=False)
        assert bridged.has_default_chg_spin() == plain.has_default_chg_spin()
        assert bridged.has_default_fparam() == plain.has_default_fparam()
        assert bridged.get_default_fparam() == plain.get_default_fparam()
