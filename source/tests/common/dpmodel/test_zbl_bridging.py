# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel ZBL bridging as COMPOSITION (review 3638077323, redesigned).

``bridging_method: ZBL`` builds a
``LinearEnergyModel(LinearEnergyAtomicModel([dp, InnerPotentialAtomicModel],
weights="sum"))`` -- the analytical term is its own atomic model summed
with the learned one, not a flag on it.
"""

import copy

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model.inner_potential import (
    InnerPotentialAtomicModel,
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
        kinds == ["EnergyAtomicModel", "InnerPotentialAtomicModel"]
        or kinds[1] == "InnerPotentialAtomicModel"
    )
    # radii wired to the LEARNED child's descriptor InnerClamp
    dp_child = am.models[0]
    assert dp_child.descriptor.inner_clamp is not None
    assert float(dp_child.descriptor.inner_clamp.r_inner) == 0.8


def test_third_child_without_common_route_raises():
    """[learned, inner_potential, pairtab] has no common execution route
    (pairtab is dense-only, the bridged pair is graph-only): the builder
    must reject it at construction like the pt backend does.
    """
    cfg = {
        "type": "linear_ener",
        "weights": "sum",
        "type_map": ["Ni", "O"],
        "models": [
            {
                "type": "dpa4",
                "descriptor": copy.deepcopy(ZBL_CONFIG["descriptor"]),
                "fitting_net": copy.deepcopy(ZBL_CONFIG["fitting_net"]),
            },
            {"type": "inner_potential", "mode": "ZBL"},
            {"type": "pairtab", "tab_file": "unused.txt", "rcut": 4.0, "sel": 8},
        ],
    }
    with pytest.raises(ValueError, match="exactly one learned"):
        get_model(cfg)


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
    zbl = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
    with pytest.raises(NotImplementedError, match="NeighborGraph route only"):
        zbl.forward_atomic(None, None, None)


def test_inner_potential_supports_graph_lower():
    """The analytical ZBL term is graph-capable (rides the NeighborGraph)."""
    zbl = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
    assert zbl.uses_graph_lower() is True


def test_linear_graph_lower_requires_all_children():
    """A composition is graph-capable iff EVERY child supports the graph lower.

    Regression: a dense-only child (a bare-minimum stand-in for
    ``PairTabAtomicModel``, standard DP+ZBL) forces the whole linear model
    onto the dense route, even alongside a graph-capable child -- otherwise
    the graph route would call ``forward_atomic_graph`` on the dense-only
    child, which does not implement it.
    """

    class _GraphChild(InnerPotentialAtomicModel):
        pass  # inherits uses_graph_lower() -> True

    class _DenseOnlyChild(InnerPotentialAtomicModel):
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
    zbl = InnerPotentialAtomicModel(type_map=["O"], rcut=4.0, sel=[8])
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


class TestInnerPotentialChangeTypeMap:
    """``change_type_map`` must rebuild the ZBL element lookup.

    The generic ``BaseAtomicModel.change_type_map`` only rewrites the public
    map and the stat/exclusion state; the nuclear-charge table belongs to
    ``InnerPotential`` and is rebuilt there (review 3649295675).  Without it
    the lookup keeps the ORIGINAL elements while ``atype`` values already mean
    the new ones -- silently wrong energies, or ``IndexError`` for a longer
    map.
    """

    def test_reorder_matches_a_freshly_built_model(self) -> None:
        model = InnerPotentialAtomicModel(type_map=["H", "O"], rcut=4.0, sel=[8])
        e_hh = _pair_energy(model)
        model.change_type_map(["O", "H"])
        fresh = InnerPotentialAtomicModel(type_map=["O", "H"], rcut=4.0, sel=[8])
        e_fresh = _pair_energy(fresh)
        # anti-vacuity: the two element pairs must be far apart, or a stale
        # lookup would be indistinguishable from a rebuilt one
        assert abs(e_fresh - e_hh) > 1.0
        np.testing.assert_allclose(_pair_energy(model), e_fresh, rtol=1e-12)
        assert list(model.potential.atomic_numbers) == [8.0, 1.0]
        assert model.potential.type_map == ["O", "H"]

    def test_added_element_extends_the_lookup(self) -> None:
        model = InnerPotentialAtomicModel(type_map=["H", "O"], rcut=4.0, sel=[8])
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
        fresh = InnerPotentialAtomicModel(type_map=["Ni"], rcut=4.0, sel=[8])
        np.testing.assert_allclose(e_nini, _pair_energy(fresh), rtol=1e-12)

    def test_dropped_element_shrinks_the_lookup(self) -> None:
        model = InnerPotentialAtomicModel(type_map=["H", "O", "Ni"], rcut=4.0, sel=[8])
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

        model = InnerPotentialAtomicModel(type_map=["H", "O"], rcut=4.0, sel=[8])
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
        zbl = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
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
        assert kinds[1] == "InnerPotentialAtomicModel", kinds
        # ... and the spin-free analytical child alone is not capable
        assert learned.models[1].supports_native_spin() is False

    def test_composition_without_a_spin_consumer_is_not_capable(self) -> None:
        """No consumer => the magnetic force would be identically zero."""
        zbl_a = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
        zbl_b = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
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
            "InnerPotentialAtomicModel"
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


class TestCompositionCarriesPairExclusion:
    """Model-level ``pair_exclude_types`` must survive the ZBL composition.

    It is a BUILD-time transform: the atomic model only carries the config,
    and whoever builds the neighbor graph (the C++ feeder, or the Python
    builder) applies it from the exported metadata. ``_collect_metadata``
    reads it off ``model.atomic_model`` -- which for a bridged model is the
    ``LinearEnergyAtomicModel``, not the learned child. Building that
    composition without forwarding the exclusion therefore dropped it
    silently: the eager model still behaved, while the frozen artifact told
    its feeder there was nothing to exclude.
    """

    @staticmethod
    def _model(bridging: bool, spin: bool = False):
        config = copy.deepcopy(ZBL_CONFIG)
        config["pair_exclude_types"] = [[0, 1]]
        if spin:
            config["spin"] = {"use_spin": [True, False], "scheme": "native"}
        if not bridging:
            for key in ("bridging_method", "bridging_r_inner", "bridging_r_outer"):
                config.pop(key, None)
        return get_model(config)

    def test_pair_exclusion_survives_bridging(self) -> None:
        plain = self._model(bridging=False)
        bridged = self._model(bridging=True)
        # anti-vacuity: the unbridged model must actually carry it
        assert plain.atomic_model.pair_exclude_types == [[0, 1]]
        assert bridged.atomic_model.pair_exclude_types == [[0, 1]]
        # ... and the bridged one really is the composition, so the value is
        # read off the wrapper rather than accidentally off a lone child.
        assert [type(c).__name__ for c in bridged.atomic_model.models][1] == (
            "InnerPotentialAtomicModel"
        )

    def test_pair_exclusion_survives_native_spin_plus_bridging(self) -> None:
        """The three-way stack must not drop it either."""
        assert self._model(
            bridging=True, spin=True
        ).atomic_model.pair_exclude_types == [[0, 1]]

    def test_no_exclusion_stays_empty(self) -> None:
        """The other branch: nothing is invented when none is configured."""
        config = copy.deepcopy(ZBL_CONFIG)
        config.pop("pair_exclude_types", None)
        assert get_model(config).atomic_model.pair_exclude_types == []


class TestCompositionCarriesAtomExclusion:
    """``atom_exclude_types`` must reach the ZBL term too.

    The twin of :class:`TestCompositionCarriesPairExclusion`. Unlike pair
    exclusion this one IS applied at runtime, which is why it was initially
    (and wrongly) left unforwarded on the theory that forwarding would
    double-apply. It does not: the composition's own ``atom_excl`` was
    ``None``, the learned child masked only itself, and the analytical child
    never heard about the exclusion -- so an excluded atom still collected
    its full share of the ZBL energy. Masking to zero is idempotent, so the
    child keeping its own copy is harmless.
    """

    @staticmethod
    def _model(bridging: bool):
        config = copy.deepcopy(ZBL_CONFIG)
        config["atom_exclude_types"] = [1]
        if not bridging:
            for key in ("bridging_method", "bridging_r_inner", "bridging_r_outer"):
                config.pop(key, None)
        return get_model(config)

    def test_atom_exclusion_reaches_the_composition(self) -> None:
        bridged = self._model(bridging=True)
        plain = self._model(bridging=False)
        # anti-vacuity: the unbridged model must actually carry it
        assert plain.atomic_model.atom_exclude_types == [1]
        assert bridged.atomic_model.atom_exclude_types == [1]
        # the mask is what actually zeroes the analytical child's output
        assert bridged.atomic_model.atom_excl is not None

    def test_no_exclusion_stays_empty(self) -> None:
        """Nothing is invented when none is configured."""
        config = copy.deepcopy(ZBL_CONFIG)
        config.pop("atom_exclude_types", None)
        model = get_model(config)
        assert model.atomic_model.atom_exclude_types == []
        assert model.atomic_model.atom_excl is None


class TestCompositionDefaultsRequireAgreement:
    """A parent default is only valid when every ACTIVE child shares it.

    The composition exposes ONE external tensor to all children, so
    advertising the first child's default made ``get_additional_data_
    requirement`` mark the input optional and inject that value into every
    child -- silently overriding the others' own defaults (reported as an
    8.13e-4 energy change for two learned children defaulting to [0.0] and
    [1.0]). Dimension-zero children (an analytical bridging term) are not
    consumers and must be ignored, so learned+ZBL still inherits the
    learned default.
    """

    class _Fake:
        """Learned-child stand-in with its own fparam default."""

        def __init__(self, dim: int, default) -> None:
            self._dim, self._default = dim, default

        def mixed_types(self) -> bool:
            return True

        def get_type_map(self) -> list:
            return ["Ni", "O"]

        def get_intensive(self) -> bool:
            return False

        def get_dim_fparam(self) -> int:
            return self._dim

        def get_dim_aparam(self) -> int:
            return 0

        def get_dim_chg_spin(self) -> int:
            return 0

        def has_default_fparam(self) -> bool:
            return self._default is not None

        def get_default_fparam(self):
            return self._default

        def has_default_chg_spin(self) -> bool:
            return False

        def get_default_chg_spin(self):
            return None

    def _compose(self, children):
        return LinearEnergyAtomicModel(children, type_map=["Ni", "O"], weights="sum")

    def test_consumers_must_agree_on_dimension(self) -> None:
        """One shared tensor means one dimension among ACTIVE consumers.

        Rejected at construction, like the intensive/extensive mixture: the
        composition feeds every child the same fparam/aparam tensor, so
        consumers wanting different widths cannot both be satisfied.
        """
        with pytest.raises(ValueError, match="fparam dimension"):
            self._compose([self._Fake(3, None), self._Fake(2, None)])

    def test_dimension_and_default_align_with_the_learned_child(self) -> None:
        """Learned + ZBL inherits BOTH the dimension and the default.

        The analytical child consumes neither fparam nor aparam, so it is
        not a consumer and must not constrain either.
        """
        zbl = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
        # anti-vacuity: the analytical child really is a non-consumer
        assert zbl.get_dim_fparam() == 0
        assert zbl.get_dim_aparam() == 0
        m = self._compose([self._Fake(3, [0.5, 0.5, 0.5]), zbl])
        assert m.get_dim_fparam() == 3
        assert m.get_default_fparam() == [0.5, 0.5, 0.5]

    def test_differing_defaults_expose_none(self) -> None:
        m = self._compose([self._Fake(1, [0.0]), self._Fake(1, [1.0])])
        assert m.has_default_fparam() is False
        assert m.get_default_fparam() is None

    def test_matching_defaults_are_exposed(self) -> None:
        m = self._compose([self._Fake(1, [1.0]), self._Fake(1, [1.0])])
        assert m.has_default_fparam() is True
        assert m.get_default_fparam() == [1.0]

    def test_dimension_zero_child_is_ignored(self) -> None:
        """Learned + ZBL must still inherit the learned default."""
        zbl = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
        assert zbl.get_dim_fparam() == 0  # anti-vacuity: really a non-consumer
        m = self._compose([self._Fake(1, [0.5]), zbl])
        assert m.has_default_fparam() is True
        assert m.get_default_fparam() == [0.5]

    def test_active_child_without_default_exposes_none(self) -> None:
        m = self._compose([self._Fake(1, [1.0]), self._Fake(1, None)])
        assert m.has_default_fparam() is False


class TestCompositionForwardsStatCapabilities:
    """``get_intensive`` / ``get_compute_stats_distinguish_types`` aggregate.

    Both silently fell through to ``BaseAtomicModel``'s defaults, so a
    bridged model would fit its out-stat bias with the wrong extensivity and
    the wrong type-distinguishing rule -- the same class of gap as the
    charge-spin and pair-exclusion accessors.
    """

    def test_intensive_mixture_is_rejected_at_construction(self) -> None:
        """An intensive/extensive mixture must not be CONSTRUCTIBLE.

        Rejected in ``__init__`` rather than at query time: such a
        composition is not physically meaningful, so it should never exist
        rather than exist and answer a plausible-looking default.
        """
        zbl_a = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
        zbl_b = InnerPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
        # anti-vacuity: matching children compose fine and report their value
        assert zbl_a.get_intensive() is False
        assert (
            LinearEnergyAtomicModel(
                [zbl_a, zbl_b], type_map=["Ni", "O"], weights="sum"
            ).get_intensive()
            is False
        )

        zbl_b.get_intensive = lambda: True  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="intensive and extensive"):
            LinearEnergyAtomicModel([zbl_a, zbl_b], type_map=["Ni", "O"], weights="sum")

    def test_forwarded_from_children(self) -> None:
        bridged = get_model(copy.deepcopy(ZBL_CONFIG))
        children = bridged.atomic_model.models
        assert bridged.atomic_model.get_intensive() == all(
            c.get_intensive() for c in children
        )
        assert bridged.atomic_model.get_compute_stats_distinguish_types() == any(
            c.get_compute_stats_distinguish_types() for c in children
        )


def _canonical_config() -> dict:
    """``ZBL_CONFIG`` spelled canonically (issue #5948): an explicit
    ``linear_ener`` composition with an ``inner_potential`` sub-model.
    """
    cfg = copy.deepcopy(ZBL_CONFIG)
    cfg["fitting_net"]["seed"] = 7
    return {
        "type": "linear_ener",
        "weights": "sum",
        "type_map": cfg["type_map"],
        "models": [
            {
                "type": "standard",
                "descriptor": cfg["descriptor"],
                "fitting_net": cfg["fitting_net"],
            },
            {
                "type": "inner_potential",
                "mode": "ZBL",
                "r_inner": 0.8,
                "r_outer": 1.2,
            },
        ],
    }


class TestCanonicalComposition:
    """The canonical ``linear_ener`` + ``inner_potential`` spelling."""

    def test_canonical_config_composes(self) -> None:
        model = get_model(_canonical_config())
        assert type(model) is LinearEnergyModel
        am = model.atomic_model
        assert isinstance(am, LinearEnergyAtomicModel)
        assert am.weights == "sum"
        assert isinstance(am.models[1], InnerPotentialAtomicModel)
        # the composition derives the learned sibling's clamp window from
        # the inner_potential child: one source of truth for the radii
        dp_child = am.models[0]
        assert dp_child.descriptor.inner_clamp is not None
        assert float(dp_child.descriptor.inner_clamp.r_inner) == 0.8
        assert dp_child.descriptor.bridging_switch is not None

    def test_canonical_matches_sugar_energy(self) -> None:
        """Same seeds, both spellings: bit-identical construction, so the
        energies must be exactly equal.
        """
        sugar = copy.deepcopy(ZBL_CONFIG)
        sugar["fitting_net"]["seed"] = 7
        m_sugar = get_model(sugar)
        m_canon = get_model(_canonical_config())
        coord, atype, box = _close_pair_inputs()
        e_sugar = m_sugar.call_common(
            coord, atype, box=box, neighbor_graph_method="dense"
        )["energy_redu"]
        e_canon = m_canon.call_common(
            coord, atype, box=box, neighbor_graph_method="dense"
        )["energy_redu"]
        np.testing.assert_array_equal(e_canon, e_sugar)

    def test_canonical_serialize_matches_sugar(self) -> None:
        """Both spellings serialize to the same wire dict: the flag is
        sugar, not a different model.
        """
        sugar = copy.deepcopy(ZBL_CONFIG)
        sugar["fitting_net"]["seed"] = 7
        d_sugar = get_model(sugar).serialize()
        d_canon = get_model(_canonical_config()).serialize()

        def _strip_arrays(obj):
            if isinstance(obj, dict):
                return {k: _strip_arrays(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_strip_arrays(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return ("ndarray", obj.shape)
            return obj

        assert _strip_arrays(d_canon) == _strip_arrays(d_sugar)

    def test_two_inner_children_raise(self) -> None:
        cfg = _canonical_config()
        cfg["models"].append(dict(cfg["models"][1]))
        with pytest.raises(ValueError, match="at most one"):
            get_model(cfg)

    def test_inner_without_learned_sibling_raises(self) -> None:
        cfg = _canonical_config()
        cfg["models"] = [cfg["models"][1]]
        with pytest.raises(ValueError, match="exactly one learned"):
            get_model(cfg)

    def test_standard_builder_rejects_the_flag(self) -> None:
        """Direct standard construction with the flag fails fast instead of
        silently dropping the analytical term.
        """
        from deepmd.dpmodel.model.model import (
            get_standard_model,
        )

        with pytest.raises(ValueError, match="bridging_method"):
            get_standard_model(copy.deepcopy(ZBL_CONFIG))


class TestCanonicalCompositionGuards:
    """Fail-fast guards of the shared linear builder."""

    def test_mean_weights_with_inner_child_raise(self) -> None:
        """`weights: "mean"` would silently halve both energy terms."""
        cfg = _canonical_config()
        cfg["weights"] = "mean"
        with pytest.raises(ValueError, match="sum"):
            get_model(cfg)

    def test_nested_bridging_flag_on_child_raises(self) -> None:
        """A `bridging_method` flag on a linear child must not be dropped."""
        cfg = _canonical_config()
        cfg["models"] = [cfg["models"][0]]
        cfg["models"][0]["bridging_method"] = "ZBL"
        with pytest.raises(ValueError, match="sub-model"):
            get_model(cfg)

    def test_inner_child_with_descriptor_raises_cleanly(self) -> None:
        """A child carrying both `type: inner_potential` and a descriptor is
        a configuration error, not a KeyError.
        """
        cfg = _canonical_config()
        cfg["models"][1]["descriptor"] = {"type": "dpa4"}
        with pytest.raises(ValueError, match="must not carry"):
            get_model(cfg)

    def test_canonical_rejects_mismatched_learned_type_map(self) -> None:
        """A remapped learned-child type_map builds a model the graph
        route rejects on every forward; fail at construction instead.
        """
        cfg = _canonical_config()
        cfg["models"][0]["type_map"] = list(reversed(cfg["type_map"]))
        with pytest.raises(ValueError, match="type_map"):
            get_model(cfg)

    def test_canonical_conflicting_top_level_option_raises(self) -> None:
        """A learned-owned option set differently at both levels must
        fail loudly instead of one value silently winning.
        """
        cfg = _canonical_config()
        cfg["data_stat_protect"] = 0.123
        cfg["models"][0]["data_stat_protect"] = 0.456
        with pytest.raises(ValueError, match="data_stat_protect"):
            get_model(cfg)

    def test_update_sel_dispatches_and_skips_inner_child(self, monkeypatch) -> None:
        """``BaseModel.update_sel`` dispatches ``linear_ener`` to a
        composite implementation that updates the learned child and
        skips the analytical one (the default neighbor-stat phase would
        otherwise crash with ``KeyError: 'descriptor'``).
        """
        from deepmd.dpmodel.model.dp_model import (
            DPModelCommon,
        )
        from deepmd.utils.argcheck import (
            model_args,
        )

        seen = []

        def _fake_update_sel(train_data, type_map, sub):
            seen.append(copy.deepcopy(sub))
            return sub, 0.9

        monkeypatch.setattr(DPModelCommon, "update_sel", staticmethod(_fake_update_sel))
        cfg = model_args().normalize_value(_canonical_config(), trim_pattern="_*")
        updated, min_dist = BaseModel.update_sel(None, cfg["type_map"], cfg)
        assert min_dist == 0.9
        assert len(seen) == 1  # only the learned child
        assert "descriptor" in seen[0]
        assert updated["models"][1]["type"] == "inner_potential"
