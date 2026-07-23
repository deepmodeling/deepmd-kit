# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel tests for :class:`NativeSpinEnergyModel` (model-level native spin)."""

import copy

import numpy as np
import pytest

from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.dpmodel.model.native_spin_model import (
    NativeSpinEnergyModel,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)

NATIVE_SPIN_CONFIG = {
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
    "spin": {"use_spin": [True, False], "scheme": "native"},
}


def _jittered_model(seed: int):
    """Build the native-spin model then jitter its zero-init residual arrays.

    A fresh DPA4 zero-initializes several residual projections, so it is
    architecturally edge/spin-independent (see
    ``dpa4_fixtures.jitter_zero_arrays`` docstring); serialize -> jitter ->
    deserialize makes the sensitivity/round-trip tests non-vacuous.
    """
    model = get_model(NATIVE_SPIN_CONFIG)
    data = model.serialize()
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
    return BaseModel.deserialize(data)


class TestDPA4NativeSpinModel:
    def setup_method(self):
        self.model = _jittered_model(seed=11)
        rng = np.random.default_rng(5)
        self.nf, self.nloc = 1, 6
        self.coord = rng.uniform(0.5, 5.5, size=(self.nf, self.nloc, 3))
        self.atype = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int64)
        self.spin = rng.normal(size=(self.nf, self.nloc, 3))
        self.box = 8.0 * np.eye(3, dtype=np.float64)[None]

    def test_call_returns_energy_and_mask_mag(self):
        out = self.model.call(self.coord, self.atype, self.spin, box=self.box)
        assert out["energy"].shape == (self.nf, 1)
        assert out["atom_energy"].shape == (self.nf, self.nloc, 1)
        np.testing.assert_array_equal(
            out["mask_mag"][..., 0], self.atype == 0
        )  # use_spin=[True, False]

    def test_spin_sensitivity(self):
        out0 = self.model.call(self.coord, self.atype, self.spin, box=self.box)
        out1 = self.model.call(self.coord, self.atype, 2.0 * self.spin, box=self.box)
        assert not np.allclose(out0["energy"], out1["energy"])

    def test_serialize_roundtrip(self):
        data = self.model.serialize()
        assert data["type"] == "native_spin"
        model2 = BaseModel.deserialize(data)
        out0 = self.model.call(self.coord, self.atype, self.spin, box=self.box)
        out1 = model2.call(self.coord, self.atype, self.spin, box=self.box)
        np.testing.assert_allclose(out0["energy"], out1["energy"], rtol=1e-12)

    @pytest.mark.parametrize(
        "legacy_type",
        [
            "dpa4_native_spin",  # descriptor-specific pre-rename wire type
            "sezm_native_spin",  # pt backend's own wire type (pt payload layout differs)
        ],
    )
    def test_legacy_wire_types_fail_fast(self, legacy_type):
        # NativeSpinEnergyModel is descriptor-agnostic; mapping it to a
        # descriptor-specific wire string would be confusing, so the ONLY
        # registered type is "native_spin". Legacy strings raise the
        # registry's unknown-type error instead of silently dispatching.
        data = self.model.serialize()
        data["type"] = legacy_type
        with pytest.raises(RuntimeError, match=legacy_type):
            BaseModel.deserialize(data)

    def test_dense_route_spin_raises(self):
        # The model IS the standard model (is-a): call_common is its own.
        with pytest.raises(NotImplementedError, match="NeighborGraph"):
            self.model.call_common(
                self.coord,
                self.atype,
                self.box,
                spin=self.spin,
                neighbor_graph_method="legacy",
            )

    def test_deepspin_scheme_with_dpa4_raises(self):
        cfg = {**NATIVE_SPIN_CONFIG, "spin": {"use_spin": [True, False]}}
        with pytest.raises(NotImplementedError):
            get_model(cfg)

    def test_non_native_spin_descriptor_raises(self):
        # The gate is the ``supports_native_spin()`` capability, not a
        # descriptor-type list.
        cfg = copy.deepcopy(NATIVE_SPIN_CONFIG)
        cfg["descriptor"] = {
            "type": "se_e2_a",
            "rcut": 4.0,
            "rcut_smth": 3.5,
            "sel": [8, 8],
        }
        with pytest.raises(NotImplementedError, match="native spin"):
            get_model(cfg)

    def test_add_chg_spin_ebd_combined_builds_and_conditions(self):
        # Combined public configuration (review 3638047227): charge-spin
        # FiLM together with native spin, as in pt's SeZMNativeSpinModel.
        cfg = copy.deepcopy(NATIVE_SPIN_CONFIG)
        cfg["descriptor"]["add_chg_spin_ebd"] = True
        model = get_model(cfg)
        assert model.has_chg_spin_ebd()
        assert model.has_spin()
        # Jitter: fresh DPA4 zero-init is architecturally input-independent.
        data = model.serialize()
        data = jitter_zero_arrays(data, np.random.default_rng(13))
        model = BaseModel.deserialize(data)
        dim_cs = model.get_dim_chg_spin()
        assert dim_cs > 0
        # charge_spin is CATEGORICAL: ChargeSpinEmbedding casts the frame
        # (charge, spin) pair to int64 lookup indices, so only integer-valued
        # changes condition the model (0.5 would truncate to 0 == baseline).
        cs0 = np.zeros((self.nf, dim_cs), dtype=np.float64)
        cs1 = np.array([[1.0, 2.0]], dtype=np.float64)
        out_base = model.call(
            self.coord, self.atype, self.spin, box=self.box, charge_spin=cs0
        )
        out_cs = model.call(
            self.coord, self.atype, self.spin, box=self.box, charge_spin=cs1
        )
        # charge_spin conditions the energy...
        assert not np.allclose(out_base["energy"], out_cs["energy"])
        # ...and spin still conditions it in the SAME combined model.
        out_spin = model.call(
            self.coord, self.atype, 2.0 * self.spin, box=self.box, charge_spin=cs0
        )
        assert not np.allclose(out_base["energy"], out_spin["energy"])

    def test_translated_output_def_has_spin_keys(self):
        out_def = self.model.translated_output_def()
        assert "mask_mag" in out_def
        assert "force_mag" in out_def
        assert "energy" in out_def
        assert "force" in out_def


class TestNativeSpinConfigForms:
    """``spin.use_spin`` index/symbol forms and ``allow_missing_label``.

    The public schema accepts a per-type boolean list, a list of magnetic
    type indices, or a list of element symbols (expanded against
    ``type_map`` by ``normalize_spin_use_spin``); ``allow_missing_label``
    must be forwarded into the constructed :class:`Spin`.
    """

    @pytest.mark.parametrize(
        ("use_spin_form", "expected"),
        [
            (["Ni"], [True, False]),  # element-symbol form
            ([0], [True, False]),  # type-index form
            ([0, 1], [True, True]),  # multiple type indices
            ([True, False], [True, False]),  # canonical boolean passthrough
        ],
    )
    def test_use_spin_forms(self, use_spin_form, expected):
        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        config["spin"] = {"use_spin": use_spin_form, "scheme": "native"}
        model = get_model(config)
        assert model.spin.use_spin.tolist() == expected
        # The descriptor consumes the SAME normalized boolean list.
        descriptor = model.atomic_model.descriptor
        assert [bool(flag) for flag in descriptor.use_spin] == expected

    def test_use_spin_unknown_symbol_raises(self):
        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        config["spin"] = {"use_spin": ["Fe"], "scheme": "native"}
        with pytest.raises(ValueError, match="absent from type_map"):
            get_model(config)

    def test_allow_missing_label_forwarded(self):
        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        config["spin"]["allow_missing_label"] = True
        model = get_model(config)
        assert model.spin.allow_missing_label is True

    def test_allow_missing_label_default_false(self):
        model = get_model(copy.deepcopy(NATIVE_SPIN_CONFIG))
        assert model.spin.allow_missing_label is False


class TestNativeSpinModelRegistryDispatch:
    """Wire-type + registry contract of ``make_native_spin_model`` classes.

    Review 3638137290 (PR #5884): dispatch goes through each backend's
    plugin registry (backend-aware), not a hard-coded branch in
    ``BaseBaseModel.deserialize``; the serialized shape is the make_model
    FLAT dict + ``spin`` field, not the legacy nested wrapper shape.
    """

    def _inputs(self):
        rng = np.random.default_rng(5)
        coord = rng.uniform(0.5, 5.5, size=(1, 6, 3))
        atype = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int64)
        spin = rng.normal(size=(1, 6, 3))
        box = 8.0 * np.eye(3, dtype=np.float64)[None]
        return coord, atype, spin, box

    def test_serialize_emits_native_spin_flat_shape(self):
        model = get_model(copy.deepcopy(NATIVE_SPIN_CONFIG))
        assert type(model) is NativeSpinEnergyModel
        data = model.serialize()
        assert data["type"] == "native_spin"
        assert "spin" in data
        assert "backbone_model" not in data  # make_model flat shape, not nested

    def test_basemodel_deserialize_dispatches_via_registry(self):
        model = get_model(copy.deepcopy(NATIVE_SPIN_CONFIG))
        m2 = BaseModel.deserialize(model.serialize())
        assert type(m2) is NativeSpinEnergyModel
        coord, atype, spin, box = self._inputs()
        e1 = model.call(coord, atype, spin, box=box)["energy"]
        e2 = m2.call(coord, atype, spin, box=box)["energy"]
        np.testing.assert_allclose(e1, e2, rtol=1e-12)
