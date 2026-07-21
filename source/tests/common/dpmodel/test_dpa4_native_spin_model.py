# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel tests for :class:`DPA4NativeSpinModel` (model-level native spin)."""

import copy

import numpy as np
import pytest

from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.model import (
    get_model,
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
        assert data["type"] == "dpa4_native_spin"
        model2 = BaseModel.deserialize(data)
        out0 = self.model.call(self.coord, self.atype, self.spin, box=self.box)
        out1 = model2.call(self.coord, self.atype, self.spin, box=self.box)
        np.testing.assert_allclose(out0["energy"], out1["energy"], rtol=1e-12)

    def test_sezm_native_spin_alias_deserializes(self):
        data = self.model.serialize()
        data["type"] = "sezm_native_spin"  # pt wire string
        model2 = BaseModel.deserialize(data)
        out1 = model2.call(self.coord, self.atype, self.spin, box=self.box)
        np.testing.assert_allclose(
            self.model.call(self.coord, self.atype, self.spin, box=self.box)["energy"],
            out1["energy"],
            rtol=1e-12,
        )

    def test_dense_route_spin_raises(self):
        with pytest.raises(NotImplementedError, match="NeighborGraph"):
            self.model.backbone_model.call_common(
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

    def test_non_dpa4_descriptor_raises(self):
        cfg = copy.deepcopy(NATIVE_SPIN_CONFIG)
        cfg["descriptor"] = {"type": "se_e2_a"}
        with pytest.raises(NotImplementedError, match="DPA4/SeZM"):
            get_model(cfg)

    def test_add_chg_spin_ebd_raises(self):
        cfg = copy.deepcopy(NATIVE_SPIN_CONFIG)
        cfg["descriptor"]["add_chg_spin_ebd"] = True
        with pytest.raises(NotImplementedError, match="charge-spin"):
            get_model(cfg)

    def test_translated_output_def_has_spin_keys(self):
        out_def = self.model.translated_output_def()
        assert "mask_mag" in out_def
        assert "force_mag" in out_def
        assert "energy" in out_def
        assert "force" in out_def
