# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fast (no-AOTI) tests for the pt -> pt_expt DPA4/SeZM checkpoint interop.

``BaseModel.deserialize`` recognises pt's ``SeZMModel`` wrapper (top-level
``type`` in {SeZM, sezm, dpa4}, ``@version`` 1) and its ``sezm_atomic`` atomic
dict (``@version`` 3), validates the versions, strips the pt-only ``dens`` head
state, and rejects pt-only features pt_expt does not implement.  These cases
are otherwise only exercised by the CI-skipped AOTI parity test
(``source/tests/pt_expt/infer/test_dpa4_deep_eval.py``); the tests here run in
CI and need neither ``torch.export`` nor AOTInductor.
"""

from __future__ import (
    annotations,
)

import copy

import pytest
import torch

from deepmd.pt.model.model import (
    get_model as pt_get_model,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

# Small fp64 DPA4 config (channels 8, n_radial 4, lmax 1, mmax 1, n_blocks 1)
# -- only large enough to serialize a real pt SeZM wrapper + sezm_atomic dict.
_DPA4_RAW_CONFIG = {
    "type": "dpa4",
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4",
        "sel": 20,
        "rcut": 4.0,
        "channels": 8,
        "n_radial": 4,
        "lmax": 1,
        "mmax": 1,
        "n_blocks": 1,
        "precision": "float64",
        "seed": 1,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [8],
        "precision": "float64",
        "seed": 1,
    },
}


def _normalize_model(model: dict) -> dict:
    config = {
        "model": copy.deepcopy(model),
        "training": {"training_data": {"systems": ["dummy"]}, "numb_steps": 1},
        "loss": {"type": "ener"},
        "learning_rate": {"type": "exp", "start_lr": 1e-3},
    }
    config = update_deepmd_input(config, warning=False)
    config = normalize(config)
    return config["model"]


@pytest.fixture(scope="module")
def pt_dpa4_model():
    """Build one real pt SeZMModel (fp64, eval); reused across tests.

    Each test calls ``.serialize()`` fresh (it returns new nested dicts), so
    in-place mutation of the serialized payload is isolated per test.
    """
    model_params = _normalize_model(_DPA4_RAW_CONFIG)
    model = pt_get_model(copy.deepcopy(model_params)).to(torch.float64)
    model.eval()
    return model


def _forward_smoke(model: EnergyModel) -> dict:
    """Run a tiny forward pass to prove the deserialized model is functional."""
    model = model.to(env.DEVICE)
    generator = torch.Generator(device=env.DEVICE).manual_seed(1)
    cell = 5.0 * torch.eye(3, dtype=torch.float64, device=env.DEVICE)
    coord = (
        torch.rand(
            [1, 5, 3],
            dtype=torch.float64,
            device=env.DEVICE,
            generator=generator,
        )
        @ cell
    ).requires_grad_(True)
    atype = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int64, device=env.DEVICE)
    return model(coord, atype, cell.reshape(1, 9))


class TestDPA4Interop:
    def test_serialize_layout(self, pt_dpa4_model) -> None:
        """The pt serialize layout matches the interop override's expectations."""
        ser = pt_dpa4_model.serialize()
        # wrapper: recognised model type + @version 1
        assert ser["type"].lower() in BaseModel._SEZM_MODEL_TYPES
        assert ser["@version"] == 1
        # nested atomic: sezm_atomic @version 3 carrying the pt-only dens state
        atomic = ser["atomic_model"]
        assert atomic["type"] in BaseModel._SEZM_ATOMIC_TYPES
        assert atomic["@version"] == 3
        assert "dens_force_rmsd" in atomic["@variables"]
        assert "active_mode" in atomic

    def test_happy_path_deserialize_and_forward(self, pt_dpa4_model) -> None:
        """A real pt checkpoint deserializes to a working pt_expt EnergyModel."""
        ser = pt_dpa4_model.serialize()
        model = BaseModel.deserialize(ser)
        assert isinstance(model, EnergyModel)
        ret = _forward_smoke(model)
        assert ret["energy"].shape == (1, 1)
        assert ret["force"].shape == (1, 5, 3)

    def test_variables_filtered_to_out_bias_out_std(self, pt_dpa4_model) -> None:
        """The pt-only ``dens_force_rmsd`` @variable is dropped on normalize."""
        atomic = pt_dpa4_model.serialize()["atomic_model"]
        assert set(atomic["@variables"]) >= {"out_bias", "out_std", "dens_force_rmsd"}
        normalized = BaseModel._normalize_pt_sezm_atomic(atomic)
        assert set(normalized["@variables"]) == {"out_bias", "out_std"}
        # version coerced to the standard atomic schema, type rewritten
        assert normalized["@version"] == 2
        assert normalized["type"] == "standard"

    # mutator(ser) edits the full pt wrapper serialize in place to trip one
    # guard; (exc_type, match) is the expected raise.  The wrapper @version
    # check runs before everything in _unwrap; the atomic @version check runs
    # first in _normalize -- both reject out-of-range versions loudly.
    @pytest.mark.parametrize(
        "mutator, exc_type, match",
        [
            # bridging_method != none -> NotImplementedError
            (
                lambda s: s.__setitem__("bridging_method", "ZBL"),
                NotImplementedError,
                "bridging_method",
            ),
            # lora not None -> NotImplementedError
            (
                lambda s: s.__setitem__("lora", {"rank": 4}),
                NotImplementedError,
                "lora",
            ),
            # populated dens fitting head -> NotImplementedError
            (
                lambda s: s["atomic_model"].__setitem__("dens_fitting", {"foo": 1}),
                NotImplementedError,
                "dens",
            ),
            # non-energy active_mode -> NotImplementedError
            (
                lambda s: s["atomic_model"].__setitem__("active_mode", "dens"),
                NotImplementedError,
                "active_mode",
            ),
            # missing atomic_model entry -> ValueError
            (
                lambda s: s.pop("atomic_model"),
                ValueError,
                "atomic_model",
            ),
            # unsupported atomic @version (Fix 1 guard) -> ValueError
            (
                lambda s: s["atomic_model"].__setitem__("@version", 4),
                ValueError,
                "not compatible",
            ),
            # unsupported wrapper @version (Fix 1 guard) -> ValueError
            (
                lambda s: s.__setitem__("@version", 2),
                ValueError,
                "not compatible",
            ),
        ],
    )
    def test_guard_branches_raise(
        self, pt_dpa4_model, mutator, exc_type, match
    ) -> None:
        """Each unsupported/invalid pt feature fails fast with a clear error."""
        ser = pt_dpa4_model.serialize()
        mutator(ser)
        with pytest.raises(exc_type, match=match):
            BaseModel.deserialize(ser)

    @pytest.mark.parametrize("version", [2, 3])  # known-compatible atomic versions
    def test_atomic_version_in_range_accepted(self, pt_dpa4_model, version) -> None:
        """Both in-range atomic @versions {2, 3} normalize without raising."""
        atomic = pt_dpa4_model.serialize()["atomic_model"]
        atomic["@version"] = version
        normalized = BaseModel._normalize_pt_sezm_atomic(atomic)
        assert normalized["@version"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
