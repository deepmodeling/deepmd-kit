# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fast (no-AOTI) tests for DPA4/SeZM model contracts across PT and PT-expt.

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
from typing import (
    TYPE_CHECKING,
)

import pytest
import torch

from deepmd.pt.model.model import get_model as pt_get_model
from deepmd.pt.optimizer.hybrid_muon import (
    HybridMuonOptimizer,
    adam_route_patterns,
    get_adam_route,
)
from deepmd.pt_expt.model import get_model as pt_expt_get_model
from deepmd.pt_expt.model.dpa4_model import (
    DPA4EnergyModel,
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

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
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
    @pytest.mark.parametrize(
        "get_model", [pt_get_model, pt_expt_get_model], ids=["pt", "pt_expt"]
    )  # backend model factory
    @pytest.mark.parametrize(
        "layout", ["plain", "hybrid", "linear", "zbl", "spin", "default"]
    )  # descriptor and model composition
    def test_adam_routing(
        self, get_model: Callable[[dict], torch.nn.Module], layout: str
    ) -> None:
        """Composite models preserve radial AdamW updates and other Muon updates."""
        config = copy.deepcopy(_DPA4_RAW_CONFIG)
        config["descriptor"].update(n_focus=1, use_env_seed=True)
        hybrid = {
            "type": "standard",
            "type_map": config["type_map"],
            "descriptor": {
                "type": "hybrid",
                "list": [
                    {
                        "type": "se_e2_a",
                        "rcut": 4.0,
                        "rcut_smth": 3.5,
                        "sel": [4, 4],
                        "neuron": [4, 8],
                        "axis_neuron": 2,
                        "seed": 1,
                    },
                    copy.deepcopy(config["descriptor"]),
                ],
            },
            "fitting_net": {"type": "ener", "neuron": [8], "seed": 1},
        }
        if layout == "hybrid":
            config = hybrid
        elif layout == "linear":
            config = {
                "type": "linear_ener",
                "type_map": config["type_map"],
                "models": [config, hybrid],
            }
        elif layout == "zbl":
            config.update(
                bridging_method="ZBL", bridging_r_inner=0.8, bridging_r_outer=1.2
            )
        elif layout == "spin":
            config["type"] = "standard"
            config["fitting_net"] = hybrid["fitting_net"]
            config["spin"] = {"use_spin": [True, False], "virtual_scale": [0.3]}
        elif layout == "default":
            config = hybrid
            config["descriptor"] = config["descriptor"]["list"][0]
        model = get_model(config).to("cpu")
        patterns = adam_route_patterns([model])
        parameters = dict(model.named_parameters())
        radial_inputs = (
            "radial_embedding.net.0.",
            "env_seed_embedding.rbf_proj_layer1.",
        )
        matrices = {
            name: parameter
            for name, parameter in parameters.items()
            if parameter.ndim == 2
            and min(parameter.shape) > 1
            and get_adam_route(name) == "muon"
        }
        expected = {
            name for name in matrices if any(path in name for path in radial_inputs)
        }
        assert len(expected) == {"default": 0, "linear": 4}.get(layout, 2)
        assert {
            name for name in matrices if any(pattern in name for pattern in patterns)
        } == expected
        for pattern in patterns:
            assert any(pattern in name for name in expected), pattern
        control = next(name for name in matrices if name not in expected)
        selected = [(name, matrices[name]) for name in sorted(expected | {control})]
        optimizer = HybridMuonOptimizer(
            [parameter for _, parameter in selected],
            named_parameters=selected,
            adam_patterns=patterns,
            lr=0.01,
            weight_decay=0.1,
            enable_gram=False,
            flash_muon=False,
        )
        for _, parameter in selected:
            parameter.grad = torch.ones_like(parameter)
        references = [parameters[name].detach().clone() for name in sorted(expected)]
        if references:
            adamw = torch.optim.AdamW(
                references, lr=0.01, weight_decay=0.1, betas=(0.9, 0.95)
            )
            for parameter in references:
                parameter.grad = torch.ones_like(parameter)
            adamw.step()
        optimizer.step()
        for name, reference in zip(sorted(expected), references, strict=True):
            torch.testing.assert_close(parameters[name], reference)
            assert "exp_avg" in optimizer.state[parameters[name]]
            assert "momentum_buffer" not in optimizer.state[parameters[name]]
        assert "momentum_buffer" in optimizer.state[parameters[control]]

    @pytest.mark.parametrize("basis_type", ["bessel", "gaussian"])
    def test_single_envelope_normalization_and_roundtrip(self, basis_type: str) -> None:
        """Preserve the integer envelope configuration and its energy/force function."""
        config = copy.deepcopy(_DPA4_RAW_CONFIG)
        config["descriptor"].update(env_exp=5, basis_type=basis_type)
        model_params = _normalize_model(config)
        assert model_params["descriptor"]["env_exp"] == 5
        pt_model = pt_get_model(model_params).to(env.DEVICE).eval()
        generator = torch.Generator(device=env.DEVICE).manual_seed(29)
        with torch.no_grad():
            for parameter in pt_model.parameters():
                parameter.add_(
                    0.01
                    * torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                        generator=generator,
                    )
                )
        expt_model = BaseModel.deserialize(pt_model.serialize()).to(env.DEVICE).eval()
        for model in (pt_model, expt_model):
            descriptor = model.atomic_model.descriptor
            assert descriptor.env_exp == 5
            assert descriptor.radial_basis.exponent == 0
            assert descriptor.radial_basis.envelope is None
        expected = _forward_smoke(pt_model)
        actual = _forward_smoke(expt_model)
        assert expected["force"].abs().max().item() > 1e-10
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                actual[key], expected[key], rtol=1e-9, atol=1e-10
            )

    def test_serialize_layout(self, pt_dpa4_model) -> None:
        """The pt serialize layout matches the interop override's expectations."""
        ser = pt_dpa4_model.serialize()
        # wrapper: recognised model type + @version 1
        assert ser["type"].lower() in ("sezm", "dpa4")
        assert ser["@version"] == 1
        # nested atomic: sezm_atomic @version 3 carrying the pt-only dens state
        atomic = ser["atomic_model"]
        assert atomic["type"] == "sezm_atomic"
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
        normalized = DPA4EnergyModel._normalize_pt_sezm_atomic(atomic)
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
        normalized = DPA4EnergyModel._normalize_pt_sezm_atomic(atomic)
        assert normalized["@version"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
