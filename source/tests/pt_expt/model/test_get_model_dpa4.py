# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the DPA4/SeZM model-type dispatch in pt_expt ``get_model``."""

import copy
import unittest

import pytest
import torch

from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
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


def _make_raw_model_config(**model_overrides) -> dict:
    """Minimal (un-normalized) dpa4 model config; small dims for speed."""
    model = {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
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
            "precision": "float64",
            "seed": 1,
        },
    }
    model.update(model_overrides)
    return model


def _normalize_model(model: dict) -> dict:
    config = {
        "model": model,
        "training": {"training_data": {"systems": ["dummy"]}, "numb_steps": 1},
        "loss": {"type": "ener"},
        "learning_rate": {"type": "exp", "start_lr": 1e-3},
    }
    config = update_deepmd_input(config, warning=False)
    config = normalize(config)
    return config["model"]


class TestGetModelDPA4(unittest.TestCase):
    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_get_model_normalized_config(self) -> None:
        """Normalized argcheck config (type key present) builds an EnergyModel."""
        model_params = _normalize_model(_make_raw_model_config())
        self.assertEqual(model_params["type"], "dpa4")
        model = get_model(model_params).to(self.device)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(model, EnergyModel)
        self.assertEqual(model.get_dim_fparam(), 0)
        self.assertEqual(model.get_type_map(), ["O", "H"])
        nparams = sum(p.numel() for p in model.parameters())
        self.assertGreater(nparams, 0)
        # forward smoke
        generator = torch.Generator(device=self.device).manual_seed(1)
        cell = 5.0 * torch.eye(3, dtype=torch.float64, device=self.device)
        coord = (
            torch.rand(
                [1, 5, 3],
                dtype=torch.float64,
                device=self.device,
                generator=generator,
            )
            @ cell
        ).requires_grad_(True)
        atype = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int64, device=self.device)
        ret = model(coord, atype, cell.reshape(1, 9))
        self.assertEqual(ret["energy"].shape, (1, 1))
        self.assertEqual(ret["force"].shape, (1, 5, 3))

    def test_get_model_type_aliases(self) -> None:
        """All model-type aliases route to the SeZM path."""
        for alias in ("dpa4", "DPA4", "sezm", "SeZM"):
            model_params = _make_raw_model_config(type=alias)
            model = get_model(model_params)
            self.assertIsInstance(model, EnergyModel, msg=f"alias={alias}")

    def test_serialize_deserialize_alias(self) -> None:
        """Round-trip locks the sezm_ener/dpa4_ener -> EnergyModel alias.

        Fast (no-AOTI) regression guard: the model-type alias is otherwise
        only exercised by the CI-skipped AOTI freeze test.
        """
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        model = get_model(_make_raw_model_config()).to(self.device)
        data = model.serialize()
        # serialized layout: top-level "standard", fitting "sezm_ener"
        self.assertEqual(data["type"], "standard")
        self.assertEqual(data["fitting"]["type"], "sezm_ener")
        self.assertEqual(
            model.atomic_model.fitting_net.serialize()["type"], "sezm_ener"
        )
        # the alias resolution must not raise and must rebuild an EnergyModel
        model2 = BaseModel.deserialize(model.serialize())
        self.assertIsInstance(model2, EnergyModel)
        model2 = model2.to(self.device)
        # forward-smoke the deserialized model to prove the round-trip works
        generator = torch.Generator(device=self.device).manual_seed(1)
        cell = 5.0 * torch.eye(3, dtype=torch.float64, device=self.device)
        coord = (
            torch.rand(
                [1, 5, 3],
                dtype=torch.float64,
                device=self.device,
                generator=generator,
            )
            @ cell
        ).requires_grad_(True)
        atype = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int64, device=self.device)
        ret0 = model(coord, atype, cell.reshape(1, 9))
        ret = model2(coord, atype, cell.reshape(1, 9))
        self.assertEqual(ret["energy"].shape, ret0["energy"].shape)

    def test_descriptor_fitting_type_defaults(self) -> None:
        """Descriptor/fitting type keys default to dpa4/dpa4_ener when absent."""
        raw = _make_raw_model_config()
        self.assertNotIn("type", raw["descriptor"])
        self.assertNotIn("type", raw["fitting_net"])
        model = get_model(raw)
        self.assertIsInstance(model, EnergyModel)

    def test_explicit_matching_component_types_ok(self) -> None:
        """Explicit dpa4/sezm descriptor and fitting types are accepted."""
        for desc_type, fit_type in (("dpa4", "dpa4_ener"), ("sezm", "sezm_ener")):
            raw = _make_raw_model_config()
            raw["descriptor"]["type"] = desc_type
            raw["fitting_net"]["type"] = fit_type
            model = get_model(raw)
            self.assertIsInstance(model, EnergyModel, msg=f"{desc_type}/{fit_type}")

    def test_explicit_mismatching_descriptor_type_raises(self) -> None:
        raw = _make_raw_model_config()
        raw["descriptor"]["type"] = "se_e2_a"
        with self.assertRaisesRegex(ValueError, "requires a DPA4/SeZM descriptor"):
            get_model(raw)

    def test_explicit_mismatching_fitting_type_raises(self) -> None:
        raw = _make_raw_model_config()
        raw["fitting_net"]["type"] = "ener"
        with self.assertRaisesRegex(ValueError, "energy fitting net"):
            get_model(raw)

    def test_pair_exclude_types_from_descriptor(self) -> None:
        """descriptor.exclude_types propagates when pair_exclude_types absent."""
        raw = _make_raw_model_config()
        raw["descriptor"]["exclude_types"] = [[0, 1]]
        model = get_model(raw)
        self.assertEqual(model.atomic_model.pair_exclude_types, [[0, 1]])

    def test_pair_exclude_types_consistent(self) -> None:
        """Matching pair_exclude_types and descriptor.exclude_types are accepted."""
        raw = _make_raw_model_config()
        raw["descriptor"]["exclude_types"] = [[0, 1]]
        raw["pair_exclude_types"] = [[0, 1]]
        model = get_model(raw)
        self.assertEqual(model.atomic_model.pair_exclude_types, [[0, 1]])

    def test_pair_exclude_types_mismatch_raises(self) -> None:
        raw = _make_raw_model_config()
        raw["descriptor"]["exclude_types"] = [[0, 1]]
        raw["pair_exclude_types"] = [[0, 0]]
        with self.assertRaisesRegex(ValueError, "must match"):
            get_model(raw)

    def test_unsupported_keys_raise(self) -> None:
        """pt-only SeZM model-level features fail fast with NotImplementedError.

        ``bridging_method`` is no longer in this list: it is supported as an
        atomic-model composition (see ``test_zbl_bridging.py``).
        """
        cases = {
            "spin": ({"use_spin": [True, False], "virtual_scale": [0.3]}, "Spin DPA4"),
            "lora": ({"rank": 4}, "`lora` is not supported"),
            "use_compile": (True, "`use_compile` is not supported"),
            "preset_out_bias": (
                {"energy": [None, 1.0]},
                "`preset_out_bias` is not supported",
            ),
        }
        for key, (value, msg_regex) in cases.items():
            raw = _make_raw_model_config()
            raw[key] = value
            with self.assertRaisesRegex(NotImplementedError, msg_regex):
                get_model(raw)

    def test_native_spin_capability_gate_standard_config(self) -> None:
        """The generic ``supports_native_spin()`` gate rejects a dense descriptor.

        Uses a complete ``type="standard"`` se_e2_a energy config so the
        DESCRIPTOR-AGNOSTIC capability gate is what fires -- not the
        dpa4-typed builder's descriptor contract (pinned separately below).
        """
        raw = {
            "type": "standard",
            "type_map": ["Ni", "O"],
            "descriptor": {
                "type": "se_e2_a",
                "rcut": 4.0,
                "rcut_smth": 3.5,
                "sel": [8, 8],
            },
            "fitting_net": {"type": "ener", "neuron": [8, 8]},
            "spin": {"use_spin": [True, False], "scheme": "native"},
        }
        with self.assertRaisesRegex(NotImplementedError, "native spin"):
            get_model(raw)

    def test_native_spin_non_dpa4_descriptor_raises(self) -> None:
        """A dpa4-typed config rejects a foreign descriptor (family contract).

        The dpa4-typed builder pins its descriptor/fitting contract before
        the generic capability gate is reached, so the mismatch surfaces as
        the family builder's ``ValueError``.
        """
        raw = _make_raw_model_config()
        raw["descriptor"] = {"type": "se_e2_a"}
        raw["spin"] = {"use_spin": [True, False], "scheme": "native"}
        with self.assertRaisesRegex(ValueError, "DPA4/SeZM descriptor"):
            get_model(raw)

    def test_native_spin_add_chg_spin_ebd_combined_builds(self) -> None:
        """Native-scheme spin combined with charge-spin FiLM is SUPPORTED.

        (Review 3638047227 lifted the old rejection; the combined model's
        behavior is pinned in ``test_dpa4_native_spin.py``.)
        """
        raw = _make_raw_model_config()
        raw["descriptor"]["add_chg_spin_ebd"] = True
        raw["spin"] = {"use_spin": [True, False], "scheme": "native"}
        model = get_model(raw)
        self.assertTrue(model.has_chg_spin_ebd())
        self.assertTrue(model.has_spin())

    def test_default_unsupported_values_pass(self) -> None:
        """Normalized defaults (bridging None, lora None, use_compile False) build."""
        model_params = _normalize_model(_make_raw_model_config())
        self.assertEqual(model_params["bridging_method"], "None")
        self.assertIsNone(model_params["lora"])
        self.assertFalse(model_params["use_compile"])
        self.assertIsNone(model_params.get("preset_out_bias"))
        model = get_model(copy.deepcopy(model_params))
        self.assertIsInstance(model, EnergyModel)


# === TF32 matmul precision ===
# Same policy as pt: training follows ``enable_tf32`` (default True), eval
# follows ``DP_TF32_INFER``.  Like pt, the knob is DPA4/SeZM-scoped.


@pytest.mark.parametrize(
    "enable_tf32",
    [
        True,  # the argcheck default; training must select TF32 ("high")
        False,  # opt-out; training must stay at full fp32
    ],
)
def test_enable_tf32_is_stored(enable_tf32) -> None:
    """The config knob reaches the model instead of being warned away."""
    model = get_model(_make_raw_model_config(enable_tf32=enable_tf32))
    assert model.enable_tf32 is enable_tf32


def test_enable_tf32_defaults_true() -> None:
    """An absent key follows pt's ``default=True`` (argcheck.py `enable_tf32`)."""
    raw = _make_raw_model_config()
    assert "enable_tf32" not in raw
    assert get_model(raw).enable_tf32 is True


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, "highest"),  # unset -> pt's "0" default, full fp32
        ("0", "highest"),
        ("1", "high"),
        ("2", "medium"),
    ],
)
def test_tf32_infer_precision_from_env(env_value, expected, monkeypatch) -> None:
    """Eval precision follows ``DP_TF32_INFER``, as in the pt backend."""
    if env_value is None:
        monkeypatch.delenv("DP_TF32_INFER", raising=False)
    else:
        monkeypatch.setenv("DP_TF32_INFER", env_value)
    assert get_model(_make_raw_model_config()).tf32_infer_precision == expected


def test_tf32_infer_precision_rejects_garbage(monkeypatch) -> None:
    """An unusable ``DP_TF32_INFER`` fails fast rather than silently defaulting."""
    monkeypatch.setenv("DP_TF32_INFER", "yes")
    with pytest.raises(ValueError, match="DP_TF32_INFER"):
        get_model(_make_raw_model_config())


@pytest.mark.parametrize(
    ("enable_tf32", "training", "expected"),
    [
        (True, True, "high"),  # the only combination that selects TF32
        (False, True, "highest"),  # opt-out keeps training at full fp32
        (True, False, "highest"),  # eval ignores enable_tf32 (uses DP_TF32_INFER)
        (False, False, "highest"),
    ],
)
def test_tf32_precision_ctx_selects_and_restores(
    enable_tf32, training, expected, monkeypatch
) -> None:
    """The context picks the right precision per mode, then restores it.

    ``set_float32_matmul_precision`` is a process global, so a leaked setting
    would change every later matmul; restoring matters as much as selecting.
    """
    if not torch.cuda.is_available():
        pytest.skip("tf32_precision_ctx is a no-op without CUDA")
    monkeypatch.delenv("DP_TF32_INFER", raising=False)
    model = get_model(_make_raw_model_config(enable_tf32=enable_tf32))
    model.train(training)

    torch.set_float32_matmul_precision("highest")
    try:
        with model.tf32_precision_ctx():
            assert torch.get_float32_matmul_precision() == expected
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision("highest")


def test_non_sezm_model_keeps_full_precision() -> None:
    """The knob is DPA4/SeZM-scoped: other pt_expt models are untouched.

    pt wires ``enable_tf32`` only in its sezm builders, so a plain se_e2_a
    model keeps the class defaults: full fp32 in both train and eval.
    """
    model = get_model(
        {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [4, 4],
                "rcut": 4.0,
                "rcut_smth": 3.5,
                "seed": 1,
            },
            "fitting_net": {"seed": 1},
        }
    )
    assert model.enable_tf32 is False
    assert model.tf32_infer_precision == "highest"


class TestNativeSpinErrorTranslation(unittest.TestCase):
    """Only the unexpected-``use_spin`` TypeError becomes the capability error."""

    def test_unrelated_construction_error_propagates(self) -> None:
        # A bogus fitting kwarg must surface as the REAL TypeError, not be
        # masked as a native-spin capability failure (review 3644847676).
        raw = _make_raw_model_config()
        raw["spin"] = {"use_spin": [True, False], "scheme": "native"}
        raw["fitting_net"]["bogus_option"] = 1
        with self.assertRaisesRegex(TypeError, "bogus_option"):
            get_model(raw)


if __name__ == "__main__":
    unittest.main()
