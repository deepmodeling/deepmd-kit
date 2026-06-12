# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the DPA4/SeZM model-type dispatch in pt_expt ``get_model``."""

import copy
import unittest

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

    def test_descriptor_fitting_type_defaults(self) -> None:
        """Descriptor/fitting type keys default to dpa4/dpa4_ener when absent."""
        raw = _make_raw_model_config()
        self.assertNotIn("type", raw["descriptor"])
        self.assertNotIn("type", raw["fitting_net"])
        model = get_model(raw)
        self.assertIsInstance(model, EnergyModel)

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
        """pt-only SeZM model-level features fail fast with NotImplementedError."""
        cases = {
            "spin": {"use_spin": [True, False], "virtual_scale": [0.3]},
            "bridging_method": "ZBL",
            "lora": {"rank": 4},
            "use_compile": True,
        }
        for key, value in cases.items():
            raw = _make_raw_model_config()
            raw[key] = value
            with self.assertRaises(NotImplementedError, msg=f"key={key}"):
                get_model(raw)

    def test_default_unsupported_values_pass(self) -> None:
        """Normalized defaults (bridging None, lora None, use_compile False) build."""
        model_params = _normalize_model(_make_raw_model_config())
        self.assertEqual(model_params["bridging_method"], "None")
        self.assertIsNone(model_params["lora"])
        self.assertFalse(model_params["use_compile"])
        model = get_model(copy.deepcopy(model_params))
        self.assertIsInstance(model, EnergyModel)


if __name__ == "__main__":
    unittest.main()
