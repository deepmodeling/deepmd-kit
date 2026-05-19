# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import json
import os
import tempfile
import unittest
import warnings
from unittest import (
    mock,
)

import torch

from deepmd.pt.loss import (
    EnergySpinLoss,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.sezm_spin_model import (
    SeZMSpinModel,
)
from deepmd.pt.train.training import (
    prepare_model_for_loss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.serialization import (
    deserialize_to_file,
)

warnings.filterwarnings(
    # Keep the compile-test warning summary focused on strict-tolerance drift.
    # PyTorch's AOTAutograd cache emits an internal Python 3.14 deprecation
    # warning that is unrelated to SeZM numerical correctness.
    "ignore",
    category=DeprecationWarning,
    module=r"torch\._functorch\._aot_autograd\.autograd_cache",
)


def _assert_close_with_strict_warning(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    strict_atol: float = 1.0e-6,
    strict_rtol: float = 1.0e-6,
    atol: float,
    rtol: float,
    msg: str,
) -> None:
    """Warn on strict compile drift, fail only outside relaxed tolerance."""
    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=strict_atol,
            rtol=strict_rtol,
            msg=msg,
        )
    except AssertionError as err:
        warnings.warn(
            f"{msg} exceeds strict tolerance "
            f"(atol={strict_atol:g}, rtol={strict_rtol:g}) but is checked "
            f"against relaxed tolerance (atol={atol:g}, rtol={rtol:g}): {err}",
            RuntimeWarning,
            stacklevel=2,
        )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=msg)


def reduce_tensor(
    extended_tensor: torch.Tensor,
    mapping: torch.Tensor,
    nloc: int,
) -> torch.Tensor:
    """Reduce an extended tensor back to local atoms."""
    nframes = extended_tensor.shape[0]
    ext_dims = extended_tensor.shape[2:]
    reduced_tensor = torch.zeros(
        [nframes, nloc, *ext_dims],
        dtype=extended_tensor.dtype,
        device=extended_tensor.device,
    )
    mldims = list(mapping.shape)
    mapping = mapping.view(mldims + [1] * len(ext_dims)).expand(
        [-1] * len(mldims) + list(ext_dims)
    )
    return torch.scatter_reduce(
        reduced_tensor,
        1,
        index=mapping,
        src=extended_tensor,
        reduce="sum",
    )


class TestSeZMSpinModel(unittest.TestCase):
    """Test spin support for the SeZM PyTorch model."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)
        self.coord = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ],
            dtype=torch.float64,
            device=self.device,
        )
        self.atype = torch.tensor([[0, 1, 0]], dtype=torch.long, device=self.device)
        self.spin = torch.tensor(
            [
                [
                    [0.20, 0.10, 0.00],
                    [0.30, 0.00, 0.10],
                    [0.10, 0.20, 0.10],
                ]
            ],
            dtype=torch.float64,
            device=self.device,
        )
        self.box = torch.eye(3, dtype=torch.float64, device=self.device).reshape(1, 9)
        self.box = self.box * 6.0

    def _build_model_params(
        self,
        *,
        use_compile: bool = False,
        bridging_method: str = "none",
    ) -> dict:
        """Build a minimal deterministic SeZM spin model config."""
        return {
            "type": "SeZM",
            "type_map": ["O", "H"],
            "spin": {
                "use_spin": [True, False],
                "virtual_scale": 0.2,
            },
            "descriptor": {
                "type": "SeZM",
                "sel": [2, 2],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": True,
                "random_gamma": False,
                "l_schedule": [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 1,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "s2_activation": [False, True],
                "mlp_bias": False,
                "layer_scale": False,
                "use_amp": False,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float32",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [8],
                "activation_function": "silu",
                "precision": "float32",
                "seed": 7,
            },
            "bridging_method": bridging_method,
            "bridging_r_inner": 0.8,
            "bridging_r_outer": 1.2,
            "use_compile": use_compile,
        }

    def test_factory_shapes_and_masks(self) -> None:
        """Factory should build SeZMSpinModel with public real-type metadata."""
        model = get_model(self._build_model_params()).to(self.device)

        self.assertIsInstance(model, SeZMSpinModel)
        self.assertTrue(model.has_spin())
        self.assertEqual(model.get_type_map(), ["O", "H"])
        self.assertEqual(model.get_sel(), [2, 2])

        out = model(self.coord, self.atype, spin=self.spin, box=self.box)

        self.assertEqual(out["energy"].shape, (1, 1))
        self.assertEqual(out["atom_energy"].shape, (1, 3, 1))
        self.assertEqual(out["force"].shape, (1, 3, 3))
        self.assertEqual(out["force_mag"].shape, (1, 3, 3))
        torch.testing.assert_close(
            out["mask_mag"],
            torch.tensor(
                [[[True], [False], [True]]],
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def test_forward_lower_matches_forward(self) -> None:
        """Lower spin interface should match the standard spin forward path."""
        model = get_model(self._build_model_params()).to(self.device)
        out = model(self.coord, self.atype, spin=self.spin, box=self.box)
        extended_coord, extended_atype, mapping, nlist = (
            extend_input_and_build_neighbor_list(
                self.coord,
                self.atype,
                model.get_rcut(),
                model.get_sel(),
                mixed_types=model.mixed_types(),
                box=self.box,
            )
        )
        extended_spin = torch.gather(
            self.spin,
            1,
            mapping.unsqueeze(-1).expand(-1, -1, 3),
        )

        out_lower = model.forward_lower(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping=mapping,
        )

        torch.testing.assert_close(out_lower["energy"], out["energy"])
        torch.testing.assert_close(out_lower["atom_energy"], out["atom_energy"])
        reduced_force = reduce_tensor(out_lower["extended_force"], mapping, nloc=3)
        reduced_force_mag = reduce_tensor(
            out_lower["extended_force_mag"], mapping, nloc=3
        )
        torch.testing.assert_close(reduced_force, out["force"])
        torch.testing.assert_close(reduced_force_mag, out["force_mag"])

    def test_serialize_deserialize_consistency(self) -> None:
        """Serialized SeZMSpinModel should restore the same predictions."""
        model = get_model(self._build_model_params()).to(self.device)
        restored = SeZMSpinModel.deserialize(model.serialize()).to(self.device)

        out = model(self.coord, self.atype, spin=self.spin, box=self.box)
        restored_out = restored(self.coord, self.atype, spin=self.spin, box=self.box)

        self.assertEqual(restored.get_type_map(), ["O", "H"])
        self.assertEqual(restored.get_sel(), [2, 2])
        for key, value in out.items():
            torch.testing.assert_close(restored_out[key], value)

    def test_deserialize_to_file_uses_spin_model(self) -> None:
        """File deserialization should route sezm_spin through SeZMSpinModel."""
        model = get_model(self._build_model_params()).to(self.device)
        data = {
            "model": model.serialize(),
            "model_def_script": self._build_model_params(),
            "@variables": {},
        }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "deepmd.pt.utils.serialization.torch.jit.script",
                side_effect=lambda model: model,
            ),
            mock.patch("deepmd.pt.utils.serialization.torch.jit.save") as save_mock,
        ):
            deserialize_to_file(f"{tmpdir}/model.pth", data)

        saved_model = save_mock.call_args.args[0]
        self.assertIsInstance(saved_model, SeZMSpinModel)
        self.assertEqual(
            saved_model.model_def_script,
            json.dumps(data["model_def_script"]),
        )

    def test_energy_spin_loss_consumes_force_mag(self) -> None:
        """EnergySpinLoss should consume force and magnetic-force predictions."""
        model = get_model(self._build_model_params()).to(self.device)
        loss = EnergySpinLoss(
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_fr=1.0,
            limit_pref_fr=1.0,
            start_pref_fm=1.0,
            limit_pref_fm=1.0,
        )
        input_dict = {
            "coord": self.coord,
            "atype": self.atype,
            "spin": self.spin,
            "box": self.box,
        }
        label = {
            "energy": torch.zeros((1, 1), dtype=torch.float64, device=self.device),
            "force": torch.zeros((1, 3, 3), dtype=torch.float64, device=self.device),
            "force_mag": torch.zeros(
                (1, 3, 3), dtype=torch.float64, device=self.device
            ),
            "find_energy": torch.tensor(1.0, device=self.device),
            "find_force": torch.tensor(1.0, device=self.device),
            "find_force_mag": torch.tensor(1.0, device=self.device),
        }

        model_pred, loss_value, more_loss = loss(
            input_dict,
            model,
            label,
            natoms=3,
            learning_rate=1.0,
        )

        self.assertIn("force_mag", model_pred)
        self.assertIn("rmse_fm", more_loss)
        self.assertTrue(torch.isfinite(loss_value))

    def test_dens_mode_is_rejected(self) -> None:
        """SeZM spin permanently rejects the dens path."""
        model = get_model(self._build_model_params()).to(self.device)

        with self.assertRaises(NotImplementedError):
            prepare_model_for_loss(model, {"type": "dens"})

    def test_bridging_masks_virtual_pairs(self) -> None:
        """ZBL bridging should ignore virtual spin types without indexing them."""
        model = get_model(self._build_model_params(bridging_method="ZBL")).to(
            self.device
        )
        self.assertIsNotNone(model.inter_potential)

        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]],
            dtype=torch.float64,
            device=self.device,
        )
        atype_with_virtual = torch.tensor(
            [[0, 1, 2]], dtype=torch.long, device=self.device
        )
        nlist_real_and_virtual = torch.tensor(
            [[[1, 2], [0, 2], [0, 1]]], dtype=torch.long, device=self.device
        )
        nlist_real_only = torch.tensor(
            [[[1, -1], [0, -1], [-1, -1]]], dtype=torch.long, device=self.device
        )

        energy_with_virtual = model.inter_potential(
            coord,
            atype_with_virtual,
            nlist_real_and_virtual,
            nloc=3,
            real_type_count=2,
        )
        energy_real_only = model.inter_potential(
            coord,
            atype_with_virtual,
            nlist_real_only,
            nloc=3,
            real_type_count=2,
        )

        torch.testing.assert_close(energy_with_virtual, energy_real_only)

    def test_compile_matches_eager(self) -> None:
        """Compiled SeZM spin path should match eager predictions."""
        eager = get_model(self._build_model_params(use_compile=False)).to(self.device)
        with mock.patch.dict(os.environ, {"DP_COMPILE_INFER": "1"}, clear=False):
            compiled = get_model(self._build_model_params(use_compile=True)).to(
                self.device
            )
        compiled.load_state_dict(copy.deepcopy(eager.state_dict()))
        eager.eval()
        compiled.eval()

        out_eager = eager(self.coord, self.atype, spin=self.spin, box=self.box)
        out_compiled = compiled(self.coord, self.atype, spin=self.spin, box=self.box)

        self.assertIn((False, False, True), compiled.compiled_core_compute_cache)
        _assert_close_with_strict_warning(
            out_compiled["energy"],
            out_eager["energy"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="spin compile energy mismatch",
        )
        _assert_close_with_strict_warning(
            out_compiled["force"],
            out_eager["force"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="spin compile force mismatch",
        )
        _assert_close_with_strict_warning(
            out_compiled["force_mag"],
            out_eager["force_mag"],
            atol=1.0e-6,
            rtol=1.0e-6,
            msg="spin compile magnetic force mismatch",
        )


if __name__ == "__main__":
    unittest.main()
