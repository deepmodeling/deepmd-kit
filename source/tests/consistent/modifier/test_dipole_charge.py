# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cross-backend consistency tests for the dpmodel dipole-charge modifier."""

import unittest

import numpy as np

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.dpmodel.modifier.dipole_charge import (
    ewald_reciprocal_energy,
    extend_dplr_system,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT_EXPT,
    INSTALLED_TF2,
)

if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict

if INSTALLED_PT_EXPT:
    import torch

if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.model.model import BaseModel as PTExptBaseModel
    from deepmd.pt_expt.modifier.dipole_charge import (
        DipoleChargeModifier as PTExptModifier,
    )
if INSTALLED_JAX:
    from deepmd.jax.model.base_model import BaseModel as JAXBaseModel
    from deepmd.jax.modifier.dipole_charge import DipoleChargeModifier as JAXModifier
if INSTALLED_TF2:
    from deepmd.tf2.model.base_model import BaseModel as TF2BaseModel
    from deepmd.tf2.modifier.dipole_charge import DipoleChargeModifier as TF2Modifier


class TestDipoleChargeModifierConsistency(unittest.TestCase):
    """Ensure dpmodel-driven adapters produce the same dipole-charge correction."""

    @classmethod
    def setUpClass(cls) -> None:
        model_config = {
            "type_map": ["O", "H"],
            "atom_exclude_types": [1],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [8, 8],
                "rcut_smth": 0.5,
                "rcut": 4.0,
                "neuron": [4, 8],
                "axis_neuron": 4,
                "precision": "float64",
                "seed": 2026,
            },
            "fitting_net": {
                "type": "dipole",
                "neuron": [8, 8],
                "precision": "float64",
                "seed": 2027,
            },
        }
        cls.serialized_model = get_model_dp(model_config).serialize()
        cls.coord = np.asarray(
            [
                [
                    [1.0, 1.0, 1.0],
                    [1.8, 1.0, 1.0],
                    [1.0, 1.8, 1.0],
                    [6.0, 6.0, 6.0],
                    [6.8, 6.0, 6.0],
                    [6.0, 6.8, 6.0],
                ]
            ],
            dtype=np.float64,
        )
        cls.atype = np.asarray([[0, 1, 1, 0, 1, 1]], dtype=np.int32)
        cls.box = np.asarray([np.eye(3) * 10.0], dtype=np.float64)
        cls.kwargs = {
            "model_name": "embedded.dp",
            "model_charge_map": [-8.0],
            "sys_charge_map": [6.0, 1.0],
            "ewald_h": 2.0,
            "ewald_beta": 1.0,
        }

    def _evaluate_backends(self) -> dict[str, dict[str, np.ndarray]]:
        results = {}
        if INSTALLED_PT_EXPT:
            model = PTExptBaseModel.deserialize(self.serialized_model).double()
            modifier = PTExptModifier(**self.kwargs, dipole_model=model).double().eval()
            result = modifier(
                torch.tensor(self.coord, dtype=torch.float64, device="cpu"),
                torch.tensor(self.atype, dtype=torch.int64, device="cpu"),
                torch.tensor(self.box, dtype=torch.float64, device="cpu"),
            )
            results["pt_expt"] = {
                key: value.detach().cpu().numpy() for key, value in result.items()
            }
        if INSTALLED_JAX:
            model = JAXBaseModel.deserialize(self.serialized_model)
            result = JAXModifier(**self.kwargs, dipole_model=model)(
                self.coord, self.atype, self.box
            )
            results["jax"] = {key: np.asarray(value) for key, value in result.items()}
        if INSTALLED_TF2:
            model = TF2BaseModel.deserialize(self.serialized_model)
            result = TF2Modifier(**self.kwargs, dipole_model=model)(
                self.coord, self.atype, self.box
            )
            results["tf2"] = {key: np.asarray(value) for key, value in result.items()}
        return results

    def test_energy_force_and_virial(self) -> None:
        results = self._evaluate_backends()
        if len(results) < 2:
            self.skipTest("At least two dpmodel-driven backends are required")
        reference_name, reference = next(iter(results.items()))
        for backend_name, result in results.items():
            for key in ("energy", "force", "virial"):
                np.testing.assert_allclose(
                    result[key],
                    reference[key],
                    rtol=1e-9,
                    atol=1e-9,
                    err_msg=f"{backend_name} does not match {reference_name} for {key}",
                )

    @unittest.skipUnless(INSTALLED_ARRAY_API_STRICT, "array_api_strict is required")
    def test_array_api_strict_core(self) -> None:
        """Ensure the shared numerical core uses only standard array operations."""
        xp = array_api_strict
        coord = xp.asarray(self.coord, dtype=xp.float64)
        atype = xp.asarray(self.atype, dtype=xp.int64)
        box = xp.asarray(self.box, dtype=xp.float64)
        all_coord, all_charge = extend_dplr_system(
            coord,
            atype,
            xp.zeros_like(coord),
            [0],
            [-8.0],
            [6.0, 1.0],
        )
        energy = ewald_reciprocal_energy(all_coord, all_charge, box, ((6, 6, 6),), 1.0)
        self.assertEqual(energy.shape, (1, 1))
        self.assertTrue(np.isfinite(np.asarray(energy)).all())
