# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest
from unittest.mock import (
    patch,
)

import numpy as np
import torch

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.descriptor import DescrptDPA1 as DPDescrptDPA1
from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.dpmodel.model.make_model import (
    make_model,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.pt_expt.model import (
    LinearEnergyModel,
)
from deepmd.pt_expt.model.dp_linear_model import (
    LinearEnergyModel as LinearEnergyModelDirect,
)
from deepmd.pt_expt.model.get_model import (
    get_linear_model,
    get_standard_model,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)
from ..export_helpers import (
    model_forward_lower_export_round_trip,
)


class TestLinearModel(unittest.TestCase):
    def setUp(self) -> None:
        self.device = env.DEVICE
        self.natoms = 5
        self.rcut = 4.0
        self.rcut_smth = 0.5
        self.sel = 20
        self.nt = 2
        self.type_map = ["foo", "bar"]

        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=self.device)
        self.cell = cell.unsqueeze(0)
        coord = torch.rand(
            [self.natoms, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0).to(self.device)
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1]], dtype=torch.int64, device=self.device
        )

    def _make_dp_atomic_model(self, seed: int) -> DPAtomicModel:
        """Build a dpmodel DPAtomicModel with DPA1 descriptor (mixed type)."""
        ds = DPDescrptDPA1(
            rcut_smth=self.rcut_smth,
            rcut=self.rcut,
            sel=self.sel,
            ntypes=self.nt,
            neuron=[3, 6],
            axis_neuron=2,
            attn=4,
            attn_layer=2,
            attn_dotr=True,
            attn_mask=False,
            activation_function="tanh",
            set_davg_zero=True,
            type_one_side=True,
            seed=seed,
        )
        ft = DPInvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=seed,
        )
        return DPAtomicModel(ds, ft, type_map=self.type_map)

    def _make_dp_linear_model(self) -> LinearEnergyAtomicModel:
        """Build a dpmodel LinearEnergyAtomicModel with two sub-models."""
        model1 = self._make_dp_atomic_model(seed=GLOBAL_SEED)
        model2 = self._make_dp_atomic_model(seed=GLOBAL_SEED + 1)
        return LinearEnergyAtomicModel(
            models=[model1, model2],
            type_map=self.type_map,
        )

    def _prepare_lower_inputs(self):
        """Build extended coords, atype, nlist, mapping as torch tensors."""
        coord_np = self.coord.detach().cpu().numpy()
        atype_np = self.atype.detach().cpu().numpy()
        cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
        coord_normalized = normalize_coord(
            coord_np.reshape(1, self.natoms, 3),
            cell_np.reshape(1, 3, 3),
        )
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype_np, cell_np, self.rcut
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            self.natoms,
            self.rcut,
            [self.sel],
            distinguish_types=False,
        )
        extended_coord = extended_coord.reshape(1, -1, 3)
        return (
            torch.tensor(extended_coord, dtype=torch.float64, device=self.device),
            torch.tensor(extended_atype, dtype=torch.int64, device=self.device),
            torch.tensor(nlist, dtype=torch.int64, device=self.device),
            torch.tensor(mapping, dtype=torch.int64, device=self.device),
        )

    def test_linear_model_consistency(self) -> None:
        """Create a LinearEnergyModel, run forward() and forward_lower(),
        verify outputs have correct keys and shapes.
        """
        md_dp = self._make_dp_linear_model()
        md_pt = LinearEnergyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()

        # Test forward()
        coord = self.coord.clone().requires_grad_(True)
        ret = md_pt(coord, self.atype, self.cell.reshape(1, 9))

        self.assertIn("energy", ret)
        self.assertIn("atom_energy", ret)
        self.assertIn("force", ret)
        self.assertIn("virial", ret)

        self.assertEqual(ret["energy"].shape, (1, 1))
        self.assertEqual(ret["atom_energy"].shape, (1, self.natoms, 1))
        self.assertEqual(ret["force"].shape, (1, self.natoms, 3))
        self.assertEqual(ret["virial"].shape, (1, 9))

        # Test forward_lower()
        ext_coord, ext_atype, nlist_t, mapping_t = self._prepare_lower_inputs()
        ret_lower = md_pt.forward_lower(
            ext_coord.requires_grad_(True),
            ext_atype,
            nlist_t,
            mapping_t,
        )

        self.assertIn("energy", ret_lower)
        self.assertIn("atom_energy", ret_lower)
        self.assertIn("extended_force", ret_lower)
        self.assertIn("virial", ret_lower)

        nall = ext_coord.shape[1]
        self.assertEqual(ret_lower["energy"].shape, (1, 1))
        self.assertEqual(ret_lower["atom_energy"].shape, (1, self.natoms, 1))
        self.assertEqual(ret_lower["extended_force"].shape, (1, nall, 3))
        self.assertEqual(ret_lower["virial"].shape, (1, 9))

    def test_linear_model_serialize(self) -> None:
        """Create a LinearEnergyModel, serialize, deserialize, verify
        outputs match.
        """
        md_dp = self._make_dp_linear_model()
        md_pt0 = LinearEnergyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt0.eval()

        # Serialize and deserialize
        md_pt1 = LinearEnergyModel.deserialize(md_pt0.serialize()).to(self.device)
        md_pt1.eval()

        coord = self.coord.clone().requires_grad_(True)
        ret0 = md_pt0(coord, self.atype, self.cell.reshape(1, 9))

        coord = self.coord.clone().requires_grad_(True)
        ret1 = md_pt1(coord, self.atype, self.cell.reshape(1, 9))

        np.testing.assert_allclose(
            ret0["energy"].detach().cpu().numpy(),
            ret1["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="energy mismatch after serialize/deserialize",
        )
        np.testing.assert_allclose(
            ret0["atom_energy"].detach().cpu().numpy(),
            ret1["atom_energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="atom_energy mismatch after serialize/deserialize",
        )
        np.testing.assert_allclose(
            ret0["force"].detach().cpu().numpy(),
            ret1["force"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="force mismatch after serialize/deserialize",
        )
        np.testing.assert_allclose(
            ret0["virial"].detach().cpu().numpy(),
            ret1["virial"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="virial mismatch after serialize/deserialize",
        )

    def test_linear_model_dpmodel_consistency(self) -> None:
        """Compare pt_expt LinearEnergyModel output with dpmodel
        LinearEnergyAtomicModel output (same weights) to verify
        cross-backend consistency.
        """
        md_dp_atomic = self._make_dp_linear_model()

        # Build pt_expt version from the same serialized data
        md_pt = LinearEnergyModel.deserialize(md_dp_atomic.serialize()).to(self.device)
        md_pt.eval()

        # Use forward_lower for both backends to compare
        coord_np = self.coord.detach().cpu().numpy()
        atype_np = self.atype.detach().cpu().numpy()
        cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
        coord_normalized = normalize_coord(
            coord_np.reshape(1, self.natoms, 3),
            cell_np.reshape(1, 3, 3),
        )
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype_np, cell_np, self.rcut
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            self.natoms,
            self.rcut,
            [self.sel],
            distinguish_types=False,
        )
        extended_coord = extended_coord.reshape(1, -1, 3)

        # dpmodel forward_lower via make_model wrapper
        DPLinearModel = make_model(LinearEnergyAtomicModel)
        md_dp = DPLinearModel.deserialize(md_dp_atomic.serialize())
        ret_dp = md_dp.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
        )

        # pt_expt forward_lower
        ext_coord = torch.tensor(
            extended_coord, dtype=torch.float64, device=self.device
        )
        ext_atype = torch.tensor(extended_atype, dtype=torch.int64, device=self.device)
        nlist_t = torch.tensor(nlist, dtype=torch.int64, device=self.device)
        mapping_t = torch.tensor(mapping, dtype=torch.int64, device=self.device)
        ret_pt = md_pt.forward_lower(
            ext_coord.requires_grad_(True),
            ext_atype,
            nlist_t,
            mapping_t,
        )

        np.testing.assert_allclose(
            ret_dp["energy_redu"],
            ret_pt["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="energy mismatch between dpmodel and pt_expt",
        )
        np.testing.assert_allclose(
            ret_dp["energy"],
            ret_pt["atom_energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="atom_energy mismatch between dpmodel and pt_expt",
        )

    def test_forward_lower_exportable(self) -> None:
        """Test that LinearEnergyModel.forward_lower_exportable returns
        an exportable module whose outputs match eager execution.
        """
        md_dp = self._make_dp_linear_model()
        md_pt = LinearEnergyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()

        ext_coord, ext_atype, nlist_t, mapping_t = self._prepare_lower_inputs()
        fparam = None
        aparam = None

        model_forward_lower_export_round_trip(
            md_pt,
            ext_coord,
            ext_atype,
            nlist_t,
            mapping_t,
            fparam,
            aparam,
            output_keys=("atom_energy", "energy"),
        )


_sub_model_1 = {
    "descriptor": {
        "type": "se_atten",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [3, 6],
        "axis_neuron": 2,
        "attn": 8,
        "attn_layer": 2,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "scaling_factor": 1.0,
        "normalize": False,
        "temperature": 1.0,
        "set_davg_zero": True,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [5, 5],
        "resnet_dt": True,
        "seed": 1,
    },
}
_sub_model_2 = copy.deepcopy(_sub_model_1)
_sub_model_2["descriptor"]["seed"] = 2
_sub_model_2["fitting_net"]["seed"] = 2

_type_map = ["O", "H"]


class TestLinearEnerWeights(unittest.TestCase):
    """Test that weights parameter affects energy, force, and virial."""

    def setUp(self) -> None:
        self.device = env.DEVICE

        # Build individual standard models for reference
        std_data_1 = copy.deepcopy(_sub_model_1)
        std_data_1["type_map"] = copy.deepcopy(_type_map)
        std_data_2 = copy.deepcopy(_sub_model_2)
        std_data_2["type_map"] = copy.deepcopy(_type_map)
        self.std_model_1 = get_standard_model(std_data_1)
        self.std_model_2 = get_standard_model(std_data_2)

        # Build linear models with different weights
        def _make_linear(weights):
            data = {
                "type_map": copy.deepcopy(_type_map),
                "models": [copy.deepcopy(_sub_model_1), copy.deepcopy(_sub_model_2)],
                "weights": weights,
            }
            return get_linear_model(data)

        self.model_mean = _make_linear("mean")
        self.model_sum = _make_linear("sum")
        self.model_custom = _make_linear([0.3, 0.7])

        # Sync sub-model weights so linear models use the same params as std models
        for linear_model in [self.model_mean, self.model_sum, self.model_custom]:
            linear_model.atomic_model.models[0].load_state_dict(
                self.std_model_1.atomic_model.state_dict()
            )
            linear_model.atomic_model.models[1].load_state_dict(
                self.std_model_2.atomic_model.state_dict()
            )

        # Test inputs
        generator = torch.Generator(device=self.device).manual_seed(20)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(
            3, dtype=torch.float64, device=self.device
        )
        self.cell = cell.unsqueeze(0)
        natoms = 6
        coord = torch.rand(
            [natoms, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0)
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1]], dtype=torch.int64, device=self.device
        )
        self.box = self.cell.reshape(1, 9)

    def _eval(self, model):
        coord = self.coord.clone().detach().requires_grad_(True)
        ret = model(
            coord,
            self.atype,
            box=self.box,
        )
        return {k: v.detach().cpu().numpy() for k, v in ret.items()}

    def test_mean_weights(self) -> None:
        ret1 = self._eval(self.std_model_1)
        ret2 = self._eval(self.std_model_2)
        ret_mean = self._eval(self.model_mean)
        for key in ["energy", "force", "virial"]:
            expected = 0.5 * ret1[key] + 0.5 * ret2[key]
            np.testing.assert_allclose(ret_mean[key], expected, atol=1e-10)

    def test_sum_weights(self) -> None:
        ret1 = self._eval(self.std_model_1)
        ret2 = self._eval(self.std_model_2)
        ret_sum = self._eval(self.model_sum)
        for key in ["energy", "force", "virial"]:
            expected = ret1[key] + ret2[key]
            np.testing.assert_allclose(ret_sum[key], expected, atol=1e-10)

    def test_custom_weights(self) -> None:
        ret1 = self._eval(self.std_model_1)
        ret2 = self._eval(self.std_model_2)
        ret_custom = self._eval(self.model_custom)
        for key in ["energy", "force", "virial"]:
            expected = 0.3 * ret1[key] + 0.7 * ret2[key]
            np.testing.assert_allclose(ret_custom[key], expected, atol=1e-10)


class TestLinearUpdateSel(unittest.TestCase):
    """Test that update_sel writes updated sub-model configs back."""

    @patch("deepmd.pt_expt.model.dp_linear_model.DPModelCommon.update_sel")
    def test_updated_sel_written_back(self, mock_update_sel) -> None:
        """Verify that update_sel returns configs with updated sel values."""

        def side_effect(train_data, type_map, sub_jdata):
            updated = copy.deepcopy(sub_jdata)
            updated["descriptor"]["sel"] = 99
            return updated, 0.5

        mock_update_sel.side_effect = side_effect

        local_jdata = {
            "type_map": ["O", "H"],
            "models": [
                {
                    "descriptor": {"type": "se_atten", "sel": 10, "rcut": 4.0},
                    "fitting_net": {"neuron": [5, 5]},
                },
                {
                    "descriptor": {"type": "se_atten", "sel": 10, "rcut": 4.0},
                    "fitting_net": {"neuron": [5, 5]},
                },
            ],
            "weights": "mean",
        }

        result, min_dist = LinearEnergyModelDirect.update_sel(
            train_data=None,
            type_map=["O", "H"],
            local_jdata=local_jdata,
        )

        for idx, sub_model in enumerate(result["models"]):
            self.assertEqual(
                sub_model["descriptor"]["sel"],
                99,
                f"Sub-model {idx} sel was not updated in returned config",
            )


if __name__ == "__main__":
    unittest.main()
