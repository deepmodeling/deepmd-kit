# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.pt_expt.model.spin_ener_model import (
    SpinEnergyModel,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)

dtype = torch.float64

SPIN_DATA = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20, 20],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [3, 6],
        "resnet_dt": False,
        "axis_neuron": 2,
        "precision": "float64",
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [5, 5],
        "resnet_dt": True,
        "precision": "float64",
        "seed": 1,
    },
    "spin": {
        "use_spin": [True, False, False],
        "virtual_scale": [0.3140],
    },
}


def finite_difference(f, x, delta=1e-6):
    in_shape = x.shape
    y0 = f(x)
    out_shape = y0.shape
    res = np.empty(out_shape + in_shape)
    for idx in np.ndindex(*in_shape):
        diff = np.zeros(in_shape)
        diff[idx] += delta
        y1p = f(x + diff)
        y1n = f(x - diff)
        res[(Ellipsis, *idx)] = (y1p - y1n) / (2 * delta)
    return res


def stretch_box(old_coord, old_box, new_box):
    ocoord = old_coord.reshape(-1, 3)
    obox = old_box.reshape(3, 3)
    nbox = new_box.reshape(3, 3)
    ncoord = ocoord @ np.linalg.inv(obox) @ nbox
    return ncoord.reshape(old_coord.shape)


def _make_model():
    dp_model = get_model_dp(SPIN_DATA)
    model = SpinEnergyModel.deserialize(dp_model.serialize()).to(env.DEVICE)
    model.eval()
    return model


def eval_model(model, coord, cell, atype, spin):
    """Evaluate the pt_expt SpinEnergyModel."""
    nframes = coord.shape[0]
    if len(atype.shape) == 1:
        atype = atype.unsqueeze(0).expand(nframes, -1)
    coord_input = coord.to(dtype=dtype, device=env.DEVICE)
    cell_input = cell.reshape(nframes, 9).to(dtype=dtype, device=env.DEVICE)
    atype_input = atype.to(dtype=torch.long, device=env.DEVICE)
    spin_input = spin.to(dtype=dtype, device=env.DEVICE)
    coord_input.requires_grad_(True)
    result = model(coord_input, atype_input, spin_input, cell_input)
    return result


class TestSpinEnerModelOutputKeys(unittest.TestCase):
    def test_output_keys(self) -> None:
        """Test that SpinEnergyModel produces expected output keys."""
        model = _make_model()
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([6, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int64)
        spin = torch.rand([6, 3], dtype=dtype, device="cpu", generator=generator) * 0.5

        result = eval_model(
            model,
            coord.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            spin.unsqueeze(0),
        )
        self.assertIn("energy", result)
        self.assertIn("atom_energy", result)
        self.assertIn("force", result)
        self.assertIn("force_mag", result)
        self.assertIn("mask_mag", result)
        self.assertIn("virial", result)

    def test_output_shapes(self) -> None:
        """Test that output shapes are correct."""
        model = _make_model()
        natoms = 6
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int64)
        spin = (
            torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
            * 0.5
        )

        result = eval_model(
            model,
            coord.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            spin.unsqueeze(0),
        )
        self.assertEqual(result["energy"].shape, (1, 1))
        self.assertEqual(result["atom_energy"].shape, (1, natoms, 1))
        self.assertEqual(result["force"].shape, (1, natoms, 3))
        self.assertEqual(result["force_mag"].shape, (1, natoms, 3))
        self.assertEqual(result["mask_mag"].shape, (1, natoms, 1))
        self.assertEqual(result["virial"].shape, (1, 9))


class TestSpinEnerModelSerialize(unittest.TestCase):
    def test_serialize_deserialize(self) -> None:
        """Test serialize/deserialize round-trip."""
        model = _make_model()
        serialized = model.serialize()
        model2 = SpinEnergyModel.deserialize(serialized).to(env.DEVICE)
        model2.eval()

        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([6, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int64)
        spin = torch.rand([6, 3], dtype=dtype, device="cpu", generator=generator) * 0.5

        ret1 = eval_model(
            model, coord.unsqueeze(0), cell.unsqueeze(0), atype, spin.unsqueeze(0)
        )
        ret2 = eval_model(
            model2, coord.unsqueeze(0), cell.unsqueeze(0), atype, spin.unsqueeze(0)
        )

        for key in ["energy", "atom_energy", "force", "force_mag", "mask_mag"]:
            np.testing.assert_allclose(
                ret1[key].detach().cpu().numpy(),
                ret2[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Mismatch in {key} after round-trip",
            )


class TestSpinEnerModelDPConsistency(unittest.TestCase):
    def test_dp_consistency(self) -> None:
        """Test numerical consistency with dpmodel."""
        dp_model = get_model_dp(SPIN_DATA)
        pt_model = SpinEnergyModel.deserialize(dp_model.serialize()).to(env.DEVICE)
        pt_model.eval()

        coords_np = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                0.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ],
            dtype=np.float64,
        ).reshape(1, -1, 3)
        atype_np = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int32)
        box_np = np.array(
            [13.0, 0, 0, 0, 13.0, 0, 0, 0, 13.0], dtype=np.float64
        ).reshape(1, 9)
        spin_np = np.array(
            [
                0.50,
                0.30,
                0.20,
                0.40,
                0.25,
                0.15,
                0.10,
                0.05,
                0.08,
                0.12,
                0.07,
                0.09,
                0.45,
                0.35,
                0.28,
                0.11,
                0.06,
                0.03,
            ],
            dtype=np.float64,
        ).reshape(1, -1, 3)

        dp_ret = dp_model(coords_np, atype_np, spin_np, box=box_np)

        pt_ret = eval_model(
            pt_model,
            torch.tensor(coords_np),
            torch.tensor(box_np),
            torch.tensor(atype_np.squeeze(), dtype=torch.int64),
            torch.tensor(spin_np),
        )

        np.testing.assert_allclose(
            dp_ret["energy"],
            pt_ret["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            dp_ret["atom_energy"],
            pt_ret["atom_energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            dp_ret["mask_mag"],
            pt_ret["mask_mag"].detach().cpu().numpy(),
        )


class ForceTest:
    def test(self) -> None:
        places = 5
        delta = 1e-5
        natoms = 6
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int64)
        spin = (
            torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
            * 0.5
        )
        coord = coord.numpy()

        def np_infer_coord(coord):
            result = eval_model(
                self.model,
                torch.tensor(coord, device=env.DEVICE).unsqueeze(0),
                cell.unsqueeze(0),
                atype,
                spin.unsqueeze(0),
            )
            ret = {
                key: result[key].squeeze(0).detach().cpu().numpy()
                for key in ["energy", "force", "force_mag", "virial"]
            }
            return ret

        def ff_coord(_coord):
            return np_infer_coord(_coord)["energy"]

        fdf = -finite_difference(ff_coord, coord, delta=delta).squeeze()
        rff = np_infer_coord(coord)["force"]
        np.testing.assert_almost_equal(fdf, rff, decimal=places)


class VirialTest:
    def test(self) -> None:
        places = 5
        delta = 1e-4
        natoms = 6
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int64)
        spin = (
            torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
            * 0.5
        )
        coord = coord.numpy()
        cell = cell.numpy()

        def np_infer(new_cell):
            result = eval_model(
                self.model,
                torch.tensor(
                    stretch_box(coord, cell, new_cell), device="cpu"
                ).unsqueeze(0),
                torch.tensor(new_cell, device="cpu").unsqueeze(0),
                atype,
                spin.unsqueeze(0),
            )
            ret = {
                key: result[key].squeeze(0).detach().cpu().numpy()
                for key in ["energy", "force", "virial"]
            }
            return ret

        def ff(bb):
            return np_infer(bb)["energy"]

        fdv = (
            -(finite_difference(ff, cell, delta=delta).transpose(0, 2, 1) @ cell)
            .squeeze()
            .reshape(9)
        )
        rfv = np_infer(cell)["virial"]
        np.testing.assert_almost_equal(fdv, rfv, decimal=places)


class TestSpinEnerModelForce(unittest.TestCase, ForceTest):
    def setUp(self) -> None:
        self.model = _make_model()


class TestSpinEnerModelVirial(unittest.TestCase, VirialTest):
    def setUp(self) -> None:
        self.model = _make_model()


class TestSpinEnerModelExportable(unittest.TestCase):
    def test_forward_lower_exportable(self) -> None:
        """Test that SpinEnergyModel.forward_lower_exportable works with make_fx and torch.export."""
        from deepmd.dpmodel.utils import (
            build_neighbor_list,
            extend_coord_with_ghosts,
            normalize_coord,
        )

        model = _make_model()
        natoms = 6
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int64)
        spin = (
            torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
            * 0.5
        )

        # Build extended inputs (use original sel, not backbone's doubled sel)
        rcut = model.get_rcut()
        sel = SPIN_DATA["descriptor"]["sel"]
        coord_np = coord.unsqueeze(0).numpy()
        atype_np = atype.unsqueeze(0).numpy()
        box_np = cell.reshape(1, 9).numpy()
        coord_normalized = normalize_coord(
            coord_np.reshape(1, natoms, 3),
            box_np.reshape(1, 3, 3),
        )
        ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
            coord_normalized,
            atype_np,
            box_np,
            rcut,
        )
        nlist = build_neighbor_list(
            ext_coord, ext_atype, natoms, rcut, sel, distinguish_types=True
        )
        ext_coord = ext_coord.reshape(1, -1, 3)
        # Extend spin to ghost atoms using mapping
        spin_np = spin.unsqueeze(0).numpy()
        ext_spin = np.take_along_axis(
            spin_np,
            np.repeat(mapping[:, :, np.newaxis], 3, axis=2),
            axis=1,
        )

        ext_coord_t = torch.tensor(ext_coord, dtype=dtype, device=env.DEVICE)
        ext_atype_t = torch.tensor(ext_atype, dtype=torch.int64, device=env.DEVICE)
        nlist_t = torch.tensor(nlist, dtype=torch.int64, device=env.DEVICE)
        mapping_t = torch.tensor(mapping, dtype=torch.int64, device=env.DEVICE)
        ext_spin_t = torch.tensor(ext_spin, dtype=dtype, device=env.DEVICE)

        output_keys = (
            "energy",
            "extended_force",
            "extended_force_mag",
            "virial",
        )

        # --- eager reference ---
        ret_eager = model.forward_lower(
            ext_coord_t.requires_grad_(True),
            ext_atype_t,
            ext_spin_t,
            nlist_t,
            mapping_t,
        )
        for key in output_keys:
            self.assertIn(key, ret_eager, f"Missing key {key} in eager result")

        # --- trace with make_fx ---
        traced = model.forward_lower_exportable(
            ext_coord_t,
            ext_atype_t,
            ext_spin_t,
            nlist_t,
            mapping_t,
        )
        self.assertIsInstance(traced, torch.nn.Module)

        # --- export with torch.export ---
        exported = torch.export.export(
            traced,
            (ext_coord_t, ext_atype_t, ext_spin_t, nlist_t, mapping_t, None, None),
            strict=False,
        )
        self.assertIsNotNone(exported)

        # --- verify traced matches eager ---
        ret_traced = traced(
            ext_coord_t, ext_atype_t, ext_spin_t, nlist_t, mapping_t, None, None
        )
        for key in output_keys:
            np.testing.assert_allclose(
                ret_eager[key].detach().cpu().numpy(),
                ret_traced[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"traced vs eager: {key}",
            )

        # --- verify exported matches eager ---
        ret_exported = exported.module()(
            ext_coord_t, ext_atype_t, ext_spin_t, nlist_t, mapping_t, None, None
        )
        for key in output_keys:
            np.testing.assert_allclose(
                ret_eager[key].detach().cpu().numpy(),
                ret_exported[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"exported vs eager: {key}",
            )


if __name__ == "__main__":
    unittest.main()
