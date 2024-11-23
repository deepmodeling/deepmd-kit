# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import numpy as np
import torch

from deepmd.dpmodel.model import SpinModel as DPSpinModel
from deepmd.pt.model.model import (
    SpinEnergyModel,
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_permutation import (
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
    model_spin,
)

dtype = torch.float64


def reduce_tensor(extended_tensor, mapping, nloc: int):
    nframes, nall = extended_tensor.shape[:2]
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
    # nf x nloc x (*ext_dims)
    reduced_tensor = torch.scatter_reduce(
        reduced_tensor,
        1,
        index=mapping,
        src=extended_tensor,
        reduce="sum",
    )
    return reduced_tensor


class SpinTest:
    def setUp(self) -> None:
        self.prec = 1e-10
        natoms = 5
        self.ntypes = 3  # ["O", "H", "B"] for test
        self.cell = 4.0 * torch.eye(3, dtype=dtype, device=env.DEVICE).unsqueeze(0)
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        self.coord = 3.0 * torch.rand(
            [natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        ).unsqueeze(0)
        self.spin = 0.5 * torch.rand(
            [natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        ).unsqueeze(0)
        self.atype = torch.tensor(
            [0, 0, 0, 1, 1], dtype=torch.int64, device=env.DEVICE
        ).unsqueeze(0)

        self.expected_mask = torch.tensor(
            [
                [True],
                [True],
                [True],
                [False],
                [False],
            ],
            dtype=torch.bool,
            device=env.DEVICE,
        ).unsqueeze(0)
        self.expected_atype_with_spin = torch.tensor(
            [0, 0, 0, 1, 1, 3, 3, 3, 4, 4], dtype=torch.int64, device=env.DEVICE
        ).unsqueeze(0)
        self.expected_nloc_spin_index = (
            torch.arange(natoms, natoms * 2, dtype=torch.int64, device=env.DEVICE)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

    def test_output_shape(
        self,
    ) -> None:
        result = self.model(
            self.coord,
            self.atype,
            self.spin,
            self.cell,
        )
        # check magnetic mask
        torch.testing.assert_close(result["mask_mag"], self.expected_mask)
        # check output shape to assure split
        nframes, nloc = self.coord.shape[:2]
        torch.testing.assert_close(result["energy"].shape, [nframes, 1])
        torch.testing.assert_close(result["atom_energy"].shape, [nframes, nloc, 1])
        torch.testing.assert_close(result["force"].shape, [nframes, nloc, 3])
        torch.testing.assert_close(result["force_mag"].shape, [nframes, nloc, 3])

    def test_input_output_process(self) -> None:
        nframes, nloc = self.coord.shape[:2]
        self.real_ntypes = self.model.spin.get_ntypes_real()
        # 1. test forward input process
        coord_updated, atype_updated = self.model.process_spin_input(
            self.coord, self.atype, self.spin
        )
        # compare atypes of real and virtual atoms
        torch.testing.assert_close(atype_updated, self.expected_atype_with_spin)
        # compare coords of real and virtual atoms
        torch.testing.assert_close(coord_updated.shape, [nframes, nloc * 2, 3])
        torch.testing.assert_close(coord_updated[:, :nloc], self.coord)
        virtual_scale = torch.tensor(
            self.model.spin.get_virtual_scale_mask()[self.atype.cpu()],
            dtype=dtype,
            device=env.DEVICE,
        )
        virtual_coord = self.coord + self.spin * virtual_scale.unsqueeze(-1)
        torch.testing.assert_close(coord_updated[:, nloc:], virtual_coord)

        # 2. test forward output process
        model_ret = self.model.backbone_model.forward_common(
            coord_updated,
            atype_updated,
            self.cell,
            do_atomic_virial=True,
        )
        if self.model.do_grad_r("energy"):
            force_all = model_ret["energy_derv_r"].squeeze(-2)
            force_real, force_mag, _ = self.model.process_spin_output(
                self.atype, force_all
            )
            torch.testing.assert_close(
                force_real, force_all[:, :nloc] + force_all[:, nloc:]
            )
            torch.testing.assert_close(
                force_mag, force_all[:, nloc:] * virtual_scale.unsqueeze(-1)
            )

        # 3. test forward_lower input process
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            self.model.get_rcut(),
            self.model.get_sel(),
            mixed_types=self.model.mixed_types(),
            box=self.cell,
        )
        nall = extended_coord.shape[1]
        nnei = nlist.shape[-1]
        extended_spin = torch.gather(
            self.spin, index=mapping.unsqueeze(-1).tile((1, 1, 3)), dim=1
        )
        (
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping_updated,
        ) = self.model.process_spin_input_lower(
            extended_coord, extended_atype, extended_spin, nlist, mapping=mapping
        )
        # compare atypes of real and virtual atoms
        # Note that the real and virtual atoms corresponding to the local ones are switch to the first nloc * 2 atoms
        torch.testing.assert_close(extended_atype_updated.shape, [nframes, nall * 2])
        torch.testing.assert_close(
            extended_atype_updated[:, :nloc], extended_atype[:, :nloc]
        )
        torch.testing.assert_close(
            extended_atype_updated[:, nloc : nloc + nloc],
            extended_atype[:, :nloc] + self.real_ntypes,
        )
        torch.testing.assert_close(
            extended_atype_updated[:, nloc + nloc : nloc + nall],
            extended_atype[:, nloc:nall],
        )
        torch.testing.assert_close(
            extended_atype_updated[:, nloc + nall :],
            extended_atype[:, nloc:nall] + self.real_ntypes,
        )
        virtual_scale = torch.tensor(
            self.model.spin.get_virtual_scale_mask()[extended_atype.cpu()],
            dtype=dtype,
            device=env.DEVICE,
        )
        # compare coords of real and virtual atoms
        virtual_coord = extended_coord + extended_spin * virtual_scale.unsqueeze(-1)
        torch.testing.assert_close(extended_coord_updated.shape, [nframes, nall * 2, 3])
        torch.testing.assert_close(
            extended_coord_updated[:, :nloc], extended_coord[:, :nloc]
        )
        torch.testing.assert_close(
            extended_coord_updated[:, nloc : nloc + nloc], virtual_coord[:, :nloc]
        )
        torch.testing.assert_close(
            extended_coord_updated[:, nloc + nloc : nloc + nall],
            extended_coord[:, nloc:nall],
        )
        torch.testing.assert_close(
            extended_coord_updated[:, nloc + nall :], virtual_coord[:, nloc:nall]
        )

        # compare mapping
        torch.testing.assert_close(mapping_updated.shape, [nframes, nall * 2])
        torch.testing.assert_close(mapping_updated[:, :nloc], mapping[:, :nloc])
        torch.testing.assert_close(
            mapping_updated[:, nloc : nloc + nloc], mapping[:, :nloc] + nloc
        )
        torch.testing.assert_close(
            mapping_updated[:, nloc + nloc : nloc + nall], mapping[:, nloc:nall]
        )
        torch.testing.assert_close(
            mapping_updated[:, nloc + nall :], mapping[:, nloc:nall] + nloc
        )

        # compare nlist
        torch.testing.assert_close(
            nlist_updated.shape, [nframes, nloc * 2, nnei * 2 + 1]
        )
        # self spin
        torch.testing.assert_close(
            nlist_updated[:, :nloc, :1], self.expected_nloc_spin_index
        )
        # real and virtual neighbors
        loc_atoms_mask = (nlist < nloc) & (nlist != -1)
        ghost_atoms_mask = nlist >= nloc
        real_neighbors = nlist.clone()
        real_neighbors[ghost_atoms_mask] += nloc
        torch.testing.assert_close(
            nlist_updated[:, :nloc, 1 : 1 + nnei], real_neighbors
        )
        virtual_neighbors = nlist.clone()
        virtual_neighbors[loc_atoms_mask] += nloc
        virtual_neighbors[ghost_atoms_mask] += nall
        torch.testing.assert_close(
            nlist_updated[:, :nloc, 1 + nnei :], virtual_neighbors
        )

        # 4. test forward_lower output process
        model_ret = self.model.backbone_model.forward_common_lower(
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping=mapping_updated,
            do_atomic_virial=True,
        )
        if self.model.do_grad_r("energy"):
            force_all = model_ret["energy_derv_r"].squeeze(-2)
            force_real, force_mag, _ = self.model.process_spin_output_lower(
                extended_atype, force_all, nloc
            )
            force_all_switched = torch.zeros_like(force_all)
            force_all_switched[:, :nloc] = force_all[:, :nloc]
            force_all_switched[:, nloc:nall] = force_all[:, nloc + nloc : nloc + nall]
            force_all_switched[:, nall : nall + nloc] = force_all[:, nloc : nloc + nloc]
            force_all_switched[:, nall + nloc :] = force_all[:, nloc + nall :]
            torch.testing.assert_close(
                force_real, force_all_switched[:, :nall] + force_all_switched[:, nall:]
            )
            torch.testing.assert_close(
                force_mag, force_all_switched[:, nall:] * virtual_scale.unsqueeze(-1)
            )

    def test_jit(self) -> None:
        model = torch.jit.script(self.model)
        self.assertEqual(model.get_rcut(), self.rcut)
        self.assertEqual(model.get_nsel(), self.nsel)
        self.assertEqual(model.get_type_map(), self.type_map)

    def test_self_consistency(self) -> None:
        if hasattr(self, "serial_test") and not self.serial_test:
            # not implement serialize and deserialize
            return
        model1 = SpinEnergyModel.deserialize(self.model.serialize())
        result = model1(
            self.coord,
            self.atype,
            self.spin,
            self.cell,
        )
        expected_result = self.model(
            self.coord,
            self.atype,
            self.spin,
            self.cell,
        )
        for key in result:
            torch.testing.assert_close(
                result[key], expected_result[key], rtol=self.prec, atol=self.prec
            )
        model1 = torch.jit.script(model1)

    def test_dp_consistency(self) -> None:
        if hasattr(self, "serial_test") and not self.serial_test:
            # not implement serialize and deserialize
            return
        dp_model = DPSpinModel.deserialize(self.model.serialize())
        # test call
        dp_ret = dp_model.call(
            to_numpy_array(self.coord),
            to_numpy_array(self.atype),
            to_numpy_array(self.spin),
            to_numpy_array(self.cell),
        )
        result = self.model.forward_common(
            self.coord,
            self.atype,
            self.spin,
            self.cell,
        )
        np.testing.assert_allclose(
            to_numpy_array(result["energy"]),
            dp_ret["energy"],
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            to_numpy_array(result["energy_redu"]),
            dp_ret["energy_redu"],
            rtol=self.prec,
            atol=self.prec,
        )

        # test call_lower
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            self.model.get_rcut(),
            self.model.get_sel(),
            mixed_types=self.model.mixed_types(),
            box=self.cell,
        )
        extended_spin = torch.gather(
            self.spin, index=mapping.unsqueeze(-1).tile((1, 1, 3)), dim=1
        )
        dp_ret_lower = dp_model.call_lower(
            to_numpy_array(extended_coord),
            to_numpy_array(extended_atype),
            to_numpy_array(extended_spin),
            to_numpy_array(nlist),
            to_numpy_array(mapping),
        )
        result_lower = self.model.forward_common_lower(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping,
        )
        np.testing.assert_allclose(
            to_numpy_array(result_lower["energy"]),
            dp_ret_lower["energy"],
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            to_numpy_array(result_lower["energy_redu"]),
            dp_ret_lower["energy_redu"],
            rtol=self.prec,
            atol=self.prec,
        )


class TestEnergyModelSpinSeA(unittest.TestCase, SpinTest):
    def setUp(self) -> None:
        SpinTest.setUp(self)
        model_params = copy.deepcopy(model_spin)
        model_params["descriptor"] = copy.deepcopy(model_se_e2_a["descriptor"])
        self.rcut = model_params["descriptor"]["rcut"]
        self.nsel = sum(model_params["descriptor"]["sel"])
        self.type_map = model_params["type_map"]
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinDPA1(unittest.TestCase, SpinTest):
    def setUp(self) -> None:
        SpinTest.setUp(self)
        model_params = copy.deepcopy(model_spin)
        model_params["descriptor"] = copy.deepcopy(model_dpa1["descriptor"])
        self.rcut = model_params["descriptor"]["rcut"]
        self.nsel = model_params["descriptor"]["sel"]
        self.type_map = model_params["type_map"]
        # not implement serialize and deserialize
        self.serial_test = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinDPA2(unittest.TestCase, SpinTest):
    def setUp(self) -> None:
        SpinTest.setUp(self)
        model_params = copy.deepcopy(model_spin)
        model_params["descriptor"] = copy.deepcopy(model_dpa2["descriptor"])
        self.rcut = model_params["descriptor"]["repinit"]["rcut"]
        self.nsel = model_params["descriptor"]["repinit"]["nsel"]
        self.type_map = model_params["type_map"]
        # not implement serialize and deserialize
        self.serial_test = False
        self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
