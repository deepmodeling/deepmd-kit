# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import numpy as np
import paddle

from deepmd.dpmodel.model import SpinModel as DPSpinModel
from deepmd.pd.model.model import (
    SpinEnergyModel,
    get_model,
)
from deepmd.pd.utils import (
    decomp,
    env,
)
from deepmd.pd.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pd.utils.utils import (
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

dtype = paddle.float64


def reduce_tensor(extended_tensor, mapping, nloc: int):
    nframes, nall = extended_tensor.shape[:2]
    ext_dims = extended_tensor.shape[2:]
    reduced_tensor = paddle.zeros(
        [nframes, nloc, *ext_dims],
        dtype=extended_tensor.dtype,
    ).to(device=extended_tensor.place)
    mldims = list(mapping.shape)
    mapping = mapping.reshape(mldims + [1] * len(ext_dims)).expand(
        [-1] * len(mldims) + list(ext_dims)
    )
    # nf x nloc x (*ext_dims)
    reduced_tensor = decomp.scatter_reduce(
        reduced_tensor,
        1,
        index=mapping,
        src=extended_tensor,
        reduce="sum",
    )
    return reduced_tensor


class SpinTest:
    def setUp(self):
        self.prec = 1e-10
        natoms = 5
        self.ntypes = 3  # ["O", "H", "B"] for test
        self.cell = 4.0 * paddle.eye(3, dtype=dtype).to(device=env.DEVICE).unsqueeze(0)
        generator = paddle.seed(GLOBAL_SEED)
        self.coord = 3.0 * paddle.rand([natoms, 3], dtype=dtype).unsqueeze(0).to(
            device=env.DEVICE
        )
        self.spin = 0.5 * paddle.rand([natoms, 3], dtype=dtype).unsqueeze(0).to(
            device=env.DEVICE
        )
        self.atype = paddle.to_tensor(
            [0, 0, 0, 1, 1], dtype=paddle.int64, place=env.DEVICE
        ).unsqueeze(0)

        self.expected_mask = paddle.to_tensor(
            [
                [True],
                [True],
                [True],
                [False],
                [False],
            ],
            dtype=paddle.bool,
            place=env.DEVICE,
        ).unsqueeze(0)
        self.expected_atype_with_spin = paddle.to_tensor(
            [0, 0, 0, 1, 1, 3, 3, 3, 4, 4], dtype=paddle.int64, place=env.DEVICE
        ).unsqueeze(0)
        self.expected_nloc_spin_index = (
            paddle.arange(natoms, natoms * 2, dtype=paddle.int64)
            .to(device=env.DEVICE)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

    def test_output_shape(
        self,
    ):
        result = self.model(
            self.coord,
            self.atype,
            self.spin,
            self.cell,
        )
        # check magnetic mask
        assert np.allclose(result["mask_mag"].numpy(), self.expected_mask.numpy())
        # check output shape to assure split
        nframes, nloc = self.coord.shape[:2]
        assert np.allclose(result["energy"].shape, [nframes, 1])
        assert np.allclose(result["atom_energy"].shape, [nframes, nloc, 1])
        assert np.allclose(result["force"].shape, [nframes, nloc, 3])
        assert np.allclose(result["force_mag"].shape, [nframes, nloc, 3])

    def test_input_output_process(self):
        nframes, nloc = self.coord.shape[:2]
        self.real_ntypes = self.model.spin.get_ntypes_real()
        # 1. test forward input process
        coord_updated, atype_updated = self.model.process_spin_input(
            self.coord, self.atype, self.spin
        )
        # compare atypes of real and virtual atoms
        assert np.allclose(atype_updated.numpy(), self.expected_atype_with_spin.numpy())
        # compare coords of real and virtual atoms
        assert np.allclose(coord_updated.shape, [nframes, nloc * 2, 3])
        assert np.allclose(coord_updated[:, :nloc].numpy(), self.coord.numpy())
        virtual_scale = paddle.to_tensor(
            self.model.spin.get_virtual_scale_mask()[self.atype.cpu()],
            dtype=dtype,
            place=env.DEVICE,
        )
        virtual_coord = self.coord + self.spin * virtual_scale.unsqueeze(-1)
        assert np.allclose(coord_updated[:, nloc:].numpy(), virtual_coord.numpy())

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
            np.testing.assert_allclose(
                force_real.numpy(), (force_all[:, :nloc] + force_all[:, nloc:]).numpy()
            )
            np.testing.assert_allclose(
                force_mag.numpy(),
                (force_all[:, nloc:] * virtual_scale.unsqueeze(-1)).numpy(),
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
        extended_spin = decomp.take_along_axis(
            self.spin, indices=mapping.unsqueeze(-1).tile((1, 1, 3)), axis=1
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
        assert np.allclose(extended_atype_updated.shape, [nframes, nall * 2])
        assert np.allclose(
            extended_atype_updated[:, :nloc].numpy(), extended_atype[:, :nloc].numpy()
        )
        assert np.allclose(
            extended_atype_updated[:, nloc : nloc + nloc].numpy(),
            extended_atype[:, :nloc].numpy() + self.real_ntypes,
        )
        assert np.allclose(
            extended_atype_updated[:, nloc + nloc : nloc + nall].numpy(),
            extended_atype[:, nloc:nall].numpy(),
        )
        assert np.allclose(
            extended_atype_updated[:, nloc + nall :].numpy(),
            extended_atype[:, nloc:nall].numpy() + self.real_ntypes,
        )
        virtual_scale = paddle.to_tensor(
            self.model.spin.get_virtual_scale_mask()[extended_atype.cpu()],
            dtype=dtype,
            place=env.DEVICE,
        )
        # compare coords of real and virtual atoms
        virtual_coord = extended_coord + extended_spin * virtual_scale.unsqueeze(-1)
        assert np.allclose(extended_coord_updated.shape, [nframes, nall * 2, 3])
        np.testing.assert_allclose(
            extended_coord_updated[:, :nloc].numpy(), extended_coord[:, :nloc].numpy()
        )
        np.testing.assert_allclose(
            extended_coord_updated[:, nloc : nloc + nloc].numpy(),
            virtual_coord[:, :nloc].numpy(),
        )
        np.testing.assert_allclose(
            extended_coord_updated[:, nloc + nloc : nloc + nall].numpy(),
            extended_coord[:, nloc:nall].numpy(),
        )
        np.testing.assert_allclose(
            extended_coord_updated[:, nloc + nall :].numpy(),
            virtual_coord[:, nloc:nall].numpy(),
        )

        # compare mapping
        assert np.allclose(mapping_updated.shape, [nframes, nall * 2])
        assert np.allclose(mapping_updated[:, :nloc].numpy(), mapping[:, :nloc].numpy())
        assert np.allclose(
            mapping_updated[:, nloc : nloc + nloc].numpy(),
            mapping[:, :nloc].numpy() + nloc,
        )
        assert np.allclose(
            mapping_updated[:, nloc + nloc : nloc + nall].numpy(),
            mapping[:, nloc:nall].numpy(),
        )
        assert np.allclose(
            mapping_updated[:, nloc + nall :].numpy(),
            mapping[:, nloc:nall].numpy() + nloc,
        )

        # compare nlist
        assert np.allclose(nlist_updated.shape, [nframes, nloc * 2, nnei * 2 + 1])
        # self spin
        assert np.allclose(
            nlist_updated[:, :nloc, :1].numpy(), self.expected_nloc_spin_index.numpy()
        )
        # real and virtual neighbors
        loc_atoms_mask = (nlist < nloc) & (nlist != -1)
        ghost_atoms_mask = nlist >= nloc
        real_neighbors = nlist.clone()
        decomp.masked_add_(real_neighbors, ghost_atoms_mask, nloc)
        # real_neighbors[ghost_atoms_mask] += nloc
        assert np.allclose(
            nlist_updated[:, :nloc, 1 : 1 + nnei].numpy(), real_neighbors.numpy()
        )
        virtual_neighbors = nlist.clone()
        # virtual_neighbors[loc_atoms_mask] += nloc
        decomp.masked_add_(virtual_neighbors, loc_atoms_mask, nloc)
        # virtual_neighbors[ghost_atoms_mask] += nall
        decomp.masked_add_(virtual_neighbors, ghost_atoms_mask, nall)
        assert np.allclose(
            nlist_updated[:, :nloc, 1 + nnei :].numpy(), virtual_neighbors.numpy()
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
            force_all_switched = paddle.zeros_like(force_all)
            force_all_switched[:, :nloc] = force_all[:, :nloc]
            force_all_switched[:, nloc:nall] = force_all[:, nloc + nloc : nloc + nall]
            force_all_switched[:, nall : nall + nloc] = force_all[:, nloc : nloc + nloc]
            force_all_switched[:, nall + nloc :] = force_all[:, nloc + nall :]
            np.testing.assert_allclose(
                force_real.numpy(),
                (force_all_switched[:, :nall] + force_all_switched[:, nall:]).numpy(),
            )
            np.testing.assert_allclose(
                force_mag.numpy(),
                (force_all_switched[:, nall:] * virtual_scale.unsqueeze(-1)).numpy(),
            )

    def test_jit(self):
        model = paddle.jit.to_static(self.model)
        self.assertEqual(model.get_rcut(), self.rcut)
        self.assertEqual(model.get_nsel(), self.nsel)
        self.assertEqual(model.get_type_map(), self.type_map)

    def test_self_consistency(self):
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
            assert np.allclose(
                result[key].numpy(),
                expected_result[key].numpy(),
                rtol=self.prec,
                atol=self.prec,
            )
        model1 = paddle.jit.to_static(model1)

    def test_dp_consistency(self):
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
        extended_spin = decomp.take_along_axis(
            self.spin, indices=mapping.unsqueeze(-1).tile((1, 1, 3)), axis=1
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
    def setUp(self):
        SpinTest.setUp(self)
        model_params = copy.deepcopy(model_spin)
        model_params["descriptor"] = copy.deepcopy(model_se_e2_a["descriptor"])
        self.rcut = model_params["descriptor"]["rcut"]
        self.nsel = sum(model_params["descriptor"]["sel"])
        self.type_map = model_params["type_map"]
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinDPA1(unittest.TestCase, SpinTest):
    def setUp(self):
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
    def setUp(self):
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
