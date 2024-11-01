# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import paddle

from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.dpmodel.fitting import EnergyFittingNet as DPEnergyFittingNet
from deepmd.dpmodel.model.ener_model import EnergyModel as DPEnergyModel
from deepmd.pd.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pd.model.model import (
    EnergyModel,
)
from deepmd.pd.model.task.ener import (
    EnergyFittingNet,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
    extend_input_and_build_neighbor_list,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
    to_paddle_tensor,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
    TestCaseSingleFrameWithoutNlist,
)

dtype = env.GLOBAL_PD_FLOAT_PRECISION


class TestDPModel(unittest.TestCase, TestCaseSingleFrameWithoutNlist):
    def setUp(self):
        TestCaseSingleFrameWithoutNlist.setUp(self)

    def test_self_consistency(self):
        nf, nloc = self.atype.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = EnergyModel(ds, ft, type_map=type_map).to(env.DEVICE)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)
        args = [to_paddle_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        ret0 = md0.forward_common(*args)
        ret1 = md1.forward_common(*args)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]),
            to_numpy_array(ret1["energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_redu"]),
            to_numpy_array(ret1["energy_redu"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_r"]),
            to_numpy_array(ret1["energy_derv_r"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c_redu"]),
            to_numpy_array(ret1["energy_derv_c_redu"]),
            atol=self.atol,
        )
        ret0 = md0.forward_common(*args, do_atomic_virial=True)
        ret1 = md1.forward_common(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c"]),
            to_numpy_array(ret1["energy_derv_c"]),
            atol=self.atol,
        )

        coord_ext, atype_ext, mapping = extend_coord_with_ghosts(
            to_paddle_tensor(self.coord),
            to_paddle_tensor(self.atype),
            to_paddle_tensor(self.cell),
            self.rcut,
        )
        nlist = build_neighbor_list(
            coord_ext,
            atype_ext,
            self.nloc,
            self.rcut,
            self.sel,
            distinguish_types=(not md0.mixed_types()),
        )
        args = [coord_ext, atype_ext, nlist]
        ret2 = md0.forward_common_lower(*args, do_atomic_virial=True)
        # check the consistency between the reduced virial from
        # forward_common and forward_common_lower
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c_redu"]),
            to_numpy_array(ret2["energy_derv_c_redu"]),
            atol=self.atol,
        )

    def test_dp_consistency(self):
        nf, nloc = self.atype.shape
        nfp, nap = 2, 3
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPEnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            numb_fparam=nfp,
            numb_aparam=nap,
        )
        type_map = ["foo", "bar"]
        md0 = DPEnergyModel(ds, ft, type_map=type_map)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)

        rng = np.random.default_rng(GLOBAL_SEED)
        fparam = rng.normal(size=[self.nf, nfp])
        aparam = rng.normal(size=[self.nf, nloc, nap])
        args0 = [self.coord, self.atype, self.cell]
        args1 = [to_paddle_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        kwargs0 = {"fparam": fparam, "aparam": aparam}
        kwargs1 = {kk: to_paddle_tensor(vv) for kk, vv in kwargs0.items()}
        ret0 = md0.call(*args0, **kwargs0)
        ret1 = md1.forward_common(*args1, **kwargs1)
        np.testing.assert_allclose(
            ret0["energy"],
            to_numpy_array(ret1["energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            ret0["energy_redu"],
            to_numpy_array(ret1["energy_redu"]),
            atol=self.atol,
        )

    def test_dp_consistency_nopbc(self):
        nf, nloc = self.atype.shape
        nfp, nap = 2, 3
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPEnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            numb_fparam=nfp,
            numb_aparam=nap,
        )
        type_map = ["foo", "bar"]
        md0 = DPEnergyModel(ds, ft, type_map=type_map)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)

        rng = np.random.default_rng(GLOBAL_SEED)
        fparam = rng.normal(size=[self.nf, nfp])
        aparam = rng.normal(size=[self.nf, self.nloc, nap])
        args0 = [self.coord, self.atype]
        args1 = [to_paddle_tensor(ii) for ii in args0]
        kwargs0 = {"fparam": fparam, "aparam": aparam}
        kwargs1 = {kk: to_paddle_tensor(vv) for kk, vv in kwargs0.items()}
        ret0 = md0.call(*args0, **kwargs0)
        ret1 = md1.forward_common(*args1, **kwargs1)
        np.testing.assert_allclose(
            ret0["energy"],
            to_numpy_array(ret1["energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            ret0["energy_redu"],
            to_numpy_array(ret1["energy_redu"]),
            atol=self.atol,
        )

    def test_prec_consistency(self):
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc = self.atype.shape
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPEnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        nfp, nap = 2, 3
        type_map = ["foo", "bar"]
        fparam = rng.normal(size=[self.nf, nfp])
        aparam = rng.normal(size=[self.nf, nloc, nap])

        md0 = DPEnergyModel(ds, ft, type_map=type_map)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)

        args64 = [to_paddle_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        args64[0] = args64[0].to(paddle.float64)
        args64[2] = args64[2].to(paddle.float64)
        args32 = [to_paddle_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        args32[0] = args32[0].to(paddle.float32)
        args32[2] = args32[2].to(paddle.float32)
        # fparam, aparam are converted to coordinate precision by model
        fparam = to_paddle_tensor(fparam)
        aparam = to_paddle_tensor(aparam)

        model_l_ret_64 = md1.forward_common(*args64, fparam=fparam, aparam=aparam)
        model_l_ret_32 = md1.forward_common(*args32, fparam=fparam, aparam=aparam)

        for ii in model_l_ret_32.keys():
            if ii[-4:] == "redu":
                self.assertEqual(model_l_ret_32[ii].dtype, paddle.float64)
            else:
                self.assertEqual(model_l_ret_32[ii].dtype, paddle.float32)
            if ii != "mask":
                self.assertEqual(model_l_ret_64[ii].dtype, paddle.float64)
            else:
                self.assertEqual(model_l_ret_64[ii].dtype, paddle.int32)
            np.testing.assert_allclose(
                to_numpy_array(model_l_ret_32[ii]),
                to_numpy_array(model_l_ret_64[ii]),
                atol=self.atol,
            )


class TestDPModelLower(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = EnergyModel(ds, ft, type_map=type_map).to(env.DEVICE)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)
        args = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = md0.forward_common_lower(*args)
        ret1 = md1.forward_common_lower(*args)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]),
            to_numpy_array(ret1["energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_redu"]),
            to_numpy_array(ret1["energy_redu"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_r"]),
            to_numpy_array(ret1["energy_derv_r"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c_redu"]),
            to_numpy_array(ret1["energy_derv_c_redu"]),
            atol=self.atol,
        )
        ret0 = md0.forward_common_lower(*args, do_atomic_virial=True)
        ret1 = md1.forward_common_lower(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c"]),
            to_numpy_array(ret1["energy_derv_c"]),
            atol=self.atol,
        )

    def test_dp_consistency(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPEnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]
        md0 = DPEnergyModel(ds, ft, type_map=type_map)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)
        args0 = [self.coord_ext, self.atype_ext, self.nlist]
        args1 = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = md0.call_lower(*args0)
        ret1 = md1.forward_common_lower(*args1)
        np.testing.assert_allclose(
            ret0["energy"],
            to_numpy_array(ret1["energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            ret0["energy_redu"],
            to_numpy_array(ret1["energy_redu"]),
            atol=self.atol,
        )

    def test_prec_consistency(self):
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPEnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        nfp, nap = 2, 3
        type_map = ["foo", "bar"]
        fparam = rng.normal(size=[self.nf, nfp])
        aparam = rng.normal(size=[self.nf, nloc, nap])

        md0 = DPEnergyModel(ds, ft, type_map=type_map)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)

        args64 = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        args64[0] = args64[0].to(paddle.float64)
        args32 = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        args32[0] = args32[0].to(paddle.float32)
        # fparam, aparam are converted to coordinate precision by model
        fparam = to_paddle_tensor(fparam)
        aparam = to_paddle_tensor(aparam)

        model_l_ret_64 = md1.forward_common_lower(*args64, fparam=fparam, aparam=aparam)
        model_l_ret_32 = md1.forward_common_lower(*args32, fparam=fparam, aparam=aparam)

        for ii in model_l_ret_32.keys():
            if ii[-4:] == "redu":
                self.assertEqual(model_l_ret_32[ii].dtype, paddle.float64)
            else:
                self.assertEqual(model_l_ret_32[ii].dtype, paddle.float32)
            if ii != "mask":
                self.assertEqual(model_l_ret_64[ii].dtype, paddle.float64)
            else:
                self.assertEqual(model_l_ret_64[ii].dtype, paddle.int32)
            np.testing.assert_allclose(
                to_numpy_array(model_l_ret_32[ii]),
                to_numpy_array(model_l_ret_64[ii]),
                atol=self.atol,
            )

    def test_jit(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = EnergyModel(ds, ft, type_map=type_map).to(env.DEVICE)
        md0 = paddle.jit.to_static(md0)
        md0.get_rcut()
        md0.get_type_map()


class TestDPModelFormatNlist(unittest.TestCase):
    def setUp(self):
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 5
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
                [2.3, 0, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.expected_nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype="int64",
        ).reshape([1, self.nloc, sum(self.sel)])
        self.atype_ext = np.array([0, 0, 1, 0, 1], dtype="int64").reshape(
            [1, self.nall]
        )
        self.rcut_smth = 0.4
        self.rcut = 2.0

        nf, nloc, nnei = self.expected_nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        self.md = EnergyModel(ds, ft, type_map=type_map).to(env.DEVICE)

    def test_nlist_eq(self):
        # n_nnei == nnei
        nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            to_paddle_tensor(self.coord_ext),
            to_paddle_tensor(self.atype_ext),
            to_paddle_tensor(nlist),
        )
        np.testing.assert_equal(self.expected_nlist, to_numpy_array(nlist1))

    def test_nlist_st(self):
        # n_nnei < nnei
        nlist = np.array(
            [
                [1, 3, -1, 2],
                [0, -1, -1, 2],
                [0, 1, -1, -1],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            to_paddle_tensor(self.coord_ext),
            to_paddle_tensor(self.atype_ext),
            to_paddle_tensor(nlist),
        )
        np.testing.assert_equal(self.expected_nlist, to_numpy_array(nlist1))

    def test_nlist_lt(self):
        # n_nnei > nnei
        nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1, -1, 4],
                [0, -1, 4, -1, -1, 2, -1, 3, -1],
                [0, 1, -1, -1, -1, 4, -1, -1, 3],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            to_paddle_tensor(self.coord_ext),
            to_paddle_tensor(self.atype_ext),
            to_paddle_tensor(nlist),
        )
        np.testing.assert_equal(self.expected_nlist, to_numpy_array(nlist1))


class TestEnergyModel(unittest.TestCase, TestCaseSingleFrameWithoutNlist):
    def setUp(self):
        TestCaseSingleFrameWithoutNlist.setUp(self)

    def test_self_consistency(self):
        nf, nloc = self.atype.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = EnergyModel(ds, ft, type_map=type_map).to(env.DEVICE)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)
        args = [to_paddle_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        ret0 = md0.forward(*args)
        ret1 = md1.forward(*args)
        np.testing.assert_allclose(
            to_numpy_array(ret0["atom_energy"]),
            to_numpy_array(ret1["atom_energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]),
            to_numpy_array(ret1["energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["force"]),
            to_numpy_array(ret1["force"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["virial"]),
            to_numpy_array(ret1["virial"]),
            atol=self.atol,
        )
        ret0 = md0.forward(*args, do_atomic_virial=True)
        ret1 = md1.forward(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["atom_virial"]),
            to_numpy_array(ret1["atom_virial"]),
            atol=self.atol,
        )
        coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
            to_paddle_tensor(self.coord),
            to_paddle_tensor(self.atype),
            self.rcut,
            self.sel,
            mixed_types=md0.mixed_types(),
            box=to_paddle_tensor(self.cell),
        )
        args = [coord_ext, atype_ext, nlist]
        ret2 = md0.forward_lower(*args, do_atomic_virial=True)
        # check the consistency between the reduced virial from
        # forward and forward_lower
        np.testing.assert_allclose(
            to_numpy_array(ret0["virial"]),
            to_numpy_array(ret2["virial"]),
            atol=self.atol,
        )


class TestEnergyModelLower(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = EnergyModel(ds, ft, type_map=type_map).to(env.DEVICE)
        md1 = EnergyModel.deserialize(md0.serialize()).to(env.DEVICE)
        args = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = md0.forward_lower(*args)
        ret1 = md1.forward_lower(*args)
        np.testing.assert_allclose(
            to_numpy_array(ret0["atom_energy"]),
            to_numpy_array(ret1["atom_energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]),
            to_numpy_array(ret1["energy"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["extended_force"]),
            to_numpy_array(ret1["extended_force"]),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["virial"]),
            to_numpy_array(ret1["virial"]),
            atol=self.atol,
        )
        ret0 = md0.forward_lower(*args, do_atomic_virial=True)
        ret1 = md1.forward_lower(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["extended_virial"]),
            to_numpy_array(ret1["extended_virial"]),
            atol=self.atol,
        )

    def test_jit(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = EnergyModel(ds, ft, type_map=type_map).to(env.DEVICE)
        md0 = paddle.jit.to_static(md0)
        self.assertEqual(md0.get_rcut(), self.rcut)
        self.assertEqual(md0.get_type_map(), type_map)
