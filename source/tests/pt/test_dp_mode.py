# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.model_format import DescrptSeA as DPDescrptSeA
from deepmd.model_format import DPModel as DPDPModel
from deepmd.model_format import InvarFitting as DPInvarFitting
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.ener import (
    DPModel,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
    TestCaseSingleFrameWithoutNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


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
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        # TODO: dirty hack to avoid data stat!!!
        md0 = DPModel(ds, ft, type_map=type_map, resuming=True).to(env.DEVICE)
        md1 = DPModel.deserialize(md0.serialize()).to(env.DEVICE)
        args = [to_torch_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        ret0 = md0.forward_common(*args)
        ret1 = md1.forward_common(*args)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]),
            to_numpy_array(ret1["energy"]),
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_redu"]),
            to_numpy_array(ret1["energy_redu"]),
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_r"]),
            to_numpy_array(ret1["energy_derv_r"]),
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c_redu"]),
            to_numpy_array(ret1["energy_derv_c_redu"]),
        )
        ret0 = md0.forward_common(*args, do_atomic_virial=True)
        ret1 = md1.forward_common(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c"]),
            to_numpy_array(ret1["energy_derv_c"]),
        )

    def test_dp_consistency(self):
        nf, nloc = self.atype.shape
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPInvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        )
        type_map = ["foo", "bar"]
        md0 = DPDPModel(ds, ft, type_map=type_map)
        md1 = DPModel.deserialize(md0.serialize()).to(env.DEVICE)
        args0 = [self.coord, self.atype, self.cell]
        args1 = [to_torch_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        ret0 = md0.call(*args0)
        ret1 = md1.forward_common(*args1)
        np.testing.assert_allclose(
            ret0["energy"],
            to_numpy_array(ret1["energy"]),
        )
        np.testing.assert_allclose(
            ret0["energy_redu"],
            to_numpy_array(ret1["energy_redu"]),
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
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        # TODO: dirty hack to avoid data stat!!!
        md0 = DPModel(ds, ft, type_map=type_map, resuming=True).to(env.DEVICE)
        md1 = DPModel.deserialize(md0.serialize()).to(env.DEVICE)
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = md0.forward_common_lower(*args)
        ret1 = md1.forward_common_lower(*args)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]),
            to_numpy_array(ret1["energy"]),
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_redu"]),
            to_numpy_array(ret1["energy_redu"]),
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_r"]),
            to_numpy_array(ret1["energy_derv_r"]),
        )
        ret0 = md0.forward_common_lower(*args, do_atomic_virial=True)
        ret1 = md1.forward_common_lower(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c"]),
            to_numpy_array(ret1["energy_derv_c"]),
        )

    def test_dp_consistency(self):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPInvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        )
        type_map = ["foo", "bar"]
        md0 = DPDPModel(ds, ft, type_map=type_map)
        md1 = DPModel.deserialize(md0.serialize()).to(env.DEVICE)
        args0 = [self.coord_ext, self.atype_ext, self.nlist]
        args1 = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = md0.call_lower(*args0)
        ret1 = md1.forward_common_lower(*args1)
        np.testing.assert_allclose(
            ret0["energy"],
            to_numpy_array(ret1["energy"]),
        )
        np.testing.assert_allclose(
            ret0["energy_redu"],
            to_numpy_array(ret1["energy_redu"]),
        )

    def test_jit(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        # TODO: dirty hack to avoid data stat!!!
        md0 = DPModel(ds, ft, type_map=type_map, resuming=True).to(env.DEVICE)
        torch.jit.script(md0)
