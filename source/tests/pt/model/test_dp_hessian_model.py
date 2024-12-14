# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model import (
    EnergyModel,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from .test_env_mat import (
    TestCaseSingleFrameWithoutNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestEnergyHessianModel(unittest.TestCase, TestCaseSingleFrameWithoutNlist):
    def setUp(self):
        TestCaseSingleFrameWithoutNlist.setUp(self)

    def test_self_consistency(self):
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
        md0.enable_hessian()
        md1.enable_hessian()
        args = [to_torch_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
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
        np.testing.assert_allclose(
            to_numpy_array(ret0["hessian"]),
            to_numpy_array(ret1["hessian"]),
            atol=self.atol,
        )
        ret0 = md0.forward(*args, do_atomic_virial=True)
        ret1 = md1.forward(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["atom_virial"]),
            to_numpy_array(ret1["atom_virial"]),
            atol=self.atol,
        )

    def test_energy_consistency(self):
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
        md1.enable_hessian()
        args = [to_torch_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
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

    def test_forward_consistency(self):
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
        md0.enable_hessian()
        md1.enable_hessian()
        md0.requires_hessian("energy")
        args = [to_torch_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        ret0 = md0.forward_common(*args)
        ret1 = md1.forward(*args)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"].squeeze()),
            to_numpy_array(ret1["atom_energy"].squeeze()),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_redu"].squeeze()),
            to_numpy_array(ret1["energy"].squeeze()),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_r"].squeeze()),
            to_numpy_array(ret1["force"].squeeze()),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c_redu"].squeeze()),
            to_numpy_array(ret1["virial"].squeeze()),
            atol=self.atol,
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_r_derv_r"].squeeze()),
            to_numpy_array(ret1["hessian"].squeeze()),
            atol=self.atol,
        )
        ret0 = md0.forward_common(*args, do_atomic_virial=True)
        ret1 = md1.forward(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c"].squeeze()),
            to_numpy_array(ret1["atom_virial"].squeeze()),
            atol=self.atol,
        )
