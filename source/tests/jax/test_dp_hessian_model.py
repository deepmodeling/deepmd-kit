# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)

if sys.version_info >= (3, 10):
    from deepmd.jax.common import (
        to_jax_array,
    )
    from deepmd.jax.descriptor.se_e2_a import (
        DescrptSeA,
    )
    from deepmd.jax.env import (
        jnp,
    )
    from deepmd.jax.fitting.fitting import (
        EnergyFittingNet,
    )
    from deepmd.jax.model.ener_model import (
        EnergyModel,
    )

    dtype = jnp.float64


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestCaseSingleFrameWithoutNlist:
    def setUp(self) -> None:
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nf, self.nt = 1, 2
        self.coord = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ).reshape([1, self.nloc * 3])
        self.atype = np.array([0, 0, 1], dtype=int).reshape([1, self.nloc])
        self.cell = 2.0 * np.eye(3).reshape([1, 9])
        # sel = [5, 2]
        self.sel = [16, 8]
        self.sel_mix = [24]
        self.natoms = [3, 3, 2, 1]
        self.rcut = 2.2
        self.rcut_smth = 0.4
        self.atol = 1e-12


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestEnergyHessianModel(unittest.TestCase, TestCaseSingleFrameWithoutNlist):
    def setUp(self):
        TestCaseSingleFrameWithoutNlist.setUp(self)

    def test_self_consistency(self):
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]
        md0 = EnergyModel(ds, ft, type_map=type_map)
        md1 = EnergyModel.deserialize(md0.serialize())
        md0.enable_hessian()
        md1.enable_hessian()
        args = [to_jax_array(ii) for ii in [self.coord, self.atype, self.cell]]
        ret0 = md0.call(*args)
        ret1 = md1.call(*args)
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
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_r_derv_r"]),
            to_numpy_array(ret1["energy_derv_r_derv_r"]),
            atol=self.atol,
        )
        ret0 = md0.call(*args, do_atomic_virial=True)
        ret1 = md1.call(*args, do_atomic_virial=True)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy_derv_c"]),
            to_numpy_array(ret1["energy_derv_c"]),
            atol=self.atol,
        )
