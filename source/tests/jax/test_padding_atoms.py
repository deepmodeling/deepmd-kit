# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np
from copy import deepcopy

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
        PropertyFittingNet,
    )
    from deepmd.jax.model.property_model import (
        PropertyModel,
    )

    dtype = jnp.float64


#@unittest.skipIf(
#    sys.version_info < (3, 10),
#    "JAX requires Python 3.10 or later",
#)
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
        self.rcut = 2.2
        self.rcut_smth = 0.4
        self.atol = 1e-12


#@unittest.skipIf(
#    sys.version_info < (3, 10),
#    "JAX requires Python 3.10 or later",
#)
class TestPaddingAtoms(unittest.TestCase, TestCaseSingleFrameWithoutNlist):
    def setUp(self):
        TestCaseSingleFrameWithoutNlist.setUp(self)

    def test_padding_atoms_consistency(self):
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = PropertyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            intensive=True,
        )
        type_map = ["foo", "bar"]
        model = PropertyModel(ds, ft, type_map=type_map)
        args = [to_jax_array(ii) for ii in [self.coord, self.atype, self.cell]]
        ret_base = model.call(*args)


        #np.testing.assert_allclose(
        #    to_numpy_array(ret0[model.get_var_name()]),
        #    to_numpy_array(ret1[md1.get_var_name()]),
        #    atol=self.atol,
        #)
