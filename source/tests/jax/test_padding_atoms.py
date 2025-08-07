# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest
from copy import (
    deepcopy,
)

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
        PropertyFittingNet,
    )
    from deepmd.jax.model.property_model import (
        PropertyModel,
    )

    dtype = jnp.float64


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestCaseSingleFrameWithoutNlist:
    def setUp(self) -> None:
        # nf=2, nloc == 3
        self.nloc = 3
        self.nt = 2
        self.coord = np.array(
            [
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                [
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 0],
                ],
            ],
            dtype=np.float64,
        )
        self.atype = np.array([[0, 0, 1], [1, 1, 0]], dtype=int).reshape([2, self.nloc])
        self.cell = 2.0 * np.eye(3).reshape([1, 9])
        self.cell = np.array([self.cell, self.cell]).reshape(2, 9)
        self.sel = [16, 8]
        self.rcut = 2.2
        self.rcut_smth = 0.4
        self.atol = 1e-12


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
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
        var_name = model.get_var_name()
        args = [to_jax_array(ii) for ii in [self.coord, self.atype, self.cell]]
        result = model.call(*args)
        # test intensive
        np.testing.assert_allclose(
            to_numpy_array(result[f"{var_name}_redu"]),
            np.mean(to_numpy_array(result[f"{var_name}"]), axis=1),
            atol=self.atol,
        )
        # test padding atoms
        padding_atoms_list = [1, 5, 10]
        for padding_atoms in padding_atoms_list:
            coord = deepcopy(self.coord)
            atype = deepcopy(self.atype)
            atype_padding = np.pad(
                atype,
                pad_width=((0, 0), (0, padding_atoms)),
                mode="constant",
                constant_values=-1,
            )
            coord_padding = np.pad(
                coord,
                pad_width=((0, 0), (0, padding_atoms), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            args = [
                to_jax_array(ii) for ii in [coord_padding, atype_padding, self.cell]
            ]
            result_padding = model.call(*args)
            np.testing.assert_allclose(
                to_numpy_array(result[f"{var_name}_redu"]),
                to_numpy_array(result_padding[f"{var_name}_redu"]),
                atol=self.atol,
            )


if __name__ == "__main__":
    unittest.main()
