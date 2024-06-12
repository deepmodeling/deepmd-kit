# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np
import torch

from deepmd.dpmodel.atomic_model import (
    DPMultiFittingAtomicModel as DPDPMultiFittingAtomicModel,
)
from deepmd.pt.model.atomic_model import (
    DPMultiFittingAtomicModel,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA1,
)
from deepmd.pt.model.model import (
    DPMultiFittingModel,
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
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestIntegration(unittest.TestCase, TestCaseSingleFrameWithNlist):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt):
        TestCaseSingleFrameWithNlist.setUp(self)
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        )
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft_dict = {
            "type": "test_multi_fitting",
            "ener_1": InvarFitting(
                "energy",
                self.nt,
                ds.get_dim_out(),
                1,
                mixed_types=ds.mixed_types(),
            ).to(env.DEVICE),
            "ener_2": InvarFitting(
                "energy",
                self.nt,
                ds.get_dim_out(),
                1,
                mixed_types=ds.mixed_types(),
            ).to(env.DEVICE),
        }
        type_map = ["foo", "bar"]
        self.md0 = DPMultiFittingAtomicModel(ds, ft_dict, type_map=type_map).to(
            env.DEVICE
        )
        self.md1 = DPMultiFittingAtomicModel.deserialize(self.md0.serialize()).to(
            env.DEVICE
        )
        self.md2 = DPDPMultiFittingAtomicModel.deserialize(self.md0.serialize())
        self.md3 = DPMultiFittingModel(
            descriptor=ds, fitting_dict=ft_dict, type_map=type_map
        )

    def test_self_consistency(self):
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = self.md0.forward_atomic(*args)
        ret1 = self.md1.forward_atomic(*args)
        ret2 = self.md2.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        np.testing.assert_allclose(
            to_numpy_array(ret0["ener_1"]),
            to_numpy_array(ret1["ener_1"]),
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["ener_2"]),
            to_numpy_array(ret1["ener_2"]),
        )

        np.testing.assert_allclose(
            to_numpy_array(ret0["ener_1"]), ret2["ener_1"], atol=0.001, rtol=0.001
        )
        np.testing.assert_allclose(
            to_numpy_array(ret0["ener_2"]), ret2["ener_2"], atol=0.001, rtol=0.001
        )

    def test_jit(self):
        md1 = torch.jit.script(self.md1)
        md3 = torch.jit.script(self.md3)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
