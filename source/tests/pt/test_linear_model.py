import unittest
from unittest.mock import (
    patch,
)
import numpy as np
import torch

from deepmd.pt.model.model.linear_model import LinearModel
from deepmd.pt.model.model.pair_tab_model import PairTabModel
from deepmd.pt.model.model.dp_atomic_model import DPAtomicModel

from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.pt.utils import (
    env,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
dtype = env.GLOBAL_PT_FLOAT_PRECISION

class TestWeightCalculation(unittest.TestCase):

        
    @patch("numpy.loadtxt")
    def test_pairwise(self, mock_loadtxt):

        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.05, 1.0, 2.0, 3.0],
                [0.1, 0.8, 1.6, 2.4],
                [0.15, 0.5, 1.0, 1.5],
                [0.2, 0.25, 0.4, 0.75],
                [0.25, 0., 0., 0.],
            ]
        )
        extended_atype = torch.tensor([[0, 0]])
        nlist = torch.tensor([[[1], [-1]]])
        
        ds = DescrptSeA(
            rcut=0.3,
            rcut_smth=0.4,
            sel=[3],
            ).to(env.DEVICE)
        ft = InvarFitting(
                "energy",
                2,
                ds.get_dim_out(),
                1,
                distinguish_types=ds.distinguish_types(),
            ).to(env.DEVICE)
        
        type_map = ["foo", "bar"]
        zbl_model = PairTabModel(tab_file=file_path, rcut=0.3, sel=2)
        dp_model = DPAtomicModel(ds, ft, type_map=type_map, resuming=True).to(env.DEVICE)
        wgt_model = LinearModel(dp_model,zbl_model)
        wgt_res = []
        for dist in  np.linspace(0.05, 0.3, 10):
            
            extended_coord = torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0],
                            [0.0, dist, 0.0],
                        ],
                    ]
                )
            
            wgt_model.forward_atomic(extended_coord, extended_atype, nlist, ra=0.1, rb=0.25)

            wgt_res.append(wgt_model.zbl_weight)
        results = torch.stack(wgt_res).reshape(10,2)
        excepted_res = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.9995, 0.0],
                [0.9236, 0.0],
                [0.6697, 0.0],
                [0.3303, 0.0],
                [0.0764, 0.0],
                [0.0005, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ], dtype=torch.float64)
        torch.testing.assert_close(results, excepted_res, rtol=0.0001, atol=0.0001)

class TestIntegration(unittest.TestCase, TestCaseSingleFrameWithNlist):
    
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt):
        TestCaseSingleFrameWithNlist.setUp(self)
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        )
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
        dp_model = DPAtomicModel(ds, ft, type_map=type_map, resuming=True).to(env.DEVICE)
        zbl_model = PairTabModel(file_path, self.rcut, sum(self.sel))
        self.md0 = LinearModel(dp_model, zbl_model).to(env.DEVICE)
        self.md1 = LinearModel.deserialize(self.md0.serialize()).to(env.DEVICE)

    def test_self_consistency(self):
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = self.md0.forward_atomic(*args, ra=0.2, rb =0.5)
        ret1 = self.md1.forward_atomic(*args, ra=0.2, rb =0.5)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]),
            to_numpy_array(ret1["energy"]),
        )
    # add cross framework consistency check.
    def test_jit(self):
        torch.jit.script(self.md1)
