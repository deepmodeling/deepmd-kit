import itertools
import unittest

import numpy as np
import torch

from deepmd.pt.utils.tabulate import (
    DPTabulate,
)

class TestCaseSingleFrameWithNlist:
    def setUp(self):
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 2, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall, 3])
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        self.mapping = np.array([0, 1, 2, 0], dtype=int).reshape([1, self.nall])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.sel_mix = [7]
        self.natoms = [3, 3, 2, 1]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut = 2.2
        self.rcut_smth = 0.4
        # permutations
        self.perm = np.array([2, 0, 1, 3], dtype=np.int32)
        inv_perm = np.array([1, 2, 0, 3], dtype=np.int32)
        # permute the coord and atype
        self.coord_ext = np.concatenate(
            [self.coord_ext, self.coord_ext[:, self.perm, :]], axis=0
        ).reshape(self.nf, self.nall * 3)
        self.atype_ext = np.concatenate(
            [self.atype_ext, self.atype_ext[:, self.perm]], axis=0
        )
        self.mapping = np.concatenate(
            [self.mapping, self.mapping[:, self.perm]], axis=0
        )

        # permute the nlist
        nlist1 = self.nlist[:, self.perm[: self.nloc], :]
        mask = nlist1 == -1
        nlist1 = inv_perm[nlist1]
        nlist1 = np.where(mask, -1, nlist1)
        self.nlist = np.concatenate([self.nlist, nlist1], axis=0)
        self.atol = 1e-12

from deepmd.pt.model.descriptor.se_a import(
    DescrptSeA,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from test_mlp import (
    get_tols,
)

class TestDescriptorSeA(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        self.device = "cpu"
        TestCaseSingleFrameWithNlist.setUp(self)
    
    def test_compression(self,):
        print(self.coord_ext)
        return
        rng = np.random.default_rng(21)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 1))
        dstd = rng.normal(size=(self.nt, nnei, 1))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec, em in itertools.product(
            [False, True],
            ["float64", "float32"],
            [[], [[0, 1]], [[1, 1]]],
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} prec={prec}"
            # sea new impl
            dd0 = DescrptSeA(
                self.rcut,
                self.rcut_smth,
                self.sel,
                precision=prec,
                resnet_dt=idt,
                old_impl=False,
                exclude_types=em,
            ).to(self.device)
            dd0.mean = torch.tensor(davg, dtype=dtype, device=self.device)
            dd0.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)

            n_result_0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
                torch.tensor(self.atype_ext, dtype=int, device=self.device),
                torch.tensor(self.nlist, dtype=int, device=self.device),
            )
            print("n_result_0 shape: ", n_result_0.shape)

            dd1 = DescrptSeA(
                self.rcut,
                self.rcut_smth,
                self.sel,
                precision=prec,
                resnet_dt=idt,
                old_impl=False,
                exclude_types=em,
            ).to(self.device)
            dd1.mean = torch.tensor(davg, dtype=dtype, device=self.device)
            dd1.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)

            print("ntypes: ", dd1.get_ntypes())
            print("neuron: ", dd1.serialize()["neuron"])
            # print(dd1.serialize())

            embedding_net_nodes = dd1.serialize()["embeddings"]["networks"]
            print("embedding networks----------------------------")
            print(len(embedding_net_nodes))
            # print(embedding_net_nodes)
            embedding_net_1 = embedding_net_nodes[0]
            # print(embedding_net_1)
            print("layers----------------------------------------")
            print(len(embedding_net_1["layers"]))           
            print(embedding_net_1["layers"])
            print("bias------------------------------------------")
            print(embedding_net_1["layers"][0]["@variables"]["b"])
            

            dd1.enable_compression(1.0)
            print("enable successful")
            print("lower: ", dd1.lower)
            print("upper: ", dd1.upper)

            

            n_result_1, _, _, _, _ = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
                torch.tensor(self.atype_ext, dtype=int, device=self.device),
                torch.tensor(self.nlist, dtype=int, device=self.device),
            )
            print("n_result_1 shape: ", n_result_1.shape)
            self.assertEqual(n_result_0.shape, n_result_1.shape)

            torch.testing.assert_close(
                n_result_0,
                n_result_1,
                atol=atol,
                rtol=atol,
            )
            return 


if __name__ == '__main__':
    unittest.main()