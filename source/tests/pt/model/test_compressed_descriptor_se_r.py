import unittest
import numpy as np
import torch

from typing import (
    Any,
)
from deepmd.pt.utils.env import DEVICE as PT_DEVICE
from deepmd.pt.model.descriptor.se_r import(
    DescrptSeR,
)
from deepmd.pt.utils.nlist import build_neighbor_list as build_neighbor_list_pt
from deepmd.pt.utils.nlist import (
    extend_coord_with_ghosts as extend_coord_with_ghosts_pt,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

def eval_pt_descriptor(
    pt_obj: Any, natoms, coords, atype, box, mixed_types: bool = False
) -> Any:
    ext_coords, ext_atype, mapping = extend_coord_with_ghosts_pt(
        torch.from_numpy(coords).to(PT_DEVICE).reshape(1, -1, 3),
        torch.from_numpy(atype).to(PT_DEVICE).reshape(1, -1),
        torch.from_numpy(box).to(PT_DEVICE).reshape(1, 3, 3),
        pt_obj.get_rcut(),
    )
    nlist = build_neighbor_list_pt(
        ext_coords,
        ext_atype,
        natoms[0],
        pt_obj.get_rcut(),
        pt_obj.get_sel(),
        distinguish_types=(not mixed_types),
    )
    result, _, _, _, _ = pt_obj(ext_coords, ext_atype, nlist, mapping=mapping)
    return result

class TestDescriptorSeR(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        if self.dtype == "float32":
            self.atol = 1e-5
        elif self.dtype == "float64":
            self.atol = 1e-10
        self.seed = 21
        self.sel = [9, 10]
        self.rcut_smth = 5.80
        self.rcut = 6.00
        self.neuron = [6, 12, 24]
        self.ntypes = 2
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

        self.se_r = DescrptSeR(
            self.rcut,
            self.rcut_smth,
            self.sel,
            self.neuron,
            seed=21,
            precision=self.dtype,
        )

    def test_compressed_forward(self):
        result_pt = eval_pt_descriptor(
            self.se_r,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

        self.se_r.enable_compression(1.0)
        result_pt_compressed = eval_pt_descriptor(
            self.se_r,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

        self.assertEqual(result_pt.shape, result_pt_compressed.shape)
        torch.testing.assert_close(
            result_pt,
            result_pt_compressed,
            atol=self.atol,
            rtol=self.atol,
        )

if __name__ == '__main__':
    unittest.main()