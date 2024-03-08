# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.pt.model.task.polarizability import (
    PolarFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.tf.fit.polar import (
    PolarFittingSeA,
)


class TestConsistency(unittest.TestCase):
    def setUp(self) -> None:
        types = torch.randint(0, 4, (1, 5), device=env.DEVICE)
        types = torch.cat((types, types, types), dim=0)
        types[:,-1] = 3
        ntypes = 4
        atomic_polarizability = torch.rand((3, 5, 9), device=env.DEVICE)
        polarizability = torch.rand((3, 9), device=env.DEVICE)
        find_polarizability = torch.rand(1, device=env.DEVICE)
        find_atomic_polarizability = torch.rand(1, device=env.DEVICE)
        self.sampled = [
            {
                "type": types,
                "find_atomic_polarizability": find_atomic_polarizability,
                "atomic_polarizability": atomic_polarizability,
                "polarizability": polarizability,
                "find_polarizability": find_polarizability,
            }
        ]
        self.all_stat = {
            k: [v.numpy(force=True)] for d in self.sampled for k, v in d.items()
        }
        self.tfpolar = PolarFittingSeA(
            ntypes=ntypes,
            dim_descrpt=1,
            embedding_width=1,
            sel_type=list(range(ntypes)),
        )
        self.ptpolar = PolarFittingNet(
            ntypes=ntypes,
            dim_descrpt=1,
            embedding_width=1,
        )

    def test_atomic_consistency(self):
        self.tfpolar.compute_output_stats(self.all_stat)
        tfbias = self.tfpolar.constant_matrix
        self.ptpolar.compute_output_stats(self.sampled)
        ptbias = self.ptpolar.constant_matrix
        np.testing.assert_allclose(tfbias, to_numpy_array(ptbias))

    def test_global_consistency(self):
        self.sampled[0]["find_atomic_polarizability"] = -1
        self.sampled[0]["polarizability"] = self.sampled[0][
            "atomic_polarizability"
        ].sum(dim=1)
        self.all_stat["find_atomic_polarizability"] = [-1]
        self.all_stat["polarizability"] = [
            self.all_stat["atomic_polarizability"][0].sum(axis=1)
        ]
        self.tfpolar.compute_output_stats(self.all_stat)
        tfbias = self.tfpolar.constant_matrix
        self.ptpolar.compute_output_stats(self.sampled)
        ptbias = self.ptpolar.constant_matrix
        np.testing.assert_allclose(tfbias, to_numpy_array(ptbias), rtol=1e-5, atol=1e-5)
