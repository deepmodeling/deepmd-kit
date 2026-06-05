# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.descriptor.env_mat_vg import (
    VG_ENV_DIM,
    prod_env_mat_vg,
)
from deepmd.pt.model.descriptor.se_a_vg import (
    DescrptSeAVg,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDescrptSeAVg(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_sigma_zero_matches_se_a_env_mat(self) -> None:
        """At sigma=0 the VG radial kernel reduces to the standard 1/r form."""
        prec = "float64"
        pt_dtype = PRECISION_DICT[prec]
        nf, nloc, nnei = self.nlist.shape
        mean = torch.zeros(
            (self.nt, nnei, VG_ENV_DIM), dtype=pt_dtype, device=env.DEVICE
        )
        stddev = torch.ones(
            (self.nt, nnei, VG_ENV_DIM), dtype=pt_dtype, device=env.DEVICE
        )
        mean_se = torch.zeros((self.nt, nnei, 4), dtype=pt_dtype, device=env.DEVICE)
        stddev_se = torch.ones((self.nt, nnei, 4), dtype=pt_dtype, device=env.DEVICE)

        coord = torch.tensor(self.coord_ext, dtype=pt_dtype, device=env.DEVICE)
        nlist = torch.tensor(self.nlist, dtype=torch.int64, device=env.DEVICE)
        atype = torch.tensor(
            self.atype_ext[:, :nloc], dtype=torch.int64, device=env.DEVICE
        )
        aparam_zero = torch.zeros((nf, nloc, 1), dtype=pt_dtype, device=env.DEVICE)

        vg_mat, _, _ = prod_env_mat_vg(
            coord,
            nlist,
            atype,
            aparam_zero,
            mean,
            stddev,
            self.rcut,
            self.rcut_smth,
        )
        se_mat, _, _ = prod_env_mat(
            coord,
            nlist,
            atype,
            mean_se,
            stddev_se,
            self.rcut,
            self.rcut_smth,
        )
        np.testing.assert_allclose(
            vg_mat[..., :4].detach().cpu().numpy(),
            se_mat.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            vg_mat[..., 4].detach().cpu().numpy(),
            0.0,
            atol=1e-10,
        )

    def test_sigma_changes_descriptor(self) -> None:
        prec = "float64"
        pt_dtype = PRECISION_DICT[prec]
        nf, nloc, _ = self.nlist.shape
        dd = DescrptSeAVg(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        coord = torch.tensor(self.coord_ext, dtype=pt_dtype, device=env.DEVICE)
        atype_ext = torch.tensor(self.atype_ext, dtype=torch.int64, device=env.DEVICE)
        nlist = torch.tensor(self.nlist, dtype=torch.int64, device=env.DEVICE)
        aparam_zero = torch.zeros((nf, nloc, 1), dtype=pt_dtype, device=env.DEVICE)
        aparam_one = torch.ones((nf, nloc, 1), dtype=pt_dtype, device=env.DEVICE)

        out0, _, _, _, _ = dd(coord, atype_ext, nlist, aparam=aparam_zero)
        out1, _, _, _, _ = dd(coord, atype_ext, nlist, aparam=aparam_one)
        diff = (out0 - out1).abs().max().item()
        self.assertGreater(diff, 0.0)

    def test_forward_shape(self) -> None:
        prec = "float64"
        pt_dtype = PRECISION_DICT[prec]
        nf, nloc, nnei = self.nlist.shape
        axis_neuron = 4
        neuron = [8, 16]
        dd = DescrptSeAVg(
            self.rcut,
            self.rcut_smth,
            self.sel,
            neuron=neuron,
            axis_neuron=axis_neuron,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        coord = torch.tensor(self.coord_ext, dtype=pt_dtype, device=env.DEVICE)
        atype_ext = torch.tensor(self.atype_ext, dtype=torch.int64, device=env.DEVICE)
        nlist = torch.tensor(self.nlist, dtype=torch.int64, device=env.DEVICE)
        aparam = torch.full((nf, nloc, 1), 0.5, dtype=pt_dtype, device=env.DEVICE)

        out, rot, _, _, sw = dd(coord, atype_ext, nlist, aparam=aparam)
        self.assertEqual(out.shape, (nf, nloc, neuron[-1] * axis_neuron))
        self.assertEqual(rot.shape, (nf, nloc, neuron[-1], 3))
        self.assertEqual(dd.sea.ndescrpt, nnei * VG_ENV_DIM)
        self.assertIsNotNone(sw)

    def test_compression_matches_uncompressed(self) -> None:
        if not hasattr(torch.ops.deepmd, "tabulate_fusion_se_a"):
            self.skipTest("tabulate_fusion_se_a op is not built")
        prec = "float64"
        pt_dtype = PRECISION_DICT[prec]
        nf, nloc, _ = self.nlist.shape
        dd = DescrptSeAVg(
            self.rcut,
            self.rcut_smth,
            self.sel,
            neuron=[8, 16],
            axis_neuron=4,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        coord = torch.tensor(self.coord_ext, dtype=pt_dtype, device=env.DEVICE)
        atype_ext = torch.tensor(self.atype_ext, dtype=torch.int64, device=env.DEVICE)
        nlist = torch.tensor(self.nlist, dtype=torch.int64, device=env.DEVICE)
        aparam = torch.full((nf, nloc, 1), 0.5, dtype=pt_dtype, device=env.DEVICE)

        out_ref, _, _, _, _ = dd(coord, atype_ext, nlist, aparam=aparam)
        dd.enable_compression(
            min_nbor_dist=0.5,
            table_extrapolate=5.0,
            table_stride_1=0.01,
            table_stride_2=0.1,
        )
        out_cmp, _, _, _, _ = dd(coord, atype_ext, nlist, aparam=aparam)
        np.testing.assert_allclose(
            out_ref.detach().cpu().numpy(),
            out_cmp.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
