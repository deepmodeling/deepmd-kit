# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.fitting import DipoleFitting as DPDipoleFitting
from deepmd.pt.model.task.dipole import DipoleFittingNet
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestInvarFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        rd0, gr, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
        )
        atype = torch.tensor(self.atype_ext[:, :nloc], dtype=int, device=env.DEVICE)

        for distinguish_types, nfp, nap in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
        ):
            ft0 = DipoleFittingNet(
                "foo",
                self.nt,
                dd0.dim_out,
                3,
                dim_rot_mat=100,
                numb_fparam=nfp,
                numb_aparam=nap,
                use_tebd=(not distinguish_types),
            ).to(env.DEVICE)
            ft1 = DPDipoleFitting.deserialize(ft0.serialize())
            ft2 = DipoleFittingNet.deserialize(ft1.serialize())

            if nfp > 0:
                ifp = torch.tensor(
                    rng.normal(size=(self.nf, nfp)), dtype=dtype, device=env.DEVICE
                )
            else:
                ifp = None
            if nap > 0:
                iap = torch.tensor(
                    rng.normal(size=(self.nf, self.nloc, nap)),
                    dtype=dtype,
                    device=env.DEVICE,
                )
            else:
                iap = None

            ret0 = ft0(rd0, atype, gr, fparam=ifp, aparam=iap)
            ret1 = ft1(
                rd0.detach().cpu().numpy(),
                atype.detach().cpu().numpy(),
                gr.detach().cpu().numpy(),
                fparam=to_numpy_array(ifp),
                aparam=to_numpy_array(iap),
            )
            ret2 = ft2(rd0, atype, gr, fparam=ifp, aparam=iap)
            np.testing.assert_allclose(
                to_numpy_array(ret0["foo"]),
                ret1["foo"],
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["foo"]),
                to_numpy_array(ret2["foo"]),
            )


    def test_jit(
        self,
    ):
        for od, distinguish_types, nfp, nap in itertools.product(
            [1, 3],
            [True, False],
            [0, 3],
            [0, 4],
        ):
            ft0 = DipoleFittingNet(
                "foo",
                self.nt,
                9,
                od,
                dim_rot_mat=100,
                numb_fparam=nfp,
                numb_aparam=nap,
                use_tebd=(not distinguish_types),
            ).to(env.DEVICE)
            torch.jit.script(ft0)

