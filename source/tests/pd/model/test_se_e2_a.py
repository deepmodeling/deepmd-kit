# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import paddle

from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.pd.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    PRECISION_DICT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PD_FLOAT_PRECISION


# to be merged with the tf test case
class TestDescrptSeA(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ):
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
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
                exclude_types=em,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)
            dd0.sea.mean = paddle.to_tensor(davg, dtype=dtype).to(device=env.DEVICE)
            dd0.sea.dstd = paddle.to_tensor(dstd, dtype=dtype).to(device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                paddle.to_tensor(self.coord_ext, dtype=dtype).to(device=env.DEVICE),
                paddle.to_tensor(self.atype_ext, dtype="int64").to(device=env.DEVICE),
                paddle.to_tensor(self.nlist, dtype="int64").to(device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptSeA.deserialize(dd0.serialize())
            rd1, gr1, _, _, sw1 = dd1(
                paddle.to_tensor(self.coord_ext, dtype=dtype).to(device=env.DEVICE),
                paddle.to_tensor(self.atype_ext, dtype="int64").to(device=env.DEVICE),
                paddle.to_tensor(self.nlist, dtype="int64").to(device=env.DEVICE),
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy()[0][self.perm[: self.nloc]],
                rd0.detach().cpu().numpy()[1],
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
            # dp impl
            dd2 = DPDescrptSeA.deserialize(dd0.serialize())
            rd2, gr2, _, _, sw2 = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
            )
            for aa, bb in zip([rd1, gr1, sw1], [rd2, gr2, sw2]):
                np.testing.assert_allclose(
                    aa.detach().cpu().numpy(),
                    bb,
                    rtol=rtol,
                    atol=atol,
                    err_msg=err_msg,
                )

    def test_jit(
        self,
    ):
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec in itertools.product(
            [False, True],
            ["float64", "float32"],
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
                seed=GLOBAL_SEED,
            )
            dd0.sea.mean = paddle.to_tensor(davg, dtype=dtype).to(device=env.DEVICE)
            dd0.sea.dstd = paddle.to_tensor(dstd, dtype=dtype).to(device=env.DEVICE)
            dd1 = DescrptSeA.deserialize(dd0.serialize())
            model = paddle.jit.to_static(dd0)
            model = paddle.jit.to_static(dd1)
