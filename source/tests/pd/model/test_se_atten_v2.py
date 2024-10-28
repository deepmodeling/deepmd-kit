# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import paddle

from deepmd.dpmodel.descriptor.se_atten_v2 import DescrptSeAttenV2 as DPDescrptSeAttenV2
from deepmd.pd.model.descriptor.se_atten_v2 import (
    DescrptSeAttenV2,
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


class TestDescrptSeAttenV2(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ):
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, to, prec, ect in itertools.product(
            [False, True],  # resnet_dt
            [False, True],  # type_one_side
            [
                "float64",
            ],  # precision
            [False, True],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} prec={prec}"

            # dpa1 new impl
            dd0 = DescrptSeAttenV2(
                self.rcut,
                self.rcut_smth,
                self.sel_mix,
                self.nt,
                attn_layer=2,
                precision=prec,
                resnet_dt=idt,
                type_one_side=to,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)
            dd0.se_atten.mean = paddle.to_tensor(davg, dtype=dtype).to(
                device=env.DEVICE
            )
            dd0.se_atten.stddev = paddle.to_tensor(dstd, dtype=dtype).to(
                device=env.DEVICE
            )
            rd0, _, _, _, _ = dd0(
                paddle.to_tensor(self.coord_ext, dtype=dtype).to(device=env.DEVICE),
                paddle.to_tensor(self.atype_ext, dtype="int64").to(device=env.DEVICE),
                paddle.to_tensor(self.nlist, dtype="int64").to(device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptSeAttenV2.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
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
            # dp impl
            dd2 = DPDescrptSeAttenV2.deserialize(dd0.serialize())
            rd2, _, _, _, _ = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd2,
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )

    def test_jit(
        self,
    ):
        rng = np.random.default_rng()
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec, to, ect in itertools.product(
            [
                False,
            ],  # resnet_dt
            [
                "float64",
            ],  # precision
            [
                False,
            ],  # type_one_side
            [False, True],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            # dpa1 new impl
            dd0 = DescrptSeAttenV2(
                self.rcut,
                self.rcut_smth,
                self.sel,
                self.nt,
                precision=prec,
                resnet_dt=idt,
                type_one_side=to,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            )
            dd0.se_atten.mean = paddle.to_tensor(davg, dtype=dtype).to(
                device=env.DEVICE
            )
            dd0.se_atten.dstd = paddle.to_tensor(dstd, dtype=dtype).to(
                device=env.DEVICE
            )
            _ = paddle.jit.to_static(dd0)
