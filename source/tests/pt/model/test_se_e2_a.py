# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
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
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


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
                old_impl=False,
                exclude_types=em,
            ).to(env.DEVICE)
            dd0.sea.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.sea.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptSeA.deserialize(dd0.serialize())
            rd1, gr1, _, _, sw1 = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
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
            # old impl
            if idt is False and prec == "float64" and em == []:
                dd3 = DescrptSeA(
                    self.rcut,
                    self.rcut_smth,
                    self.sel,
                    precision=prec,
                    resnet_dt=idt,
                    old_impl=True,
                ).to(env.DEVICE)
                dd0_state_dict = dd0.sea.state_dict()
                dd3_state_dict = dd3.sea.state_dict()
                for i in dd3_state_dict:
                    dd3_state_dict[i] = (
                        dd0_state_dict[
                            i.replace(".deep_layers.", ".layers.").replace(
                                "filter_layers_old.", "filter_layers.networks."
                            )
                        ]
                        .detach()
                        .clone()
                    )
                    if ".bias" in i:
                        dd3_state_dict[i] = dd3_state_dict[i].unsqueeze(0)
                dd3.sea.load_state_dict(dd3_state_dict)

                rd3, gr3, _, _, sw3 = dd3(
                    torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                    torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                    torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                )
                for aa, bb in zip([rd1, gr1, sw1], [rd3, gr3, sw3]):
                    np.testing.assert_allclose(
                        aa.detach().cpu().numpy(),
                        bb.detach().cpu().numpy(),
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
                old_impl=False,
            )
            dd0.sea.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.sea.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            dd1 = DescrptSeA.deserialize(dd0.serialize())
            model = torch.jit.script(dd0)
            model = torch.jit.script(dd1)
