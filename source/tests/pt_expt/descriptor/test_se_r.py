# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import DescrptSeR as DPDescrptSeR
from deepmd.pt_expt.descriptor.se_r import (
    DescrptSeR,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)

from ...pt.model.test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from ...pt.model.test_mlp import (
    get_tols,
)
from ...seed import (
    GLOBAL_SEED,
)


class TestDescrptSeR(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def test_consistency(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
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
            dd0 = DescrptSeR(
                self.rcut,
                self.rcut_smth,
                self.sel,
                precision=prec,
                resnet_dt=idt,
                exclude_types=em,
                seed=GLOBAL_SEED,
            ).to(self.device)
            dd0.davg = torch.tensor(davg, dtype=dtype, device=self.device)
            dd0.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)

            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
                torch.tensor(self.atype_ext, dtype=int, device=self.device),
                torch.tensor(self.nlist, dtype=int, device=self.device),
            )
            dd1 = DescrptSeR.deserialize(dd0.serialize())
            rd1, _, _, _, sw1 = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
                torch.tensor(self.atype_ext, dtype=int, device=self.device),
                torch.tensor(self.nlist, dtype=int, device=self.device),
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
            dd2 = DPDescrptSeR.deserialize(dd0.serialize())
            rd2, _, _, _, sw2 = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
            )
            for aa, bb in zip([rd1, sw1], [rd2, sw2], strict=True):
                np.testing.assert_allclose(
                    aa.detach().cpu().numpy(),
                    bb,
                    rtol=rtol,
                    atol=atol,
                    err_msg=err_msg,
                )

    def test_exportable(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 1))
        dstd = rng.normal(size=(self.nt, nnei, 1))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec in itertools.product(
            [False, True],
            ["float64", "float32"],
        ):
            dtype = PRECISION_DICT[prec]
            dd0 = DescrptSeR(
                self.rcut,
                self.rcut_smth,
                self.sel,
                precision=prec,
                resnet_dt=idt,
                seed=GLOBAL_SEED,
            ).to(self.device)
            dd0.davg = torch.tensor(davg, dtype=dtype, device=self.device)
            dd0.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)
            dd0 = dd0.eval()
            inputs = (
                torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
                torch.tensor(self.atype_ext, dtype=int, device=self.device),
                torch.tensor(self.nlist, dtype=int, device=self.device),
            )
            torch.export.export(dd0, inputs)
