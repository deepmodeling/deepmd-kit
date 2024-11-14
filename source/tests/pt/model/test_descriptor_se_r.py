# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import DescrptSeR as DPDescrptSeR
from deepmd.pt.model.descriptor.se_r import (
    DescrptSeR,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
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
class TestDescrptSeR(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ) -> None:
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
            # sea new impl
            dd0 = DescrptSeR(
                self.rcut,
                self.rcut_smth,
                self.sel,
                precision=prec,
                resnet_dt=idt,
                exclude_mask=em,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)
            dd0.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)

            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptSeR.deserialize(dd0.serialize())
            rd1, _, _, _, sw1 = dd1(
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
            dd2 = DPDescrptSeR.deserialize(dd0.serialize())
            rd2, _, _, _, sw2 = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
            )
            for aa, bb in zip([rd1, sw1], [rd2, sw2]):
                np.testing.assert_allclose(
                    aa.detach().cpu().numpy(),
                    bb,
                    rtol=rtol,
                    atol=atol,
                    err_msg=err_msg,
                )

    def test_load_stat(self) -> None:
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

            # sea new impl
            dd0 = DescrptSeR(
                self.rcut,
                self.rcut_smth,
                self.sel,
                precision=prec,
                resnet_dt=idt,
                seed=GLOBAL_SEED,
            )
            dd0.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            dd1 = DescrptSeR.deserialize(dd0.serialize())
            dd1.compute_input_stats(
                [
                    {
                        "r0": None,
                        "coord": torch.from_numpy(self.coord_ext)
                        .reshape(-1, self.nall, 3)
                        .to(env.DEVICE),
                        "atype": torch.from_numpy(self.atype_ext).to(env.DEVICE),
                        "box": None,
                        "natoms": self.nall,
                    }
                ]
            )

            with self.assertRaises(ValueError) as cm:
                ev = EnvMatStatSe(dd1)
                ev.last_dim = 3
                ev.load_or_compute_stats([])
            self.assertEqual(
                "last_dim should be 1 for raial-only or 4 for full descriptor.",
                str(cm.exception),
            )

    def test_jit(
        self,
    ) -> None:
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

            # sea new impl
            dd0 = DescrptSeR(
                self.rcut,
                self.rcut_smth,
                self.sel,
                precision=prec,
                resnet_dt=idt,
                seed=GLOBAL_SEED,
            )
            dd0.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            dd1 = DescrptSeR.deserialize(dd0.serialize())
            torch.jit.script(dd0)
            torch.jit.script(dd1)
