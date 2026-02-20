# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt_expt.utils.exclude_mask import (
    PairExcludeMask,
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


class TestDescrptSeA(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("idt", [False, True])  # resnet_dt
    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    @pytest.mark.parametrize("em", [[], [[0, 1]], [[1, 1]]])  # exclude_types
    def test_consistency(self, idt, prec, em) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        err_msg = f"idt={idt} prec={prec}"
        dd0 = DescrptSeA(
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
        dd1 = DescrptSeA.deserialize(dd0.serialize())
        rd1, gr1, _, _, sw1 = dd1(
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
        dd2 = DPDescrptSeA.deserialize(dd0.serialize())
        rd2, gr2, _, _, sw2 = dd2.call(
            self.coord_ext,
            self.atype_ext,
            self.nlist,
        )
        for aa, bb in zip([rd1, gr1, sw1], [rd2, gr2, sw2], strict=True):
            np.testing.assert_allclose(
                aa.detach().cpu().numpy(),
                bb,
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
        if em:
            dd1.reinit_exclude([tuple(x) for x in em])
            assert isinstance(dd1.emask, PairExcludeMask)

    @pytest.mark.parametrize("idt", [False, True])  # resnet_dt
    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_exportable(self, idt, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        dd0 = DescrptSeA(
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

    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_make_fx(self, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        dd0 = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)
        dd0.davg = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0 = dd0.eval()
        coord_ext = torch.tensor(self.coord_ext, dtype=dtype, device=self.device)
        atype_ext = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)

        def fn(coord_ext, atype_ext, nlist):
            coord_ext = coord_ext.detach().requires_grad_(True)
            rd = dd0(coord_ext, atype_ext, nlist)[0]
            grad = torch.autograd.grad(rd.sum(), coord_ext, create_graph=False)[0]
            return rd, grad

        rd_eager, grad_eager = fn(coord_ext, atype_ext, nlist)
        traced = make_fx(fn)(coord_ext, atype_ext, nlist)
        rd_traced, grad_traced = traced(coord_ext, atype_ext, nlist)
        np.testing.assert_allclose(
            rd_eager.detach().cpu().numpy(),
            rd_traced.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            grad_eager.detach().cpu().numpy(),
            grad_traced.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
