# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DPDescrptDPA4
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)

from ...common.test_mixins import (
    TestCaseSingleFrameWithNlist,
)


def make_descriptor(nt, sel, rcut, **overrides) -> DescrptDPA4:
    kwargs = {
        "ntypes": nt,
        "sel": sel,
        "rcut": rcut,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "grid_branch": [1, 1, 1],
        "s2_activation": [False, True],
        "random_gamma": False,
        "precision": "float64",
        "seed": 7,
    }
    kwargs.update(overrides)
    return DescrptDPA4(**kwargs)


class TestDescrptDPA4(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("use_env_seed", [True, False])  # env seed feature
    @pytest.mark.parametrize("use_mapping", [True, False])  # pass mapping vs None
    def test_consistency(self, use_env_seed, use_mapping) -> None:
        dtype = PRECISION_DICT["float64"]
        err_msg = f"use_env_seed={use_env_seed} use_mapping={use_mapping}"
        dd0 = make_descriptor(
            self.nt,
            self.sel_mix,
            self.rcut,
            use_env_seed=use_env_seed,
        ).to(self.device)
        coord_ext = torch.tensor(self.coord_ext, dtype=dtype, device=self.device)
        atype_ext = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)
        mapping = (
            torch.tensor(self.mapping, dtype=int, device=self.device)
            if use_mapping
            else None
        )
        rd0 = dd0(coord_ext, atype_ext, nlist, mapping)[0]
        # serialization round-trip within pt_expt
        dd1 = DescrptDPA4.deserialize(dd0.serialize())
        rd1 = dd1(coord_ext, atype_ext, nlist, mapping)[0]
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd1.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
            err_msg=err_msg,
        )
        # dpmodel (numpy) impl
        dd2 = DPDescrptDPA4.deserialize(dd0.serialize())
        rd2 = dd2.call(
            self.coord_ext,
            self.atype_ext,
            self.nlist,
            mapping=self.mapping if use_mapping else None,
        )[0]
        # CPU: strict same-math parity; CUDA: ULP / nondeterministic reduction slack
        if self.device == "cpu" or str(self.device) == "cpu":
            rtol, atol = 1e-12, 1e-14
        else:
            rtol, atol = 1e-10, 1e-12
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd2,
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )

    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_exportable(self, prec) -> None:
        dtype = PRECISION_DICT[prec]
        dd0 = make_descriptor(self.nt, self.sel_mix, self.rcut, precision=prec).to(
            self.device
        )
        dd0 = dd0.eval()
        inputs = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        torch.export.export(dd0, inputs)

    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_make_fx(self, prec) -> None:
        dtype = PRECISION_DICT[prec]
        dd0 = make_descriptor(self.nt, self.sel_mix, self.rcut, precision=prec).to(
            self.device
        )
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
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            grad_eager.detach().cpu().numpy(),
            grad_traced.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
