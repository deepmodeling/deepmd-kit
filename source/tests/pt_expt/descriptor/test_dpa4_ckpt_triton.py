# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt DPA4 runtime features: activation checkpointing and Triton fallback.

These features are pt_expt-only (the array-API dpmodel cannot express
``torch.utils.checkpoint`` or Triton kernels).  The Triton kernels themselves
run only on CUDA; on CPU the opt-in path falls back to the eager reference,
which must reproduce the dpmodel dense result bit-for-bit.
"""

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DPDescrptDPA4
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.descriptor.dpa4_nn.block import (
    SeZMInteractionBlock,
)
from deepmd.pt_expt.descriptor.dpa4_nn.so2 import (
    DynamicRadialDegreeMixer,
    SO2Convolution,
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


class TestDPA4RuntimeFeatures(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE
        self.dtype = PRECISION_DICT["float64"]
        # one serialized reference so every descriptor shares identical weights
        self.data = make_descriptor(self.nt, self.sel_mix, self.rcut).serialize()

    def _inputs(self, requires_grad=False):
        coord = torch.tensor(self.coord_ext, dtype=self.dtype, device=self.device)
        if requires_grad:
            coord = coord.detach().requires_grad_(True)
        atype = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)
        return coord, atype, nlist

    def test_activation_checkpoint_matches_eager(self, monkeypatch) -> None:
        # Baseline: eval-time forward + autograd force with checkpointing OFF.
        monkeypatch.delenv("DP_ACT_INFER", raising=False)
        m0 = DescrptDPA4.deserialize(self.data).to(self.device).eval()
        c0, atype, nlist = self._inputs(requires_grad=True)
        out0 = m0(c0, atype, nlist)[0]
        g0 = torch.autograd.grad(out0.sum(), c0)[0]

        # Checkpointing ON: the block must engage recomputation and return the
        # same value and gradient (checkpoint only trades compute for memory).
        monkeypatch.setenv("DP_ACT_INFER", "1")
        m1 = DescrptDPA4.deserialize(self.data).to(self.device).eval()
        block = next(m for m in m1.modules() if isinstance(m, SeZMInteractionBlock))
        assert block._act_infer
        assert block._use_infer_activation_checkpoint(
            torch.zeros(1, requires_grad=True)
        )
        c1, atype, nlist = self._inputs(requires_grad=True)
        out1 = m1(c1, atype, nlist)[0]
        g1 = torch.autograd.grad(out1.sum(), c1)[0]

        np.testing.assert_allclose(
            out1.detach().cpu().numpy(),
            out0.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            g1.detach().cpu().numpy(),
            g0.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_checkpoint_off_when_training(self, monkeypatch) -> None:
        # Training mode never checkpoints (only the eval-time autograd path does).
        monkeypatch.setenv("DP_ACT_INFER", "1")
        m = DescrptDPA4.deserialize(self.data).to(self.device).train()
        block = next(x for x in m.modules() if isinstance(x, SeZMInteractionBlock))
        assert block._act_infer
        assert not block._use_infer_activation_checkpoint(
            torch.zeros(1, requires_grad=True)
        )

    def test_triton_eager_fallback_parity(self, monkeypatch) -> None:
        # With the kernels bound but no CUDA/Triton present, the SO(2) rotation
        # and radial-mix paths fall back to the eager reference, which must equal
        # the dpmodel dense result bit-for-bit.
        monkeypatch.setenv("DP_TRITON_INFER", "1")
        m = DescrptDPA4.deserialize(self.data).to(self.device).eval()
        so2 = next(x for x in m.modules() if isinstance(x, SO2Convolution))
        assert so2.use_triton_infer
        assert so2._rotate_to_local_fn is not None
        assert so2._rotate_back_fn is not None
        mixers = [x for x in m.modules() if isinstance(x, DynamicRadialDegreeMixer)]
        assert mixers and all(x.use_triton_infer for x in mixers)

        coord, atype, nlist = self._inputs()
        out = m(coord, atype, nlist)[0]

        dd = DPDescrptDPA4.deserialize(self.data)
        ref = dd.call(self.coord_ext, self.atype_ext, self.nlist)[0]
        np.testing.assert_allclose(
            out.detach().cpu().numpy(),
            np.asarray(ref),
            rtol=1e-12,
            atol=1e-14,
        )
