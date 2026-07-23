# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the fused Triton environment-matrix kernel.

The kernel (:mod:`deepmd.kernels.triton.env_mat`) is a drop-in for the
descriptors' ``prod_env_mat`` front end under ``DP_TRITON_INFER >= 1`` on CUDA.
These tests check, against the eager reference path (level 0):

* forward parity of ``(env_mat, diff, switch)`` in fp32 and fp64, for the full
  and radial-only outputs and both smooth switches;
* force parity, i.e. the coordinate gradient produced by the registered
  closed-form backward;
* ``NaN``-safety of the exponential switch backward in fp32 (the factored
  ``-a e w`` overflows; the kernel uses the fused ``-a exp(xarg - e)`` form);
* composability under ``make_fx`` (the operator is captured as one opaque node).
"""

import os
import unittest

import torch

from deepmd.kernels.triton.env_mat import (
    TRITON_AVAILABLE,
    env_mat,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)

_SKIP = not (torch.cuda.is_available() and TRITON_AVAILABLE)


def _set_level(level: int) -> None:
    os.environ["DP_TRITON_INFER"] = str(level)


def _make_system(dtype, seed, nf=2, nloc=48, nnei=64, nall=160, ntypes=3):
    """Random extended system with a self-free, partially padded neighbor list."""
    g = torch.Generator(device=env.DEVICE).manual_seed(seed)
    coord = torch.rand(nf, nall, 3, generator=g, device=env.DEVICE, dtype=dtype) * 9.0
    nlist = torch.randint(0, nall, (nf, nloc, nnei), generator=g, device=env.DEVICE)
    center = torch.arange(nloc, device=env.DEVICE)[None, :, None]
    drop = (torch.rand(nf, nloc, nnei, generator=g, device=env.DEVICE) < 0.2) | (
        nlist == center
    )
    nlist = torch.where(drop, torch.full_like(nlist, -1), nlist)
    atype = torch.randint(0, ntypes, (nf, nloc), generator=g, device=env.DEVICE)
    return coord, nlist, atype, ntypes


@unittest.skipIf(_SKIP, "CUDA and Triton are required for the env_mat kernel")
class TestEnvMatTriton(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("DP_TRITON_INFER", None)

    def _eager(self, coord, nlist, atype, mean, std, rcut, rsmth, radial, prot, ue):
        _set_level(0)
        c = coord.clone().reshape(coord.shape[0], -1).requires_grad_()
        e, d, s = prod_env_mat(
            c,
            nlist,
            atype,
            mean,
            std,
            rcut,
            rsmth,
            radial_only=radial,
            protection=prot,
            use_exp_switch=ue,
        )
        return c, e, d, s

    def _triton(self, coord, nlist, atype, mean, std, rcut, rsmth, radial, prot, ue):
        _set_level(1)
        c = coord.clone().requires_grad_()
        e, d, s = env_mat(
            c,
            nlist,
            atype,
            mean,
            std,
            rcut,
            rsmth,
            radial_only=radial,
            protection=prot,
            use_exp_switch=ue,
        )
        return c, e, d, s

    def test_forward_and_force_parity(self) -> None:
        rcut, rsmth = 6.0, 2.0  # representable scalars -> fp64 stays exact
        for dtype in (torch.float32, torch.float64):
            for radial in (False, True):
                for ue in (False, True):
                    with self.subTest(dtype=dtype, radial=radial, use_exp=ue):
                        ch = 1 if radial else 4
                        coord, nlist, atype, nt = _make_system(dtype, GLOBAL_SEED)
                        nnei = nlist.shape[2]
                        gg = torch.Generator(device=env.DEVICE).manual_seed(1)
                        mean = torch.randn(
                            nt, nnei, ch, generator=gg, device=env.DEVICE, dtype=dtype
                        )
                        std = 0.5 + torch.rand(
                            nt, nnei, ch, generator=gg, device=env.DEVICE, dtype=dtype
                        )
                        ge = torch.randn(
                            *(coord.shape[0], nlist.shape[1], nnei, ch),
                            generator=gg,
                            device=env.DEVICE,
                            dtype=dtype,
                        )
                        gd = torch.randn(
                            *(coord.shape[0], nlist.shape[1], nnei, 3),
                            generator=gg,
                            device=env.DEVICE,
                            dtype=dtype,
                        )
                        gs = torch.randn(
                            *(coord.shape[0], nlist.shape[1], nnei, 1),
                            generator=gg,
                            device=env.DEVICE,
                            dtype=dtype,
                        )

                        c1, e1, d1, s1 = self._triton(
                            coord, nlist, atype, mean, std, rcut, rsmth, radial, 0.0, ue
                        )
                        (g1,) = torch.autograd.grad(
                            (e1 * ge).sum() + (d1 * gd).sum() + (s1 * gs).sum(), c1
                        )
                        c0, e0, d0, s0 = self._eager(
                            coord, nlist, atype, mean, std, rcut, rsmth, radial, 0.0, ue
                        )
                        (g0,) = torch.autograd.grad(
                            (e0 * ge).sum() + (d0 * gd).sum() + (s0 * gs).sum(), c0
                        )

                        ftol = 1e-4 if dtype == torch.float32 else 1e-10
                        gtol = 1e-3 if dtype == torch.float32 else 1e-8
                        self.assertLess((e1 - e0).abs().max().item(), ftol)
                        self.assertLess((d1 - d0).abs().max().item(), ftol)
                        self.assertLess((s1 - s0).abs().max().item(), ftol)
                        gscale = g0.abs().max().item() + 1e-30
                        self.assertLess(
                            (g1 - g0.reshape_as(g1)).abs().max().item() / gscale, gtol
                        )

    def test_exp_switch_backward_is_finite_fp32(self) -> None:
        # Extreme window (a = 20 / rcut_smth large): the factored switch
        # derivative overflows in fp32; the kernel must stay finite.
        _set_level(1)
        coord, nlist, atype, nt = _make_system(torch.float32, GLOBAL_SEED)
        nnei = nlist.shape[2]
        mean = torch.zeros(nt, nnei, 4, device=env.DEVICE)
        std = torch.ones(nt, nnei, 4, device=env.DEVICE)
        c = coord.clone().requires_grad_()
        e, d, s = env_mat(c, nlist, atype, mean, std, 6.0, 1.0, use_exp_switch=True)
        (g,) = torch.autograd.grad(e.sum() + d.sum() + s.sum(), c)
        self.assertTrue(torch.isfinite(g).all().item())

    def test_make_fx_opaque_operator(self) -> None:
        from torch.fx.experimental.proxy_tensor import (
            make_fx,
        )

        _set_level(1)
        coord, nlist, atype, nt = _make_system(torch.float32, GLOBAL_SEED)
        nnei = nlist.shape[2]
        mean = torch.zeros(nt, nnei, 4, device=env.DEVICE)
        std = torch.ones(nt, nnei, 4, device=env.DEVICE)

        def fn(c, nl, at, mn, sd):
            e, d, s = env_mat(c, nl, at, mn, sd, 6.0, 2.0)
            return e.sum() + d.sum() + s.sum()

        gm = make_fx(fn, tracing_mode="fake")(coord, nlist, atype, mean, std)
        targets = [str(n.target) for n in gm.graph.nodes]
        self.assertTrue(any("env_mat.default" in t for t in targets))


if __name__ == "__main__":
    unittest.main()
