# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the block-diagonal Triton SO(2)/Wigner rotation kernels
(opt-in via ``DP_TRITON_INFER``).

Two properties are checked against the eager PyTorch reference:

1. Numerical correctness of ``rotate_to_local`` / ``rotate_back`` (forward and
   backward) across ``lmax`` 2-5 with ``mmax == 1`` -- the only layout the block
   kernels accept. The Wigner-D is block-diagonal by ``l``, so the kernel touches
   only the structural non-zeros; the gradient w.r.t. the Wigner therefore matches
   the reference on the block entries (the off-block reference gradient is
   structurally discarded by the model, which builds the Wigner with zero
   off-block entries).
2. ``torch.compile`` composability: gradients through the functional
   ``custom_op`` must match the eager reference when the op is traced under
   ``make_fx`` -- the autograd path that compiled inference uses to obtain
   forces.
"""

import unittest

import torch

from deepmd.pt.model.descriptor.sezm_nn.indexing import (
    build_m_major_index,
    get_so3_dim_of_lmax,
)
from deepmd.pt.model.descriptor.sezm_nn.triton.so2_rotation import (
    TRITON_ROTATION_AVAILABLE,
    rotate_back,
    rotate_back_reference,
    rotate_to_local,
    rotate_to_local_reference,
)

_CUDA = torch.cuda.is_available()


def _block_diagonal_wigner(n_edge, lmax, device, dtype, generator):
    """Random Wigner-D that is block-diagonal by ``l`` (block ``l`` occupies
    rows/cols ``[l**2 : (l+1)**2]``); off-block entries are exactly zero.
    """
    dim = get_so3_dim_of_lmax(lmax)
    wigner = torch.zeros(n_edge, dim, dim, device=device, dtype=dtype)
    for ll in range(lmax + 1):
        start, end = ll * ll, (ll + 1) ** 2
        wigner[:, start:end, start:end] = torch.randn(
            n_edge,
            end - start,
            end - start,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    return wigner


def _block_mask(lmax, device):
    dim = get_so3_dim_of_lmax(lmax)
    mask = torch.zeros(dim, dim, dtype=torch.bool, device=device)
    for ll in range(lmax + 1):
        start, end = ll * ll, (ll + 1) ** 2
        mask[start:end, start:end] = True
    return mask


@unittest.skipIf(not _CUDA, "CUDA is required for the Triton rotation kernels")
@unittest.skipIf(not TRITON_ROTATION_AVAILABLE, "Triton is not available")
class TestSeZMTritonRotation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.dtype = torch.float32
        self.n_node, self.n_edge, self.channels = 64, 2000, 16
        self.tol = {"rtol": 2e-4, "atol": 2e-4}

    def _inputs(self, lmax, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        dim = get_so3_dim_of_lmax(lmax)
        coeff_index = build_m_major_index(lmax, 1, device=self.device)
        x = torch.randn(
            self.n_node,
            dim,
            self.channels,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        src = torch.randint(
            0, self.n_node, (self.n_edge,), device=self.device, generator=gen
        )
        wigner = _block_diagonal_wigner(self.n_edge, lmax, self.device, self.dtype, gen)
        return x, src, wigner, coeff_index, dim

    def test_rotate_to_local_matches_reference(self):
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                x0, src, w0, coeff_index, dim = self._inputs(lmax, seed=lmax)
                mask = _block_mask(lmax, self.device)

                xa = x0.clone().requires_grad_(True)
                wa = w0.clone().requires_grad_(True)
                out = rotate_to_local(xa, src, wa, coeff_index, dim)
                xr = x0.clone().requires_grad_(True)
                wr = w0.clone().requires_grad_(True)
                ref = rotate_to_local_reference(xr, src, wr, coeff_index, dim)

                torch.testing.assert_close(out, ref, **self.tol)

                grad_out = torch.randn_like(ref)
                gxa, gwa = torch.autograd.grad(
                    out, [xa, wa], grad_out, retain_graph=True
                )
                gxr, gwr = torch.autograd.grad(ref, [xr, wr], grad_out)
                torch.testing.assert_close(gxa, gxr, **self.tol)
                torch.testing.assert_close(gwa[:, mask], gwr[:, mask], **self.tol)
                # The kernel never writes off-block Wigner gradient entries.
                self.assertEqual(float(gwa[:, ~mask].abs().max()), 0.0)

    def test_rotate_back_matches_reference(self):
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                _, _, w0, coeff_index, dim = self._inputs(lmax, seed=lmax)
                reduced = int(coeff_index.numel())
                gen = torch.Generator(device=self.device).manual_seed(100 + lmax)
                xl0 = torch.randn(
                    self.n_edge,
                    reduced,
                    self.channels,
                    device=self.device,
                    dtype=self.dtype,
                    generator=gen,
                )
                mask = _block_mask(lmax, self.device)

                xa = xl0.clone().requires_grad_(True)
                wa = w0.clone().requires_grad_(True)
                out = rotate_back(xa, wa, coeff_index, dim)
                xr = xl0.clone().requires_grad_(True)
                wr = w0.clone().requires_grad_(True)
                ref = rotate_back_reference(xr, wr, coeff_index, dim)

                torch.testing.assert_close(out, ref, **self.tol)

                grad_out = torch.randn_like(ref)
                gxa, gwa = torch.autograd.grad(
                    out, [xa, wa], grad_out, retain_graph=True
                )
                gxr, gwr = torch.autograd.grad(ref, [xr, wr], grad_out)
                torch.testing.assert_close(gxa, gxr, **self.tol)
                torch.testing.assert_close(gwa[:, mask], gwr[:, mask], **self.tol)

    def test_torch_compile_composability(self):
        """Gradients through the op match between eager and compiled tracing."""
        lmax = 3
        x0, src, w0, coeff_index, dim = self._inputs(lmax, seed=7)
        weight = torch.randn_like(
            rotate_to_local_reference(x0, src, w0, coeff_index, dim)
        )
        mask = _block_mask(lmax, self.device)

        def scalar_output(x, wigner):
            return (rotate_to_local(x, src, wigner, coeff_index, dim) * weight).sum()

        xe = x0.clone().requires_grad_(True)
        we = w0.clone().requires_grad_(True)
        gxe, gwe = torch.autograd.grad(scalar_output(xe, we), [xe, we])

        compiled = torch.compile(scalar_output, dynamic=True)
        xc = x0.clone().requires_grad_(True)
        wc = w0.clone().requires_grad_(True)
        gxc, gwc = torch.autograd.grad(compiled(xc, wc), [xc, wc])

        torch.testing.assert_close(gxc, gxe, **self.tol)
        # Also check the Wigner gradient (nonzero on the block entries) survives
        # tracing, since it flows through the custom op's registered backward.
        torch.testing.assert_close(gwc[:, mask], gwe[:, mask], **self.tol)
        self.assertGreater(float(gwe[:, mask].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
