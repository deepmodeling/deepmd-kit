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
2. ``make_fx(tracing_mode="symbolic")`` composability: the traced graph contains
   both the rotation forward and the autograd graph used by inference forces.
   This mirrors the SeZM inference path, which traces with ``make_fx`` before
   lowering the resulting graph through AOTAutograd's forward-only compiler.
"""

import math
import unittest

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.pt.model.descriptor.sezm_nn.indexing import (
    build_m_major_index,
    get_so3_dim_of_lmax,
)
from deepmd.pt.model.descriptor.sezm_nn.triton.so2_rotation import (
    TRITON_ROTATION_AVAILABLE,
    rotate_back_block,
    rotate_back_dense,
    rotate_back_reference,
    rotate_to_local_block,
    rotate_to_local_dense,
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


class TestSeZMTritonRotationDispatch(unittest.TestCase):
    def test_noncanonical_same_length_uses_dense_reference(self):
        device = torch.device("cpu")
        dtype = torch.float32
        lmax = 3
        dim = get_so3_dim_of_lmax(lmax)
        canonical = build_m_major_index(lmax, 1, device=device)
        coeff_index = torch.roll(canonical, shifts=1)
        x = torch.randn(4, dim, 3, device=device, dtype=dtype)
        src = torch.tensor([0, 2, 1, 3, 0], dtype=torch.long, device=device)
        wigner = torch.randn(src.numel(), dim, dim, device=device, dtype=dtype)
        x_local = torch.randn(
            src.numel(), coeff_index.numel(), 3, device=device, dtype=dtype
        )

        torch.testing.assert_close(
            rotate_to_local_dense(x, src, wigner, coeff_index, dim),
            rotate_to_local_reference(x, src, wigner, coeff_index, dim),
        )
        torch.testing.assert_close(
            rotate_back_dense(x_local, wigner, coeff_index, dim),
            rotate_back_reference(x_local, wigner, coeff_index, dim),
        )

    def test_symbolic_trace_noncanonical_same_length_uses_dense_op(self):
        device = torch.device("cpu")
        dtype = torch.float32
        lmax = 3
        dim = get_so3_dim_of_lmax(lmax)
        canonical = build_m_major_index(lmax, 1, device=device)
        coeff_index = torch.roll(canonical, shifts=1)
        x = torch.randn(4, dim, 3, device=device, dtype=dtype)
        src = torch.tensor([0, 2, 1, 3, 0], dtype=torch.long, device=device)
        wigner = torch.randn(src.numel(), dim, dim, device=device, dtype=dtype)

        def fn(x, src, wigner, coeff_index):
            return rotate_to_local_dense(x, src, wigner, coeff_index, dim)

        graph_module = make_fx(
            fn,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(x, src, wigner, coeff_index)
        graph_code = graph_module.code
        self.assertIn("sezm_triton.rotate_to_local.default", graph_code)
        self.assertNotIn("sezm_triton.rotate_to_local_block.default", graph_code)


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

    def _local_inputs(self, lmax, seed):
        _, _, wigner, coeff_index, dim = self._inputs(lmax, seed=seed)
        gen = torch.Generator(device=self.device).manual_seed(100 + seed)
        x_local = torch.randn(
            self.n_edge,
            int(coeff_index.numel()),
            self.channels,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        return x_local, wigner, coeff_index, dim

    def _assert_to_local_matches_reference(self, x0, src, w0, coeff_index, dim):
        lmax = math.isqrt(int(dim)) - 1
        mask = _block_mask(lmax, self.device)

        xa = x0.clone().requires_grad_(True)
        wa = w0.clone().requires_grad_(True)
        out = rotate_to_local_block(xa, src, wa, lmax)
        xr = x0.clone().requires_grad_(True)
        wr = w0.clone().requires_grad_(True)
        ref = rotate_to_local_reference(xr, src, wr, coeff_index, dim)

        torch.testing.assert_close(out, ref, **self.tol)

        grad_out = torch.randn_like(ref)
        gxa, gwa = torch.autograd.grad(out, [xa, wa], grad_out, retain_graph=True)
        gxr, gwr = torch.autograd.grad(ref, [xr, wr], grad_out)
        torch.testing.assert_close(gxa, gxr, **self.tol)
        torch.testing.assert_close(gwa[:, mask], gwr[:, mask], **self.tol)
        self.assertEqual(float(gwa[:, ~mask].abs().max()), 0.0)

    def _assert_back_matches_reference(self, xl0, w0, coeff_index, dim):
        lmax = math.isqrt(int(dim)) - 1
        mask = _block_mask(lmax, self.device)

        xa = xl0.clone().requires_grad_(True)
        wa = w0.clone().requires_grad_(True)
        out = rotate_back_block(xa, wa, lmax)
        xr = xl0.clone().requires_grad_(True)
        wr = w0.clone().requires_grad_(True)
        ref = rotate_back_reference(xr, wr, coeff_index, dim)

        torch.testing.assert_close(out, ref, **self.tol)

        grad_out = torch.randn_like(ref)
        gxa, gwa = torch.autograd.grad(out, [xa, wa], grad_out, retain_graph=True)
        gxr, gwr = torch.autograd.grad(ref, [xr, wr], grad_out)
        torch.testing.assert_close(gxa, gxr, **self.tol)
        torch.testing.assert_close(gwa[:, mask], gwr[:, mask], **self.tol)
        self.assertEqual(float(gwa[:, ~mask].abs().max()), 0.0)

    def test_eager_rotate_to_local_forward_backward_matches_reference(self):
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                x0, src, w0, coeff_index, dim = self._inputs(lmax, seed=lmax)
                self._assert_to_local_matches_reference(x0, src, w0, coeff_index, dim)

    def test_eager_rotate_back_forward_backward_matches_reference(self):
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                xl0, w0, coeff_index, dim = self._local_inputs(lmax, seed=lmax)
                self._assert_back_matches_reference(xl0, w0, coeff_index, dim)

    def test_symbolic_make_fx_rotate_to_local_forward_backward_matches_eager(self):
        """Symbolic FX captures rotate_to_local forward and autograd graph."""
        lmax = 3
        x0, src, w0, coeff_index, dim = self._inputs(lmax, seed=7)
        mask = _block_mask(lmax, self.device)
        grad_seed = torch.randn(
            self.n_edge,
            int(coeff_index.numel()),
            self.channels,
            device=self.device,
            dtype=self.dtype,
        )

        def forward_and_grad(x, wigner):
            x_req = x.detach().requires_grad_(True)
            w_req = wigner.detach().requires_grad_(True)
            out = rotate_to_local_block(x_req, src, w_req, lmax)
            grad_x, grad_wigner = torch.autograd.grad(
                out,
                (x_req, w_req),
                grad_seed,
            )
            return out, grad_x, grad_wigner

        out_eager, grad_x_eager, grad_w_eager = forward_and_grad(x0, w0)

        traced = make_fx(
            forward_and_grad,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(x0, w0)
        out_traced, grad_x_traced, grad_w_traced = traced(x0, w0)

        torch.testing.assert_close(out_traced, out_eager, **self.tol)
        torch.testing.assert_close(grad_x_traced, grad_x_eager, **self.tol)
        torch.testing.assert_close(
            grad_w_traced[:, mask],
            grad_w_eager[:, mask],
            **self.tol,
        )
        self.assertGreater(float(grad_w_eager[:, mask].abs().max()), 0.0)
        self.assertEqual(float(grad_w_traced[:, ~mask].abs().max()), 0.0)

    def test_symbolic_make_fx_rotate_back_forward_backward_matches_eager(self):
        """Symbolic FX captures rotate_back forward and autograd graph."""
        lmax = 3
        xl0, w0, coeff_index, dim = self._local_inputs(lmax, seed=7)
        mask = _block_mask(lmax, self.device)
        grad_seed = torch.randn(
            self.n_edge,
            dim,
            self.channels,
            device=self.device,
            dtype=self.dtype,
        )

        def forward_and_grad(x_local, wigner):
            x_req = x_local.detach().requires_grad_(True)
            w_req = wigner.detach().requires_grad_(True)
            out = rotate_back_block(x_req, w_req, lmax)
            grad_x, grad_wigner = torch.autograd.grad(
                out,
                (x_req, w_req),
                grad_seed,
            )
            return out, grad_x, grad_wigner

        out_eager, grad_x_eager, grad_w_eager = forward_and_grad(xl0, w0)

        traced = make_fx(
            forward_and_grad,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(xl0, w0)
        out_traced, grad_x_traced, grad_w_traced = traced(xl0, w0)

        torch.testing.assert_close(out_traced, out_eager, **self.tol)
        torch.testing.assert_close(grad_x_traced, grad_x_eager, **self.tol)
        torch.testing.assert_close(
            grad_w_traced[:, mask],
            grad_w_eager[:, mask],
            **self.tol,
        )
        self.assertGreater(float(grad_w_eager[:, mask].abs().max()), 0.0)
        self.assertEqual(float(grad_w_traced[:, ~mask].abs().max()), 0.0)

    def _check_make_fx_force(self, forward_and_grad, eager_args):
        """Trace ``forward_and_grad`` (forward + ``autograd.grad``) under symbolic
        ``make_fx`` and assert the traced graph reproduces the eager result.

        Returns the eager ``(out, grad_x, grad_wigner)`` triple for further checks.
        """
        eager = forward_and_grad(*eager_args)
        traced = make_fx(
            forward_and_grad,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(*eager_args)
        for got, want in zip(traced(*eager_args), eager, strict=True):
            torch.testing.assert_close(got, want, **self.tol)
        return eager

    def test_symbolic_make_fx_rotate_to_local_dense_matches_eager_and_reference(self):
        """Dense op composes with symbolic make_fx + autograd and matches the eager
        reference on a full (non-block) Wigner-D, honoring ``coeff_index``.
        """
        lmax = 3
        _, src, _, coeff_index, dim = self._inputs(lmax, seed=11)
        gen = torch.Generator(device=self.device).manual_seed(11)
        x0 = torch.randn(
            self.n_node,
            dim,
            self.channels,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        w0 = torch.randn(
            self.n_edge, dim, dim, device=self.device, dtype=self.dtype, generator=gen
        )
        grad_seed = torch.randn(
            self.n_edge,
            int(coeff_index.numel()),
            self.channels,
            device=self.device,
            dtype=self.dtype,
        )

        def forward_and_grad(x, wigner):
            x_req = x.detach().requires_grad_(True)
            w_req = wigner.detach().requires_grad_(True)
            out = rotate_to_local_dense(x_req, src, w_req, coeff_index, dim)
            grad_x, grad_wigner = torch.autograd.grad(out, (x_req, w_req), grad_seed)
            return out, grad_x, grad_wigner

        out_eager, grad_x_eager, grad_w_eager = self._check_make_fx_force(
            forward_and_grad, (x0, w0)
        )

        xr = x0.detach().requires_grad_(True)
        wr = w0.detach().requires_grad_(True)
        ref = rotate_to_local_reference(xr, src, wr, coeff_index, dim)
        grad_x_ref, grad_w_ref = torch.autograd.grad(ref, (xr, wr), grad_seed)
        torch.testing.assert_close(out_eager, ref, **self.tol)
        torch.testing.assert_close(grad_x_eager, grad_x_ref, **self.tol)
        torch.testing.assert_close(grad_w_eager, grad_w_ref, **self.tol)
        self.assertGreater(float(grad_w_eager.abs().max()), 0.0)

    def test_symbolic_make_fx_rotate_back_dense_matches_eager_and_reference(self):
        """Dense rotate_back composes with symbolic make_fx + autograd and matches
        the eager reference on a full (non-block) Wigner-D, honoring ``coeff_index``.
        """
        lmax = 3
        _, _, _, coeff_index, dim = self._inputs(lmax, seed=11)
        gen = torch.Generator(device=self.device).manual_seed(11)
        xl0 = torch.randn(
            self.n_edge,
            int(coeff_index.numel()),
            self.channels,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        w0 = torch.randn(
            self.n_edge, dim, dim, device=self.device, dtype=self.dtype, generator=gen
        )
        grad_seed = torch.randn(
            self.n_edge, dim, self.channels, device=self.device, dtype=self.dtype
        )

        def forward_and_grad(x_local, wigner):
            x_req = x_local.detach().requires_grad_(True)
            w_req = wigner.detach().requires_grad_(True)
            out = rotate_back_dense(x_req, w_req, coeff_index, dim)
            grad_x, grad_wigner = torch.autograd.grad(out, (x_req, w_req), grad_seed)
            return out, grad_x, grad_wigner

        out_eager, grad_x_eager, grad_w_eager = self._check_make_fx_force(
            forward_and_grad, (xl0, w0)
        )

        xr = xl0.detach().requires_grad_(True)
        wr = w0.detach().requires_grad_(True)
        ref = rotate_back_reference(xr, wr, coeff_index, dim)
        grad_x_ref, grad_w_ref = torch.autograd.grad(ref, (xr, wr), grad_seed)
        torch.testing.assert_close(out_eager, ref, **self.tol)
        torch.testing.assert_close(grad_x_eager, grad_x_ref, **self.tol)
        torch.testing.assert_close(grad_w_eager, grad_w_ref, **self.tol)
        self.assertGreater(float(grad_w_eager.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
