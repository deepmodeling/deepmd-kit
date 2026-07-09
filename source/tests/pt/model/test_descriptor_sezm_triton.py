# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the opt-in Triton inference kernels of the SeZM descriptor
(enabled via the ``DP_TRITON_INFER`` level, see
:func:`deepmd.kernels.utils.triton_infer_level`): the
block-diagonal SO(2)/Wigner rotation, the fused dynamic radial degree mixer,
the fused value path with its table-routed edge-block backwards, and the
level-3 fp16x3 mixing stack.

For the rotation kernels two properties are checked against the eager PyTorch
reference:

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
import typing
import unittest

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.kernels.triton.sezm import (
    TRITON_AVAILABLE,
)
from deepmd.kernels.triton.sezm.radial_mix import (
    radial_mix_block,
    radial_mix_reference,
)
from deepmd.kernels.triton.sezm.so2_rotation import (
    rotate_back_block,
    rotate_back_block_so2,
    rotate_back_dense,
    rotate_back_reference,
    rotate_to_local_block,
    rotate_to_local_dense,
    rotate_to_local_reference,
)
from deepmd.pt.model.descriptor.sezm_nn.indexing import (
    build_m_major_index,
    get_so3_dim_of_lmax,
)
from deepmd.pt.model.descriptor.sezm_nn.so2 import (
    DynamicRadialDegreeMixer,
)

_CUDA = torch.cuda.is_available()

# All SeZM Triton kernels ship together (every per-module availability flag
# reduces to "the triton package imports"), so one gate covers every
# GPU-kernel test class.  Dispatch- and parsing-only classes stay ungated.
_GPU_KERNELS = unittest.skipUnless(
    _CUDA and TRITON_AVAILABLE,
    "CUDA and Triton are required for the SeZM GPU kernels",
)


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


@_GPU_KERNELS
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

    def test_rotate_back_so2_matches_block_on_focus_layout(self):
        """The layout-aware rotate_back reads the per-focus layout (E, F, D_m, Cf)
        directly and reproduces the standard block kernel on the transposed
        (E, D_m, F * Cf) input, forward and backward (including grad_wigner on the
        block entries). This is the in-place transpose the SeZM SO(2) pipeline
        avoids materializing.
        """
        n_focus, focus_dim = 2, 8
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                gen = torch.Generator(device=self.device).manual_seed(lmax)
                reduced = 3 * lmax + 1
                mask = _block_mask(lmax, self.device)
                x4 = torch.randn(
                    self.n_edge,
                    n_focus,
                    reduced,
                    focus_dim,
                    device=self.device,
                    dtype=self.dtype,
                    generator=gen,
                )
                w0 = _block_diagonal_wigner(
                    self.n_edge, lmax, self.device, self.dtype, gen
                )

                xa = x4.clone().requires_grad_(True)
                wa = w0.clone().requires_grad_(True)
                out = rotate_back_block_so2(xa, wa, lmax)

                x_std = x4.transpose(1, 2).reshape(
                    self.n_edge, reduced, n_focus * focus_dim
                )
                xr = x_std.clone().requires_grad_(True)
                wr = w0.clone().requires_grad_(True)
                ref = rotate_back_block(xr, wr, lmax)
                torch.testing.assert_close(out, ref, **self.tol)

                grad_out = torch.randn_like(ref)
                gxa, gwa = torch.autograd.grad(
                    out, [xa, wa], grad_out, retain_graph=True
                )
                gxr, gwr = torch.autograd.grad(ref, [xr, wr], grad_out)
                gxa_std = gxa.transpose(1, 2).reshape(
                    self.n_edge, reduced, n_focus * focus_dim
                )
                torch.testing.assert_close(gxa_std, gxr, **self.tol)
                torch.testing.assert_close(gwa[:, mask], gwr[:, mask], **self.tol)

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


@_GPU_KERNELS
class TestSeZMTritonRadialMix(unittest.TestCase):
    """Fused dynamic radial degree mixer (``degree_channel``, ``mmax == 1``).

    The Triton kernel and its eager reference are checked against the production
    scatter path of :class:`DynamicRadialDegreeMixer`, and the forward/backward
    are checked for symbolic ``make_fx`` composability with the inference-force
    autograd graph.
    """

    def setUp(self):
        self.device = torch.device("cuda")
        self.dtype = torch.float32
        self.n_edge, self.channels, self.rank = 4096, 64, 1
        self.tol = {"rtol": 2e-4, "atol": 2e-4}

    def _mixer(self, lmax):
        return (
            DynamicRadialDegreeMixer(
                lmax=lmax,
                mmax=1,
                channels=self.channels,
                mode="degree_channel",
                rank=self.rank,
                dtype=self.dtype,
                seed=1,
                trainable=True,
            )
            .to(self.device)
            .eval()
        )

    def _inputs(self, mixer, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        x_local = torch.randn(
            self.n_edge,
            mixer.reduced_dim,
            self.channels,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        radial_feat = torch.randn(
            self.n_edge,
            mixer.reduced_dim,
            self.channels,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        compact = mixer._project_radial(radial_feat).view(
            self.n_edge, mixer.degree_kernel_size, self.rank
        )
        return x_local, radial_feat, compact

    def test_reference_matches_module_eager_path(self):
        """The block-split reference reproduces the module's dense scatter path."""
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                mixer = self._mixer(lmax)
                # Force the dense scatter path regardless of the ambient flag.
                mixer._radial_mix_block = None
                x_local, radial_feat, compact = self._inputs(mixer, seed=lmax)
                with torch.no_grad():
                    module_out = mixer(x_local, radial_feat)
                    ref_out = radial_mix_reference(
                        compact, x_local, mixer.channel_basis, lmax
                    )
                torch.testing.assert_close(ref_out, module_out, **self.tol)

    def test_triton_forward_matches_reference(self):
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                mixer = self._mixer(lmax)
                x_local, _, compact = self._inputs(mixer, seed=lmax)
                with torch.no_grad():
                    out = radial_mix_block(compact, x_local, mixer.channel_basis, lmax)
                    ref = radial_mix_reference(
                        compact, x_local, mixer.channel_basis, lmax
                    )
                torch.testing.assert_close(out, ref, **self.tol)

    def test_triton_backward_matches_reference(self):
        """Backward correctness on a fresh first call (checks reset_to_zero)."""
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                mixer = self._mixer(lmax)
                x_local, _, compact = self._inputs(mixer, seed=lmax)
                grad_out = torch.randn_like(x_local)

                ca = compact.detach().requires_grad_(True)
                xa = x_local.detach().requires_grad_(True)
                out = radial_mix_block(ca, xa, mixer.channel_basis, lmax)
                grad_ca, grad_xa = torch.autograd.grad(out, [ca, xa], grad_out)

                cr = compact.detach().requires_grad_(True)
                xr = x_local.detach().requires_grad_(True)
                ref = radial_mix_reference(cr, xr, mixer.channel_basis, lmax)
                grad_cr, grad_xr = torch.autograd.grad(ref, [cr, xr], grad_out)

                torch.testing.assert_close(grad_ca, grad_cr, **self.tol)
                torch.testing.assert_close(grad_xa, grad_xr, **self.tol)

    def test_symbolic_make_fx_forward_backward_matches_eager(self):
        """Symbolic FX captures the mixer forward and its inference-force graph."""
        lmax = 3
        mixer = self._mixer(lmax)
        x_local, _, compact = self._inputs(mixer, seed=7)
        channel_basis = mixer.channel_basis
        grad_seed = torch.randn_like(x_local)

        def forward_and_grad(compact, x_local):
            compact_req = compact.detach().requires_grad_(True)
            x_req = x_local.detach().requires_grad_(True)
            out = radial_mix_block(compact_req, x_req, channel_basis, lmax)
            grad_compact, grad_x = torch.autograd.grad(
                out, (compact_req, x_req), grad_seed
            )
            return out, grad_compact, grad_x

        eager = forward_and_grad(compact, x_local)
        traced = make_fx(
            forward_and_grad,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(compact, x_local)
        for got, want in zip(traced(compact, x_local), eager, strict=True):
            torch.testing.assert_close(got, want, **self.tol)
        self.assertIn("sezm_triton.radial_mix_block", traced.code)


@_GPU_KERNELS
class TestSeZMTritonValuePath(unittest.TestCase):
    """Cross-check the fused SO(2) value path against ``SO2Convolution``.

    The fused entry replaces ``so2_message(..., return_local=True)``
    end to end (rotation, radial degree mixing, gated mixing stack, focus
    competition), so the reference is the module's own eager path.  Cases
    span the supported family axes: mixer-free ``lmax = 1``, the rank-1
    ``degree_channel`` mixer at small and large ``lmax``, a rank-2 mixer with
    two focus streams, and a non-power-of-two focus width.
    """

    CASES: typing.ClassVar[list[tuple]] = [
        # (lmax, channels, n_focus, focus_dim, mixing_layers, mode, rank)
        (1, 32, 1, 0, 3, "none", 0),
        (2, 32, 1, 0, 3, "degree_channel", 1),
        (4, 64, 1, 0, 4, "degree_channel", 1),
        (5, 64, 2, 0, 4, "degree_channel", 2),
        (3, 64, 2, 96, 4, "degree_channel", 1),
    ]

    N_NODE = 512
    N_EDGE = 20000

    def _build_conv(self, lmax, channels, n_focus, focus_dim, layers, mode, rank):
        from deepmd.pt.model.descriptor.sezm_nn.so2 import (
            SO2Convolution,
        )

        return (
            SO2Convolution(
                lmax=lmax,
                mmax=1,
                channels=channels,
                n_focus=n_focus,
                focus_dim=focus_dim,
                mixing_layers=layers,
                radial_so2_mode=mode,
                radial_so2_rank=rank,
                n_atten_head=1,
                dtype=torch.float32,
                seed=7,
                trainable=False,
            )
            .to("cuda")
            .eval()
        )

    def _edge_inputs(self, conv, lmax, channels):
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            WignerDCalculator,
        )

        generator = torch.Generator(device="cuda").manual_seed(42)
        dim = (lmax + 1) ** 2
        c_wide = conv.n_focus * conv.so2_focus_dim
        x = (
            torch.randn(self.N_NODE, dim, c_wide, device="cuda", generator=generator)
            * 0.5
        )
        src = torch.randint(
            0, self.N_NODE, (self.N_EDGE,), device="cuda", generator=generator
        )
        radial = (
            torch.randn(
                self.N_EDGE, lmax + 1, channels, device="cuda", generator=generator
            )
            * 0.3
        )
        calculator = WignerDCalculator(lmax, dtype=torch.float32).to("cuda")
        quaternion = torch.randn(self.N_EDGE, 4, device="cuda", generator=generator)
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
        wigner = calculator(quaternion)[0]

        class _Cache:
            pass

        cache = _Cache()
        cache.src = src
        cache.dst = torch.randint(
            0, self.N_NODE, (self.N_EDGE,), device="cuda", generator=generator
        )
        cache.D_full = wigner
        cache.D_to_m_cache = {}
        return x, cache, radial

    def test_forward_backward_matches_reference_across_family(self):
        from deepmd.kernels.triton.sezm.so2_value_path import (
            make_triton_value_path,
        )

        for case in self.CASES:
            lmax, channels, n_focus, focus_dim, layers, mode, rank = case
            with self.subTest(case=case):
                conv = self._build_conv(*case)
                fused = make_triton_value_path(conv)
                self.assertIsNotNone(fused)
                x, cache, radial = self._edge_inputs(conv, lmax, channels)

                x_ref = x.clone().requires_grad_(True)
                rad_ref = radial.clone().requires_grad_(True)
                wigner_ref = cache.D_full.clone().requires_grad_(True)
                cache.D_full = wigner_ref
                ref_local, _ = conv.so2_message(
                    x_ref, cache, rad_ref, return_local=True
                )

                x_fused = x.clone().requires_grad_(True)
                rad_fused = radial.clone().requires_grad_(True)
                wigner_fused = wigner_ref.detach().clone().requires_grad_(True)
                cache.D_full = wigner_fused
                out_local, _ = fused(x_fused, cache, rad_fused)

                scale = ref_local.abs().max().item()
                torch.testing.assert_close(
                    out_local, ref_local, atol=5e-5 * max(scale, 1.0), rtol=1e-4
                )

                grad_seed = torch.randn_like(ref_local)
                ref_grads = torch.autograd.grad(
                    ref_local, [x_ref, rad_ref, wigner_ref], grad_seed
                )
                fused_grads = torch.autograd.grad(
                    out_local, [x_fused, rad_fused, wigner_fused], grad_seed
                )
                # The Wigner gradient is compared on the structural block
                # diagonal only: off-block entries multiply exactly-zero
                # Wigner values, so the model discards them.
                mask = _block_mask(lmax, "cuda")
                comparisons = [
                    (ref_grads[0], fused_grads[0]),
                    (ref_grads[1], fused_grads[1]),
                    (ref_grads[2] * mask, fused_grads[2] * mask),
                ]
                for ref_grad, fused_grad in comparisons:
                    grad_scale = ref_grad.abs().max().item()
                    torch.testing.assert_close(
                        fused_grad,
                        ref_grad,
                        atol=1e-4 * max(grad_scale, 1.0),
                        rtol=1e-4,
                    )

    def test_factory_rejects_unsupported_layouts(self):
        from deepmd.kernels.triton.sezm.so2_value_path import (
            make_triton_value_path,
        )

        conv = self._build_conv(2, 32, 1, 0, 3, "degree_channel", 1)
        conv.mmax = 2
        self.assertIsNone(make_triton_value_path(conv))


@_GPU_KERNELS
class TestSeZMTritonWignerMonomials(unittest.TestCase):
    """Check the fused quaternion monomial operator against its reference."""

    def _exponents(self, degree):
        """All exponent 4-tuples of the given total degree, flattened."""
        exps = []
        for a in range(degree + 1):
            for b in range(degree + 1 - a):
                for c in range(degree + 1 - a - b):
                    exps.extend((a, b, c, degree - a - b - c))
        return exps

    def test_forward_backward_matches_reference(self):
        from deepmd.kernels.triton.sezm.wigner_monomials import (
            _monomials_reference,
            wigner_monomials,
        )

        generator = torch.Generator(device="cuda").manual_seed(3)
        for degree in (4, 6, 8, 10, 12):
            with self.subTest(degree=degree):
                exponents = self._exponents(degree)
                q = torch.randn(4096, 4, device="cuda", generator=generator)
                q = q / q.norm(dim=-1, keepdim=True)

                q_fused = q.clone().requires_grad_(True)
                out = wigner_monomials(q_fused, exponents, degree)
                q_ref = q.clone().requires_grad_(True)
                ref = _monomials_reference(q_ref, exponents, degree)
                torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)

                grad_seed = torch.randn_like(ref)
                (grad_fused,) = torch.autograd.grad(out, q_fused, grad_seed)
                (grad_ref,) = torch.autograd.grad(ref, q_ref, grad_seed)
                torch.testing.assert_close(grad_fused, grad_ref, atol=1e-5, rtol=1e-5)

    def test_wigner_calculator_matches_reference_chain(self):
        """The calculator's fused monomial path reproduces the dense chain."""
        import os
        from unittest import (
            mock,
        )

        from deepmd.pt.model.descriptor.sezm_nn import wignerd as wignerd_module

        generator = torch.Generator(device="cuda").manual_seed(11)
        q = torch.randn(2048, 4, device="cuda", generator=generator)
        q = q / q.norm(dim=-1, keepdim=True)
        for lmax in (2, 3, 4, 5):
            with self.subTest(lmax=lmax):
                with mock.patch.dict(os.environ, {"DP_TRITON_INFER": "1"}):
                    fused_calc = (
                        wignerd_module.WignerDCalculator(lmax, dtype=torch.float32)
                        .to("cuda")
                        .eval()
                    )
                with mock.patch.dict(os.environ, {"DP_TRITON_INFER": "0"}):
                    ref_calc = (
                        wignerd_module.WignerDCalculator(lmax, dtype=torch.float32)
                        .to("cuda")
                        .eval()
                    )
                self.assertTrue(fused_calc._use_triton_monomials)
                got = fused_calc(q)[0]
                want = ref_calc(q)[0]
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@_GPU_KERNELS
class TestSeZMTritonForceAssembly(unittest.TestCase):
    """Check the segmented force / virial assembly against ``index_add``."""

    def _topology(self, n_edge, n_ext, device, generator):
        dst = torch.randint(0, n_ext, (n_edge,), device=device, generator=generator)
        src = torch.randint(0, n_ext, (n_edge,), device=device, generator=generator)
        dst_order = torch.argsort(dst)
        src_order = torch.argsort(src)
        boundaries = torch.arange(n_ext + 1, device=device, dtype=dst.dtype)
        dst_row_ptr = torch.searchsorted(dst.index_select(0, dst_order), boundaries)
        src_row_ptr = torch.searchsorted(src.index_select(0, src_order), boundaries)
        return dst, src, dst_order, dst_row_ptr, src_order, src_row_ptr

    def test_matches_index_add_assembly(self):
        from deepmd.kernels.triton.sezm.force_assembly import (
            edge_force_assembly,
        )

        generator = torch.Generator(device="cuda").manual_seed(5)
        n_edge, n_ext = 50000, 700
        g = torch.randn(n_edge, 3, device="cuda", generator=generator)
        edge_vec = torch.randn(n_edge, 3, device="cuda", generator=generator)
        dst, src, dst_order, dst_row_ptr, src_order, src_row_ptr = self._topology(
            n_edge, n_ext, "cuda", generator
        )

        force, virial = edge_force_assembly(
            g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
        )

        force_ref = torch.zeros(n_ext, 3, device="cuda")
        force_ref.index_add_(0, dst, g)
        force_ref.index_add_(0, src, -g)
        half_w = -0.5 * torch.einsum("ek,ej->ekj", g, edge_vec).reshape(-1, 9)
        virial_ref = torch.zeros(n_ext, 9, device="cuda")
        virial_ref.index_add_(0, dst, half_w)
        virial_ref.index_add_(0, src, half_w)

        torch.testing.assert_close(force, force_ref, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(virial, virial_ref, atol=1e-4, rtol=1e-5)


@_GPU_KERNELS
class TestSeZMTritonFlashAttenSegmented(unittest.TestCase):
    """Check the destination-segmented flash forward against the reference.

    Destinations are deliberately unsorted: the traced SeZM graph keeps
    masked padding edges in arbitrary destination order, so the operator must
    build its own sorted CSR topology (a sorted-input-only regression once
    produced silently wrong aggregates on the compiled path).
    """

    def test_forward_matches_reference_on_unsorted_destinations(self):
        from deepmd.kernels.triton.sezm.flash_atten import (
            build_row_ptr,
            flash_atten_aggregate,
            flash_atten_aggregate_reference,
        )

        generator = torch.Generator(device="cuda").manual_seed(9)
        lmax, n_focus, focus_dim, n_head = 3, 2, 32, 2
        n_edge, n_node = 30000, 400
        reduced_dim = 3 * lmax + 1
        dim = (lmax + 1) ** 2

        x_local = torch.randn(
            n_edge, n_focus, reduced_dim, focus_dim, device="cuda", generator=generator
        )
        wigner_dt = _block_diagonal_wigner(
            n_edge, lmax, "cuda", torch.float32, generator
        )
        rescale = torch.rand(dim, device="cuda", generator=generator) + 0.5
        alpha = torch.rand(n_edge, n_focus, n_head, device="cuda", generator=generator)
        dst = torch.randint(0, n_node, (n_edge,), device="cuda", generator=generator)
        row_ptr = build_row_ptr(torch.sort(dst).values, n_node)

        got = flash_atten_aggregate(
            x_local, wigner_dt, rescale, alpha, row_ptr, dst, lmax, n_head
        )
        want = flash_atten_aggregate_reference(
            x_local, wigner_dt, rescale, alpha, dst, n_node, lmax, n_head
        )
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-5)

    def test_backward_matches_reference_on_both_dispatch_paths(self):
        """The backward is exact on the edge-block and the per-edge dispatch.

        The routing table keys on ``(C_wide, lmax)``.  Both routes are pinned
        through explicit runtime registrations (an entry for the narrow case,
        ``None`` for the wide one) so the test exercises both kernels
        regardless of the built-in coverage of the running GPU.
        """
        from deepmd.kernels.triton.sezm import (
            tile_configs,
        )
        from deepmd.kernels.triton.sezm.flash_atten import (
            _flash_atten_backward_reference,
            _flash_bwd_op,
        )

        runtime = tile_configs._RUNTIME["flash_bwd_block"]
        saved = dict(runtime)
        self.addCleanup(lambda: (runtime.clear(), runtime.update(saved)))
        tile_configs.register_tile_configs(
            "flash_bwd_block", {(64, 3): (4, 2, 2), (256, 3): None}
        )

        generator = torch.Generator(device="cuda").manual_seed(13)
        cases = [
            # (lmax, n_focus, focus_dim, expects_block_dispatch)
            (3, 2, 32, True),
            (3, 2, 128, False),
        ]
        for lmax, n_focus, focus_dim, expects_block in cases:
            with self.subTest(lmax=lmax, c_wide=n_focus * focus_dim):
                self.assertEqual(
                    tile_configs.flash_bwd_block_config(n_focus * focus_dim, lmax)
                    is not None,
                    expects_block,
                )
                n_edge, n_node, n_head = 20000, 300, 1
                reduced_dim = 3 * lmax + 1
                dim = (lmax + 1) ** 2
                grad_pre_gate = torch.randn(
                    n_node, dim, n_focus * focus_dim, device="cuda", generator=generator
                )
                x_local = torch.randn(
                    n_edge,
                    n_focus,
                    reduced_dim,
                    focus_dim,
                    device="cuda",
                    generator=generator,
                )
                wigner_dt = _block_diagonal_wigner(
                    n_edge, lmax, "cuda", torch.float32, generator
                )
                rescale = torch.rand(dim, device="cuda", generator=generator) + 0.5
                alpha = torch.rand(
                    n_edge, n_focus, n_head, device="cuda", generator=generator
                )
                dst = torch.randint(
                    0, n_node, (n_edge,), device="cuda", generator=generator
                )

                got = _flash_bwd_op(
                    grad_pre_gate, x_local, wigner_dt, rescale, alpha, dst, lmax, n_head
                )
                want = _flash_atten_backward_reference(
                    grad_pre_gate, x_local, wigner_dt, rescale, alpha, dst, lmax, n_head
                )
                mask = _block_mask(lmax, "cuda")
                comparisons = [
                    (got[0], want[0]),
                    (got[1] * mask, want[1] * mask),
                    (got[2], want[2]),
                ]
                for got_grad, want_grad in comparisons:
                    scale = want_grad.abs().max().item()
                    torch.testing.assert_close(
                        got_grad,
                        want_grad,
                        atol=1e-4 * max(scale, 1.0),
                        rtol=1e-4,
                    )


class TestTritonInferLevel(unittest.TestCase):
    """Parse and reject semantics of the ``DP_TRITON_INFER`` level."""

    def test_levels_parse_and_non_numeric_values_are_rejected(self):
        import os
        from unittest import (
            mock,
        )

        from deepmd.kernels.utils import (
            triton_infer_level,
        )

        for raw, expected in (("0", 0), ("1", 1), ("2", 2), ("3", 3), (" 2 ", 2)):
            with mock.patch.dict(os.environ, {"DP_TRITON_INFER": raw}):
                self.assertEqual(triton_infer_level(), expected)
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("DP_TRITON_INFER", None)
            self.assertEqual(triton_infer_level(), 0)
        for raw in ("on", "off", "true", "false", "yes", "4", "-1"):
            with (
                mock.patch.dict(os.environ, {"DP_TRITON_INFER": raw}),
                self.assertRaises(ValueError),
            ):
                triton_infer_level()


@_GPU_KERNELS
class TestSeZMStackFP16x3(unittest.TestCase):
    """Correctness of the level-3 fp16x3 mixing-stack operator.

    The accuracy contract is relative: the fp16x3 error against an fp64
    reference must stay within a small factor of the fp32 operator's own
    rounding error on identical data (absolute thresholds mis-fire because
    fp32 rounding grows with the reduction width).  Finiteness is asserted
    over the full tensors and across input magnitude scales spanning the
    documented dynamic-range protections.
    """

    LMAX = 3
    FOCUS_DIM = 32
    N_FOCUS = 2
    N_LAYERS = 3
    N_EDGE = 20000

    def setUp(self) -> None:
        """Pin launch configurations so the tests run on any CUDA device.

        The fp16x3 operator refuses shapes without a table entry, and the
        built-in tables only cover swept GPU models.  On uncovered devices a
        conservative single-stage configuration is registered for the test
        shapes: ``num_stages = 1`` disables the software pipeliner, which is
        the component whose miscompilation the validated tables guard
        against, and the fp64 comparisons of this class independently verify
        the numerics on whatever device runs the suite.
        """
        from deepmd.kernels.triton.sezm import (
            tile_configs,
        )

        for key in ((self.FOCUS_DIM, self.LMAX), (32, 2)):
            if tile_configs.stack_fp16x3_configs(*key) is None:
                tile_configs.register_tile_configs(
                    "stack_fp16x3", {key: ((64, 64, 32, 4, 1),) * 4}
                )
                self.addCleanup(
                    lambda key=key: tile_configs._RUNTIME["stack_fp16x3"].pop(key, None)
                )

    def _stack_inputs(self, generator):
        lmax, focus_dim, n_focus = self.LMAX, self.FOCUS_DIM, self.N_FOCUS
        m0 = (lmax + 1) * focus_dim
        half = lmax * focus_dim
        row = (3 * lmax + 1) * focus_dim

        def randn(*shape):
            return torch.randn(*shape, device="cuda", generator=generator)

        u0 = randn(n_focus, self.N_EDGE, row)
        alpha = (
            torch.rand(self.N_EDGE, n_focus, device="cuda", generator=generator) + 0.1
        )
        w0_all = randn(self.N_LAYERS, n_focus, m0, m0) * 0.2
        block_u = randn(self.N_LAYERS, n_focus, half, half) * 0.2
        block_v = randn(self.N_LAYERS, n_focus, half, half) * 0.2
        # The |m| = 1 weight carries the [[U, V], [-V, U]] complex structure
        # of SO2Linear so the synthetic stack matches the production operator.
        w1_all = torch.cat(
            [
                torch.cat([block_u, block_v], dim=3),
                torch.cat([-block_v, block_u], dim=3),
            ],
            dim=2,
        ).contiguous()
        gw_all = randn(self.N_LAYERS - 1, n_focus, focus_dim, half) * 0.3
        return u0, alpha, w0_all, w1_all, gw_all

    def _errors_against_fp64(self, op, u0, alpha, w0_all, w1_all, gw_all, grad_seed):
        from deepmd.kernels.triton.sezm.so2_value_path import (
            _mixing_stack_reference,
        )

        u0_ref = u0.double().requires_grad_(True)
        alpha_ref = alpha.double().requires_grad_(True)
        x_ref, _ = _mixing_stack_reference(
            u0_ref,
            alpha_ref,
            w0_all.double(),
            w1_all.double(),
            gw_all.double(),
            self.LMAX,
            self.FOCUS_DIM,
            True,
        )
        gu_ref, _ = torch.autograd.grad(x_ref, [u0_ref, alpha_ref], grad_seed.double())

        u0_run = u0.clone().requires_grad_(True)
        alpha_run = alpha.clone().requires_grad_(True)
        x_run, z_run = op(
            u0_run, alpha_run, w0_all, w1_all, gw_all, self.LMAX, self.FOCUS_DIM, True
        )
        self.assertTrue(bool(torch.isfinite(x_run).all()))
        self.assertTrue(bool(torch.isfinite(z_run).all()))
        gu_run, _ = torch.autograd.grad(x_run, [u0_run, alpha_run], grad_seed)
        self.assertTrue(bool(torch.isfinite(gu_run).all()))

        def relerr(a, b):
            return float((a - b.float()).abs().max() / b.abs().max().clamp_min(1e-30))

        return relerr(x_run, x_ref), relerr(gu_run, gu_ref)

    def test_matches_fp64_within_fp32_error_budget(self):
        from deepmd.kernels.triton.sezm.so2_stack_fp16x3 import (
            mixing_stack_fp16x3,
        )
        from deepmd.kernels.triton.sezm.so2_value_path import (
            _mixing_stack_op,
        )

        generator = torch.Generator(device="cuda").manual_seed(21)
        inputs = self._stack_inputs(generator)
        row = (3 * self.LMAX + 1) * self.FOCUS_DIM
        grad_seed = torch.randn(
            self.N_EDGE, self.N_FOCUS, row, device="cuda", generator=generator
        )

        fp32_fwd, fp32_bwd = self._errors_against_fp64(
            _mixing_stack_op, *inputs, grad_seed
        )
        x3_fwd, x3_bwd = self._errors_against_fp64(
            mixing_stack_fp16x3, *inputs, grad_seed
        )
        self.assertLess(x3_fwd, max(3.0 * fp32_fwd, 2e-6))
        self.assertLess(x3_bwd, max(3.0 * fp32_bwd, 8e-6))

    def test_extreme_input_scales_stay_finite_and_accurate(self):
        """The tail scaling and the activation prescale hold across magnitudes.

        Inputs four orders of magnitude below and two above the typical
        working point must stay finite with the error budget intact; this
        pins the ``2^11`` tail scaling (small magnitudes) and the ``2^-4``
        activation prescale (large magnitudes).
        """
        from deepmd.kernels.triton.sezm.so2_stack_fp16x3 import (
            mixing_stack_fp16x3,
        )
        from deepmd.kernels.triton.sezm.so2_value_path import (
            _mixing_stack_op,
        )

        generator = torch.Generator(device="cuda").manual_seed(22)
        u0, alpha, w0_all, w1_all, gw_all = self._stack_inputs(generator)
        row = (3 * self.LMAX + 1) * self.FOCUS_DIM
        grad_seed = torch.randn(
            self.N_EDGE, self.N_FOCUS, row, device="cuda", generator=generator
        )
        for scale in (1e-4, 1e2):
            with self.subTest(scale=scale):
                scaled = (u0 * scale, alpha, w0_all, w1_all, gw_all)
                fp32_fwd, fp32_bwd = self._errors_against_fp64(
                    _mixing_stack_op, *scaled, grad_seed
                )
                x3_fwd, x3_bwd = self._errors_against_fp64(
                    mixing_stack_fp16x3, *scaled, grad_seed
                )
                self.assertLess(x3_fwd, max(3.0 * fp32_fwd, 2e-6))
                self.assertLess(x3_bwd, max(3.0 * fp32_bwd, 8e-6))

    def test_inductor_compiled_matches_eager(self):
        """The Inductor-lowered operator is bitwise identical to eager.

        Guards the weight fp16 splits: the tail of a split is defined by an
        ``fp32 -> fp16 -> fp32`` rounding round-trip, which Inductor's
        pointwise fusion elides when the split is expressed in aten (the
        intermediate stays in an fp32 register), zeroing the tails and
        silently degrading the compiled operator to fp16-head weights.  The
        split therefore runs as a Triton kernel, and this test pins the
        compiled-versus-eager parity through the same make_fx + Inductor
        pipeline that model freezing uses.
        """
        from torch._functorch.aot_autograd import (
            aot_module_simplified,
        )
        from torch._inductor.compile_fx import (
            compile_fx_inner,
        )
        from torch._inductor.decomposition import (
            select_decomp_table,
        )
        from torch.fx.experimental.proxy_tensor import (
            make_fx,
        )

        from deepmd.kernels.triton.sezm.so2_stack_fp16x3 import (
            mixing_stack_fp16x3,
        )

        generator = torch.Generator(device="cuda").manual_seed(23)
        inputs = self._stack_inputs(generator)
        lmax, focus_dim = self.LMAX, self.FOCUS_DIM

        def fn(u0, alpha, w0_all, w1_all, gw_all):
            x_local, z_all = mixing_stack_fp16x3(
                u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, True
            )
            return (x_local, z_all)

        eager_x, eager_z = fn(*inputs)
        graph = make_fx(fn, tracing_mode="symbolic")(*inputs)
        # AOTAutograd's PhiloxStateTracker allocates tensors without an
        # explicit device and would trip the pt-test default-device sentinel
        # (source/tests/pt/__init__.py), so the sentinel is suspended here.
        saved_device = torch.get_default_device()
        torch.set_default_device(None)
        try:
            compiled = aot_module_simplified(
                graph,
                inputs,
                fw_compiler=lambda gm, args: compile_fx_inner(gm, args),
                decompositions=select_decomp_table(),
            )
            with torch.no_grad():
                compiled_x, compiled_z = compiled(*inputs)
        finally:
            torch.set_default_device(saved_device)
        torch.testing.assert_close(compiled_x, eager_x, atol=0.0, rtol=0.0)
        torch.testing.assert_close(compiled_z, eager_z, atol=0.0, rtol=0.0)

    def test_dynamic_compile_survives_int32_stride_overflow_edge_counts(self):
        """A graph traced on a small system must run beyond 2^31 / ROW edges.

        Triton specializes scalar kernel arguments to int32 when the first
        compilation sees a small value, so any scalar that grows as
        ``n_edge * ROW`` overflows the launcher on large systems (observed
        at 1.1e7 edges, ``ROW = 224``).  The mixing-stack kernels (fp32 and
        fp16x3 alike) therefore derive their strides in-kernel from
        constexpr layout flags; this test pins that contract for both stack
        operators by tracing at 4e4 edges and running at 9.7e6 edges, where
        ``n_edge * ROW`` exceeds int32.
        """
        if torch.cuda.get_device_properties(0).total_memory < 60 * 2**30:
            self.skipTest("requires ~40 GB of free device memory")
        from deepmd.kernels.triton.sezm.so2_stack_fp16x3 import (
            mixing_stack_fp16x3,
        )
        from deepmd.kernels.triton.sezm.so2_value_path import (
            _mixing_stack_op,
        )

        lmax, focus_dim, n_focus = 2, 32, 1
        generator = torch.Generator(device="cuda").manual_seed(29)

        def stack_inputs(n_edge):
            m0, half = (lmax + 1) * focus_dim, lmax * focus_dim
            row = (3 * lmax + 1) * focus_dim
            u0 = torch.randn(n_focus, n_edge, row, device="cuda", generator=generator)
            alpha = (
                torch.rand(n_edge, n_focus, device="cuda", generator=generator) + 0.1
            )
            w0 = torch.randn(3, n_focus, m0, m0, device="cuda", generator=generator)
            bu = torch.randn(3, n_focus, half, half, device="cuda", generator=generator)
            bv = torch.randn(3, n_focus, half, half, device="cuda", generator=generator)
            w1 = torch.cat(
                [torch.cat([bu, bv], 3), torch.cat([-bv, bu], 3)], 2
            ).contiguous()
            gw = torch.randn(
                2, n_focus, focus_dim, half, device="cuda", generator=generator
            )
            return u0, alpha, w0 * 0.2, w1 * 0.2, gw * 0.3

        def make_fn(op):
            def fn(u0, alpha, w0, w1, gw):
                x_local, _ = op(u0, alpha, w0, w1, gw, lmax, focus_dim, True)
                return (x_local,)

            return fn

        for stack_op in (mixing_stack_fp16x3, _mixing_stack_op):
            with self.subTest(stack_op=getattr(stack_op, "__name__", str(stack_op))):
                fn = make_fn(stack_op)

                small = stack_inputs(40000)
                graph = make_fx(fn, tracing_mode="symbolic")(*small)
                compiled = torch.compile(graph, backend="inductor", dynamic=True)
                with torch.no_grad():
                    compiled(*small)
                    big = stack_inputs(9_700_000)
                    out_big = compiled(*big)[0]
                    eager_big = fn(*big)[0]
                torch.testing.assert_close(out_big, eager_big, atol=0.0, rtol=0.0)
                del big, out_big, eager_big
                torch.cuda.empty_cache()

    def test_value_path_selects_fp16x3_only_at_level_3(self):
        """The stack operator selection follows the gate level and the table."""
        import os
        from unittest import (
            mock,
        )

        from deepmd.kernels.triton.sezm.so2_stack_fp16x3 import (
            mixing_stack_fp16x3,
        )
        from deepmd.kernels.triton.sezm.so2_value_path import (
            _mixing_stack_op,
            make_triton_value_path,
        )
        from deepmd.pt.model.descriptor.sezm_nn.so2 import (
            SO2Convolution,
        )

        def build_conv():
            return SO2Convolution(
                lmax=self.LMAX,
                mmax=1,
                channels=self.FOCUS_DIM,
                n_focus=self.N_FOCUS,
                focus_dim=0,
                mixing_layers=3,
                radial_so2_mode="degree_channel",
                radial_so2_rank=1,
                n_atten_head=1,
                dtype=torch.float32,
                seed=7,
                trainable=False,
            )

        with mock.patch.dict(os.environ, {"DP_TRITON_INFER": "3"}):
            entry = make_triton_value_path(build_conv())
        self.assertIs(entry._stack_op, mixing_stack_fp16x3)

        with mock.patch.dict(os.environ, {"DP_TRITON_INFER": "2"}):
            entry = make_triton_value_path(build_conv())
        self.assertIs(entry._stack_op, _mixing_stack_op)

    def test_unswept_shape_has_no_config_and_operator_refuses_it(self):
        from deepmd.kernels.triton.sezm.so2_stack_fp16x3 import (
            mixing_stack_fp16x3,
        )
        from deepmd.kernels.triton.sezm.tile_configs import (
            stack_fp16x3_configs,
        )

        self.assertIsNone(stack_fp16x3_configs(48, 3))
        row = (3 * 3 + 1) * 48
        u0 = torch.zeros(1, 8, row, device="cuda")
        alpha = torch.ones(8, 1, device="cuda")
        w0 = torch.zeros(3, 1, 4 * 48, 4 * 48, device="cuda")
        w1 = torch.zeros(3, 1, 6 * 48, 6 * 48, device="cuda")
        gw = torch.zeros(2, 1, 48, 3 * 48, device="cuda")
        with self.assertRaises(RuntimeError):
            mixing_stack_fp16x3(u0, alpha, w0, w1, gw, 3, 48, True)


class _TileConfigRuntimeIsolation(unittest.TestCase):
    """Base fixture: snapshot and restore the process-local runtime tables."""

    def setUp(self) -> None:
        from deepmd.kernels.triton.sezm import (
            tile_configs,
        )

        self.tile_configs = tile_configs
        saved = {family: dict(table) for family, table in tile_configs._RUNTIME.items()}

        def restore() -> None:
            for family, table in tile_configs._RUNTIME.items():
                table.clear()
                table.update(saved[family])

        self.addCleanup(restore)


class TestTileConfigLookup(_TileConfigRuntimeIsolation):
    """Device-independent resolution semantics of the launch-config tables.

    Configurations resolve through the process-local runtime registrations,
    then the built-in tables of the running GPU, then the family fallback;
    ``has_tile_config`` distinguishes "swept, default won" (an explicit
    ``None`` entry) from "never swept" so the freeze auto-tuner only sweeps
    genuinely uncovered keys.  These semantics hold identically on hosts
    without CUDA, where the built-in layer is empty.
    """

    def test_runtime_registration_precedes_builtin_and_none_means_default(self):
        tc = self.tile_configs
        # An unswept key resolves to the family fallback and reports uncovered.
        self.assertEqual(tc.gate_config(48, 3), (16, 8, 2))
        self.assertFalse(tc.has_tile_config("gate", (48, 3)))
        # A registration serves lookups and marks the key covered.
        tc.register_tile_configs("gate", {(48, 3): (32, 4, 1)})
        self.assertEqual(tc.gate_config(48, 3), (32, 4, 1))
        self.assertTrue(tc.has_tile_config("gate", (48, 3)))
        # A None registration records "swept, default won": the lookup keeps
        # the fallback while the key counts as covered.
        tc.register_tile_configs("rotate_mix_fwd", {(96, 3): None})
        self.assertEqual(tc.rotate_mix_fwd_config(96, 3), (2, 2))
        self.assertTrue(tc.has_tile_config("rotate_mix_fwd", (96, 3)))
        # Unknown families are rejected on every entry point.
        with self.assertRaises(ValueError):
            tc.register_tile_configs("bogus", {})
        with self.assertRaises(ValueError):
            tc.has_tile_config("bogus", (32, 3))
        with self.assertRaises(ValueError):
            tc._runtime_tile_configs("bogus")


@_GPU_KERNELS
class TestTileConfigLayering(_TileConfigRuntimeIsolation):
    """GPU-bound layering behaviour: built-in dispatch, collection, tuning."""

    def test_unknown_gpu_resolves_every_family_to_its_fallback(self):
        from unittest import (
            mock,
        )

        tc = self.tile_configs
        tc._builtin_tables.cache_clear()
        self.addCleanup(tc._builtin_tables.cache_clear)
        with mock.patch.object(
            torch.cuda, "get_device_name", return_value="NVIDIA Imaginary GPU"
        ):
            self.assertEqual(tc.gate_config(32, 3), (16, 8, 2))
            self.assertEqual(tc.rotate_mix_fwd_config(64, 3), (2, 2))
            self.assertIsNone(tc.flash_bwd_block_config(64, 3))
            self.assertIsNone(tc.stack_fp16x3_configs(32, 3))
            self.assertFalse(tc.has_tile_config("gate", (32, 3)))
            # Runtime registrations still resolve on an untuned GPU.
            tc.register_tile_configs(
                "stack_fp16x3", {(32, 3): ((64, 64, 32, 4, 1),) * 4}
            )
            self.assertIsNotNone(tc.stack_fp16x3_configs(32, 3))
        tc._builtin_tables.cache_clear()

    def test_collect_model_shape_keys_reports_supported_convolutions(self):
        from deepmd.kernels.triton.sezm.sweep_tile_configs import (
            collect_model_shape_keys,
        )
        from deepmd.pt.model.descriptor.sezm_nn.so2 import (
            SO2Convolution,
        )

        def build_conv(**overrides):
            kwargs = {
                "lmax": 3,
                "mmax": 1,
                "channels": 32,
                "n_focus": 2,
                "focus_dim": 0,
                "mixing_layers": 3,
                "radial_so2_mode": "degree_channel",
                "radial_so2_rank": 1,
                "n_atten_head": 1,
                "dtype": torch.float32,
                "seed": 7,
                "trainable": False,
            }
            kwargs.update(overrides)
            return SO2Convolution(**kwargs)

        model = torch.nn.ModuleList(
            [
                build_conv(),
                build_conv(),  # duplicate shape, deduplicated
                build_conv(mmax=3),  # unsupported layout, contributes no key
            ]
        )
        self.assertEqual(collect_model_shape_keys(model), [(32, 3, 2, 1)])

    def test_tune_missing_configs_sweeps_only_uncovered_groups(self):
        from unittest import (
            mock,
        )

        from deepmd.kernels.triton.sezm import (
            sweep_tile_configs,
        )

        tc = self.tile_configs
        # Cover the pointwise and fp16x3 groups; leave the (C_wide, lmax)
        # groups uncovered so only they should be swept at level 2.
        tc.register_tile_configs("gate", {(48, 2): (16, 4, 1)})
        tc.register_tile_configs("stack_fp16x3", {(48, 2): None})
        calls: list[str] = []

        def fake_sweep(group, family, key):
            def run(cf, lmax, **kwargs):
                calls.append(group)
                return {family: {key: (4, 2, 1)}}

            return run

        fake_sweeps = {
            "pointwise": fake_sweep("pointwise", "gate", (48, 2)),
            "rotate_fwd": fake_sweep("rotate_fwd", "rotate_mix_fwd", (96, 2)),
            "rotate_bwd": fake_sweep("rotate_bwd", "rotate_mix_bwd_block", (96, 2)),
            "flash_bwd": fake_sweep("flash_bwd", "flash_bwd_block", (96, 2)),
            "fp16x3": fake_sweep("fp16x3", "stack_fp16x3", (48, 2)),
        }
        shape_keys = [(48, 2, 2, 1)]
        with mock.patch.dict(sweep_tile_configs._SWEEPS, fake_sweeps):
            registered = sweep_tile_configs.tune_missing_configs(
                shape_keys, level=2, device="cuda"
            )
        self.assertEqual(sorted(calls), ["flash_bwd", "rotate_bwd", "rotate_fwd"])
        self.assertEqual(
            sorted(registered),
            ["flash_bwd_block", "rotate_mix_bwd_block", "rotate_mix_fwd"],
        )
        # The registrations are now covered: a second tune is a no-op, and
        # level 3 adds only the fp16x3 group (whose key was pre-covered by
        # the explicit None above).
        calls.clear()
        with mock.patch.dict(sweep_tile_configs._SWEEPS, fake_sweeps):
            self.assertEqual(
                sweep_tile_configs.tune_missing_configs(
                    shape_keys, level=3, device="cuda"
                ),
                {},
            )
        self.assertEqual(calls, [])
        # Levels below 2 never sweep.
        with mock.patch.dict(sweep_tile_configs._SWEEPS, fake_sweeps):
            self.assertEqual(
                sweep_tile_configs.tune_missing_configs(
                    [(64, 5, 2, 1)], level=1, device="cuda"
                ),
                {},
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
