# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the hand-written CUDA operators of the SeZM / DPA4 inference path.

The references are dense transcriptions of the documented math, so a failure
localizes to the kernel rather than to another accelerated path. The Wigner
matrices are built block diagonal, which is the structure the model produces and
the contract the rotation kernels are written against.
"""

from __future__ import (
    annotations,
)

import unittest

import torch

try:
    # Loading the operator library is what registers ``torch.ops.deepmd``.
    import deepmd.pt.cxx_op  # noqa: F401
    from deepmd.pt.model.descriptor.sezm_nn.radial import (
        C3CutoffEnvelope,
        RadialBasis,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4 import (
        op_available,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.edge_radial import (
        make_cuda_edge_radial,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.edge_radial import (
        op_available as radial_op_available,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.grid_pair import (
        grid_pair,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.grid_pair import (
        op_available as grid_op_available,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv import (
        edge_csr,
        ensure_registered,
        wigner_run_tables,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.wigner_dense import (
        WignerDenseCuda,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.wigner_dense import (
        ensure_registered as wigner_dense_ensure_registered,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.wigner_dense import (
        op_available as wigner_dense_op_available,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
        op_available as zonal_op_available,
    )
    from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
        zonal_scatter,
    )

    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

CUDA_CONV = _IMPORT_OK and torch.cuda.is_available() and op_available()
CUDA_GRID = _IMPORT_OK and torch.cuda.is_available() and grid_op_available()
CUDA_ZONAL = _IMPORT_OK and torch.cuda.is_available() and zonal_op_available()
CUDA_RADIAL = _IMPORT_OK and torch.cuda.is_available() and radial_op_available()
CUDA_WIGNER = _IMPORT_OK and torch.cuda.is_available() and wigner_dense_op_available()

# Coefficient slots, channel width and grid size of every grid call site in the
# zoo: the SO(3) grids of degrees one to five, the degree-six grid the operator
# also carries, the matching S2 grid, and a second channel width.
GRID_SHAPES = (
    (9, 64, 32),
    (12, 32, 24),
    (27, 96, 104),
    (48, 64, 152),
    (75, 192, 296),
    (108, 128, 344),
    (147, 64, 440),
)

# Degree, focus width, focus streams, mixing layers, mixer rank, attention heads
# of the production model zoo.
ZOO_SHAPES = {
    "nano": (1, 32, 1, 3, 0, 1),
    "mini": (2, 32, 1, 3, 1, 1),
    "neo": (3, 32, 2, 3, 1, 1),
    "air": (3, 64, 1, 4, 1, 1),
    "plus": (4, 64, 1, 4, 2, 1),
    "pro": (5, 64, 2, 4, 2, 1),
}


def selected_wigner_rows(quat: torch.Tensor, lmax: int) -> torch.Tensor:
    """Reduced Wigner rows from the fitted run tables, shape (E, RED, DIM).

    The rows are evaluated through the same polynomial tables the operator
    consumes, so the comparison isolates the kernels rather than the fit.
    """
    from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv import (
        _monomial_exponents,
        _monomials,
    )

    tables = wigner_run_tables(lmax)
    exps = _monomial_exponents(2 * lmax).to(quat.device)
    run = _monomials(quat, exps) @ tables[0].t().to(quat.device)  # (E, NW)
    dim = (lmax + 1) ** 2
    red = 3 * lmax + 1
    rows = quat.new_zeros(quat.shape[0], red, dim)
    offset = 0
    for r in range(red):
        l = r if r <= lmax else (r - lmax if r <= 2 * lmax else r - 2 * lmax)
        width = 2 * l + 1
        rows[:, r, l * l : l * l + width] = run[:, offset : offset + width]
        offset += width
    return rows


class ConvCase:
    """Inputs of one fused-convolution evaluation and its dense reference."""

    def __init__(
        self,
        n_node: int = 29,
        degree: int = 17,
        lmax: int = 2,
        focus_dim: int = 32,
        n_focus: int = 1,
        n_layers: int = 3,
        rank: int = 1,
        n_head: int = 1,
        seed: int = 0,
        device: str = "cuda",
    ) -> None:
        torch.manual_seed(seed)
        self.lmax = lmax
        self.focus_dim = focus_dim
        self.n_focus = n_focus
        self.n_layers = n_layers
        self.rank = rank
        self.n_head = n_head
        self.dim = (lmax + 1) ** 2
        self.red = 3 * lmax + 1
        n_edge = n_node * degree
        c_wide = n_focus * focus_dim
        m0 = (lmax + 1) * focus_dim
        m1 = 2 * lmax * focus_dim
        gate = lmax * focus_dim
        kernel_size = (lmax + 1) ** 2 + lmax**2
        kc_len = (lmax + 1) * c_wide if rank == 0 else kernel_size * rank
        self.x = torch.randn(n_node, self.dim, c_wide, device=device) * 0.5
        self.src = torch.randint(0, n_node, (n_edge,), device=device)
        self.dst = torch.arange(n_node, device=device).repeat_interleave(degree)
        self.quat = torch.nn.functional.normalize(
            torch.randn(n_edge, 4, device=device), dim=1
        )
        self.kc = torch.randn(n_edge, kc_len, device=device) * 0.5
        self.cb = (
            torch.randn(max(rank, 1), c_wide, device=device) * 0.5 + 1.0
            if rank
            else torch.zeros(1, device=device)
        )
        self.w0 = torch.randn(n_layers, n_focus, m0, m0, device=device) / m0**0.5
        self.w1 = torch.randn(n_layers, n_focus, m1, m1, device=device) / m1**0.5
        self.gw = torch.randn(n_layers - 1, n_focus, focus_dim, gate, device=device)
        self.gw /= focus_dim**0.5
        self.q = torch.randn(n_node, c_wide, device=device) * 0.5
        self.k = torch.randn(n_node, c_wide, device=device) * 0.5
        self.logit_w = torch.randn(n_focus, focus_dim, n_head, device=device) * 0.3
        self.null_logit = torch.randn(n_focus, n_head, device=device) * 0.5
        self.env = torch.rand(n_edge, device=device) * 0.9
        self.rad0 = torch.randn(n_edge, c_wide, device=device) * 0.5
        # The cross-focus competition scale rides through the operator as a
        # post-softmax weight multiplier; a single stream passes it empty.
        self.fscale = (
            torch.rand(n_edge, n_focus, device=device) + 0.5
            if n_focus > 1
            else torch.empty(0, device=device)
        )
        self.head_gate = torch.rand(n_node, n_focus, n_head, device=device)
        self.rescale = torch.rand(self.dim, device=device) + 0.5

    @classmethod
    def of(cls, name: str, **kwargs) -> ConvCase:
        """Build the case of one named zoo shape."""
        lmax, focus_dim, n_focus, n_layers, rank, n_head = ZOO_SHAPES[name]
        return cls(
            lmax=lmax,
            focus_dim=focus_dim,
            n_focus=n_focus,
            n_layers=n_layers,
            rank=rank,
            n_head=n_head,
            **kwargs,
        )

    def _degree_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel degree kernels of the m=0 and m=+-1 blocks.

        Returns
        -------
        tuple of torch.Tensor
            Shapes (E, lmax+1, lmax+1, C_wide) and (E, lmax, lmax, C_wide),
            indexed ``[edge, in_degree, out_degree, channel]``.
        """
        lmax = self.lmax
        n_deg = lmax + 1
        n_edge, c_wide = self.src.shape[0], self.x.shape[2]
        if self.rank == 0:
            radial = self.kc.reshape(n_edge, n_deg, c_wide)
            k_m0 = torch.zeros(n_edge, n_deg, n_deg, c_wide, device=self.x.device)
            k_m1 = torch.zeros(n_edge, lmax, lmax, c_wide, device=self.x.device)
            for degree in range(n_deg):
                k_m0[:, degree, degree, :] = radial[:, degree, :]
            for degree in range(lmax):
                k_m1[:, degree, degree, :] = radial[:, degree + 1, :]
            return k_m0, k_m1
        compact = self.kc.reshape(n_edge, -1, self.rank)
        effective = torch.einsum("esr,rc->esc", compact, self.cb)
        split = n_deg * n_deg
        return (
            effective[:, :split, :].reshape(n_edge, n_deg, n_deg, c_wide),
            effective[:, split:, :].reshape(n_edge, lmax, lmax, c_wide),
        )

    def reference_alpha(self) -> torch.Tensor:
        """Envelope-gated segment softmax with a null mass, dense form."""
        n_edge = self.src.shape[0]
        n_node = self.x.shape[0]
        n_focus, n_head = self.n_focus, self.n_head
        head_dim = self.focus_dim // n_head
        q = self.q.reshape(n_node, n_focus, n_head, head_dim)
        k = self.k.reshape(n_node, n_focus, n_head, head_dim)
        logits = (q[self.dst] * k[self.src]).sum(-1) * head_dim**-0.5
        rad = self.rad0.reshape(n_edge, n_focus, self.focus_dim)
        logits = logits + torch.einsum("efi,fih->efh", rad, self.logit_w)
        eff = torch.where(
            (self.env > 0).view(n_edge, 1, 1),
            logits + 2.0 * torch.log(self.env.clamp_min(1e-30)).view(n_edge, 1, 1),
            torch.full_like(logits, float("-inf")),
        )
        null = self.null_logit.view(1, n_focus, n_head)
        group_max = null.expand(n_node, n_focus, n_head).clone()
        idx = self.dst.view(n_edge, 1, 1).expand_as(eff)
        group_max = torch.scatter_reduce(
            group_max, 0, idx, eff, reduce="amax", include_self=True
        )
        edge_exp = torch.exp(eff - group_max[self.dst])
        denom = torch.zeros_like(group_max).scatter_add_(0, idx, edge_exp)
        denom = denom + torch.exp(null - group_max)
        alpha = edge_exp / denom[self.dst]
        if self.fscale.numel() > 0:
            alpha = alpha * self.fscale.unsqueeze(-1)
        return alpha

    def reference(self) -> torch.Tensor:
        """Dense transcription of the fused convolution."""
        lmax, cf, n_focus = self.lmax, self.focus_dim, self.n_focus
        n_deg = lmax + 1
        m0 = n_deg * cf
        n_edge = self.src.shape[0]
        n_node, c_wide = self.x.shape[0], self.x.shape[2]
        dsel = selected_wigner_rows(self.quat, self.lmax)
        x_local = torch.bmm(dsel, self.x[self.src])  # (E, RED, C_wide)

        # === Step 1. Radial degree mixing inside each order block ===
        k_m0, k_m1 = self._degree_kernels()
        mixed = torch.zeros_like(x_local)
        for out_deg in range(n_deg):
            mixed[:, out_deg, :] = sum(
                k_m0[:, in_deg, out_deg, :] * x_local[:, in_deg, :]
                for in_deg in range(n_deg)
            )
        for out_deg in range(lmax):
            mixed[:, n_deg + out_deg, :] = sum(
                k_m1[:, in_deg, out_deg, :] * x_local[:, n_deg + in_deg, :]
                for in_deg in range(lmax)
            )
            mixed[:, n_deg + lmax + out_deg, :] = sum(
                k_m1[:, in_deg, out_deg, :] * x_local[:, n_deg + lmax + in_deg, :]
                for in_deg in range(lmax)
            )

        # === Step 2. Gated mixing stack over the focus-major row ===
        u = (
            mixed.reshape(n_edge, self.red, n_focus, cf)
            .permute(0, 2, 1, 3)
            .reshape(n_edge, n_focus, -1)
        )
        for layer in range(self.n_layers):
            z0 = torch.einsum("efi,fio->efo", u[:, :, :m0], self.w0[layer])
            z1 = torch.einsum("efi,fio->efo", u[:, :, m0:], self.w1[layer])
            if layer < self.n_layers - 1:
                sig = torch.sigmoid(
                    torch.einsum("efi,fio->efo", z0[:, :, :cf], self.gw[layer])
                )
                add0 = torch.cat(
                    [
                        z0[:, :, :cf] * torch.sigmoid(z0[:, :, :cf]),
                        z0[:, :, cf:] * sig,
                    ],
                    -1,
                )
                u = u + torch.cat([add0, z1 * sig.repeat(1, 1, 2)], -1)
            else:
                u = u + torch.cat([z0, z1], -1)

        # === Step 3. Inverse rotation, attention weighting, destination sum ===
        u_red = (
            u.reshape(n_edge, n_focus, self.red, cf)
            .permute(0, 2, 1, 3)
            .reshape(n_edge, self.red, c_wide)
        )
        rotated_back = torch.bmm(dsel.transpose(1, 2), u_red)  # (E, DIM, C_wide)
        head_dim = cf // self.n_head
        weight = (
            self.reference_alpha()
            .reshape(n_edge, n_focus, self.n_head, 1)
            .expand(n_edge, n_focus, self.n_head, head_dim)
            .reshape(n_edge, 1, c_wide)
        )
        pre_gate = torch.zeros(
            n_node, self.dim, c_wide, device=self.x.device, dtype=self.x.dtype
        )
        pre_gate.index_add_(0, self.dst, rotated_back * weight)
        pre_gate = pre_gate * self.rescale.view(1, -1, 1)

        # === Step 4. Output-side head gate ===
        gate = (
            self.head_gate.reshape(n_node, n_focus, self.n_head, 1)
            .expand(n_node, n_focus, self.n_head, head_dim)
            .reshape(n_node, 1, c_wide)
        )
        return pre_gate * gate

    def inputs(self) -> tuple[torch.Tensor, ...]:
        """The tensor arguments of the forward operator, in order."""
        n_node = self.x.shape[0]
        csr = edge_csr(self.dst, n_node) + edge_csr(self.src, n_node)
        tables = wigner_run_tables(self.lmax)
        runs = torch.ops.deepmd.dpa4_wigner_runs(
            self.quat,
            tables[0].to(self.quat.device),
            tables[2].to(self.quat.device),
            self.lmax,
        )
        return (
            self.x,
            self.src,
            self.dst,
            *csr,
            runs,
            self.kc,
            self.cb,
            self.w0,
            self.w1,
            self.gw,
            self.q,
            self.k,
            self.logit_w,
            self.null_logit,
            self.env,
            self.rad0,
            self.fscale,
            self.head_gate,
            self.rescale,
        )

    def fused(self) -> tuple[torch.Tensor, ...]:
        ensure_registered()
        return torch.ops.deepmd.dpa4_so2_conv(
            *self.inputs(), self.lmax, self.focus_dim, self.rank
        )


@unittest.skipUnless(CUDA_CONV, "requires the CUDA dpa4_so2_conv operator")
class TestSeZMConvCuda(unittest.TestCase):
    """Numerical contract of the fused SO(2) convolution."""

    def test_rejects_multiple_attention_heads(self) -> None:
        case = ConvCase(n_head=2, focus_dim=64)
        with self.assertRaisesRegex(RuntimeError, "supports one attention head"):
            case.fused()

    def test_zero_edge_forward_and_backward(self) -> None:
        case = ConvCase(n_node=5, degree=0)
        leaves = ("x", "quat", "kc", "q", "k", "env", "rad0", "head_gate")
        for name in leaves:
            setattr(case, name, getattr(case, name).detach().requires_grad_(True))

        out, alpha, _, _ = case.fused()
        torch.testing.assert_close(out, torch.zeros_like(out))
        self.assertEqual(alpha.shape[0], 0)
        gradients = torch.autograd.grad(
            out,
            [getattr(case, name) for name in leaves],
            torch.randn_like(out),
        )
        for name, gradient in zip(leaves, gradients, strict=True):
            with self.subTest(gradient=name):
                self.assertEqual(gradient.shape, getattr(case, name).shape)
                self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_zero_envelope_has_zero_finite_gradient(self) -> None:
        case = ConvCase(n_node=7, degree=3)
        case.env[:4] = 0.0
        case.env.requires_grad_(True)

        out = case.fused()[0]
        (grad_env,) = torch.autograd.grad(out, case.env, torch.randn_like(out))
        self.assertTrue(torch.isfinite(grad_env).all())
        torch.testing.assert_close(grad_env[:4], torch.zeros_like(grad_env[:4]))

    def test_forward_matches_dense_reference_on_every_zoo_shape(self) -> None:
        for name in ZOO_SHAPES:
            with self.subTest(model=name):
                case = ConvCase.of(name)
                want = case.reference()
                out = case.fused()
                scale = want.abs().max()
                self.assertLess(((out[0] - want).abs().max() / scale).item(), 5e-6)
                want_alpha = case.reference_alpha()
                self.assertLess(
                    ((out[1] - want_alpha).abs().max() / want_alpha.abs().max()).item(),
                    5e-6,
                )

    def test_forward_survives_a_short_tail_chunk(self) -> None:
        # A degree below the chunk width exercises the padded edge groups, whose
        # slots alias edge zero.
        for degree in (1, 7, 32, 33):
            with self.subTest(degree=degree):
                case = ConvCase(n_node=11, degree=degree)
                want = case.reference()
                got = case.fused()[0]
                scale = want.abs().max()
                self.assertLess(((got - want).abs().max() / scale).item(), 5e-6)

    def test_backward_matches_autograd_on_the_reference(self) -> None:
        # Driving the registered autograd end to end covers the kernel backward
        # and the softmax and logit cotangents assembled around it.
        base = ("x", "quat", "kc", "q", "k", "env", "rad0", "head_gate")
        for name in ("nano", "mini", "neo", "plus"):
            with self.subTest(model=name):
                case = ConvCase.of(name)
                leaves = base + (("fscale",) if case.fscale.numel() else ())
                for leaf in leaves:
                    setattr(
                        case, leaf, getattr(case, leaf).detach().requires_grad_(True)
                    )
                out = case.reference()
                grad_out = torch.randn_like(out)
                want = torch.autograd.grad(
                    out, [getattr(case, leaf) for leaf in leaves], grad_out
                )
                for leaf in leaves:
                    setattr(case, leaf, getattr(case, leaf).detach())

                for leaf in leaves:
                    setattr(
                        case, leaf, getattr(case, leaf).detach().requires_grad_(True)
                    )
                fused_out = case.fused()[0]
                got = torch.autograd.grad(
                    fused_out, [getattr(case, leaf) for leaf in leaves], grad_out
                )
                for leaf in leaves:
                    setattr(case, leaf, getattr(case, leaf).detach())

                for leaf, got_g, want_g in zip(leaves, got, want, strict=True):
                    with self.subTest(gradient=leaf):
                        scale = want_g.abs().max()
                        self.assertLess(
                            ((got_g - want_g).abs().max() / scale).item(), 5e-6
                        )

    def test_reduction_is_reproducible(self) -> None:
        case = ConvCase()
        first = case.fused()[0]
        second = case.fused()[0]
        self.assertTrue(torch.equal(first, second))

    def test_make_fx_traces_the_operator(self) -> None:
        from torch.fx.experimental.proxy_tensor import (
            make_fx,
        )

        case = ConvCase(n_node=7, degree=5)
        ensure_registered()

        def run(*args: torch.Tensor) -> torch.Tensor:
            return torch.ops.deepmd.dpa4_so2_conv(
                *args, case.lmax, case.focus_dim, case.rank
            )[0]

        graph = make_fx(run, tracing_mode="symbolic")(*case.inputs())
        names = {str(node.target) for node in graph.graph.nodes}
        self.assertTrue(any("dpa4_so2_conv" in name for name in names))


@unittest.skipUnless(CUDA_GRID, "requires the CUDA dpa4_grid_pair operator")
class TestSeZMGridPairCuda(unittest.TestCase):
    """Numerical contract of the fused grid pair product."""

    def _case(self, n_node: int, p_dim: int, channels: int, n_grid: int) -> tuple:
        torch.manual_seed(0)
        dev = "cuda"
        left = torch.randn(n_node, p_dim, channels, device=dev) * 0.5
        right = torch.randn(n_node, p_dim, channels, device=dev) * 0.5
        to_grid = torch.randn(n_grid, p_dim, device=dev) / p_dim**0.5
        from_grid_t = torch.randn(n_grid, p_dim, device=dev) / n_grid**0.5
        return left, right, to_grid, from_grid_t

    @staticmethod
    def _reference(
        left: torch.Tensor,
        right: torch.Tensor,
        to_grid: torch.Tensor,
        from_grid_t: torch.Tensor,
    ) -> torch.Tensor:
        lg = torch.einsum("gp,npc->ngc", to_grid, left)
        rg = torch.einsum("gp,npc->ngc", to_grid, right)
        return torch.einsum("gp,ngc->npc", from_grid_t, lg * rg)

    def test_forward_matches_the_projector_composition(self) -> None:
        # Every coefficient-slot count of the zoo, at the grid size that ships
        # with it, plus the S2 grid and a second channel width.
        for p_dim, channels, n_grid in GRID_SHAPES:
            with self.subTest(p_dim=p_dim, channels=channels):
                left, right, to_grid, from_grid_t = self._case(
                    23, p_dim, channels, n_grid
                )
                got = grid_pair(left, right, to_grid, from_grid_t)
                want = self._reference(left, right, to_grid, from_grid_t)
                scale = want.abs().max()
                self.assertLess(((got - want).abs().max() / scale).item(), 5e-6)

    def test_backward_matches_autograd_on_every_slot_count(self) -> None:
        for p_dim, channels, n_grid in GRID_SHAPES:
            with self.subTest(p_dim=p_dim, channels=channels):
                left, right, to_grid, from_grid_t = self._case(
                    23, p_dim, channels, n_grid
                )
                left = left.requires_grad_(True)
                right = right.requires_grad_(True)
                want = self._reference(left, right, to_grid, from_grid_t)
                grad_out = torch.randn_like(want)
                want.backward(grad_out)
                g_left, g_right = torch.ops.deepmd.dpa4_grid_pair_backward(
                    grad_out, left.detach(), right.detach(), to_grid, from_grid_t
                )
                for got, ref in ((g_left, left.grad), (g_right, right.grad)):
                    scale = ref.abs().max()
                    self.assertLess(((got - ref).abs().max() / scale).item(), 5e-6)


@unittest.skipUnless(CUDA_ZONAL, "requires the CUDA dpa4_zonal_scatter operator")
class TestSeZMZonalScatterCuda(unittest.TestCase):
    """Numerical contract of the fused geometric initial embedding."""

    @staticmethod
    def _case(lmax: int, n_node: int, degree: int, channels: int) -> tuple:
        torch.manual_seed(0)
        dev = "cuda"
        n_row = (lmax + 1) ** 2 - 1
        n_edge = n_node * degree
        zonal = torch.randn(n_edge, n_row, device=dev)
        radial = torch.randn(n_edge, lmax, channels, device=dev)
        dst = torch.randint(0, n_node, (n_edge,), device=dev)
        scale = torch.rand(n_node, device=dev) + 0.5
        order, row_ptr = edge_csr(dst, n_node)
        # Degree l holds 2l+1 packed rows and reads radial slot l-1.
        slot = torch.tensor(
            [l - 1 for l in range(1, lmax + 1) for _ in range(2 * l + 1)],
            dtype=torch.long,
            device=dev,
        )
        return zonal, radial, dst, order, row_ptr, scale, slot

    @staticmethod
    def _reference(
        zonal: torch.Tensor,
        radial: torch.Tensor,
        slot: torch.Tensor,
        dst: torch.Tensor,
        scale: torch.Tensor,
        n_node: int,
    ) -> torch.Tensor:
        message = zonal.unsqueeze(-1) * radial.index_select(1, slot)  # (E, R, C)
        acc = message.new_zeros(n_node, zonal.shape[1], radial.shape[2])
        acc = acc.index_add_(0, dst, message)
        pad = acc.new_zeros(n_node, 1, radial.shape[2])
        return torch.cat([pad, acc], dim=1) * scale.reshape(-1, 1, 1)

    def test_forward_matches_the_message_composition(self) -> None:
        for lmax in range(1, 7):
            with self.subTest(lmax=lmax):
                n_node = 37
                zonal, radial, dst, order, row_ptr, scale, slot = self._case(
                    lmax, n_node, 7, 32
                )
                got = zonal_scatter(zonal, radial, dst, order, row_ptr, scale, n_node)
                want = self._reference(zonal, radial, slot, dst, scale, n_node)
                self.assertEqual(got.shape, want.shape)
                rel = ((got - want).abs().max() / want.abs().max()).item()
                self.assertLess(rel, 5e-6)

    def test_backward_matches_autograd_at_every_degree(self) -> None:
        for lmax in range(1, 7):
            with self.subTest(lmax=lmax):
                n_node = 37
                zonal, radial, dst, _, _, scale, slot = self._case(lmax, n_node, 7, 32)
                zonal = zonal.requires_grad_(True)
                radial = radial.requires_grad_(True)
                want = self._reference(zonal, radial, slot, dst, scale, n_node)
                grad_out = torch.randn_like(want)
                want.backward(grad_out)
                g_zonal, g_radial = torch.ops.deepmd.dpa4_zonal_scatter_backward(
                    grad_out, zonal.detach(), radial.detach(), dst, scale
                )
                for got, ref in ((g_zonal, zonal.grad), (g_radial, radial.grad)):
                    rel = ((got - ref).abs().max() / ref.abs().max()).item()
                    self.assertLess(rel, 5e-6)

    def test_backward_reaches_the_degree_normalization(self) -> None:
        # The normalization descends from the cutoff envelope, so its cotangent
        # is part of the force. An operator that folds the scaling in and
        # returns no gradient for it passes every other check here.
        n_node = 37
        zonal, radial, dst, order, row_ptr, scale, slot = self._case(2, n_node, 7, 32)
        scale = scale.requires_grad_(True)
        want = self._reference(zonal, radial, slot, dst, scale, n_node)
        grad_out = torch.randn_like(want)
        want.backward(grad_out)

        fused_scale = scale.detach().requires_grad_(True)
        got = zonal_scatter(zonal, radial, dst, order, row_ptr, fused_scale, n_node)
        got.backward(grad_out)
        rel = (
            (fused_scale.grad - scale.grad).abs().max() / scale.grad.abs().max()
        ).item()
        self.assertLess(rel, 5e-6)

    def test_channel_widths_beyond_one_block(self) -> None:
        # A lane owns one channel of a 32-wide block, so wider features sweep
        # the edge list more than once.
        n_node = 19
        for channels in (32, 64, 128):
            with self.subTest(channels=channels):
                zonal, radial, dst, order, row_ptr, scale, slot = self._case(
                    2, n_node, 5, channels
                )
                got = zonal_scatter(zonal, radial, dst, order, row_ptr, scale, n_node)
                want = self._reference(zonal, radial, slot, dst, scale, n_node)
                rel = ((got - want).abs().max() / want.abs().max()).item()
                self.assertLess(rel, 5e-6)


@unittest.skipUnless(CUDA_WIGNER, "requires the CUDA dpa4_wigner_dense operator")
class TestSeZMWignerDenseCuda(unittest.TestCase):
    """Numerical contract of the fused dense Wigner-D build.

    The reference is the module calculator itself: the operator's fitted
    tables must reproduce it element-wise, transpose included, and its
    quaternion cotangent must match reference autograd once both paths share
    the upstream normalization that projects the radial component out.
    """

    @staticmethod
    def _paths(lmax: int, n_edge: int) -> tuple:
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            WignerDCalculator,
        )

        torch.manual_seed(2026 + lmax)
        wigner_dense_ensure_registered()
        raw = torch.randn(n_edge, 4, device="cuda")
        calc = WignerDCalculator(lmax=lmax, dtype=torch.float32).to("cuda")
        return raw, calc, WignerDenseCuda(lmax)

    def test_forward_matches_the_reference_calculator(self) -> None:
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            quaternion_normalize,
        )

        for lmax in range(1, 7):
            with self.subTest(lmax=lmax):
                raw, calc, fused = self._paths(lmax, 4096)
                quat = quaternion_normalize(raw)
                d_ref, dt_ref = calc(quat)
                d_got, dt_got = fused(quat)
                self.assertEqual(d_got.shape, d_ref.shape)
                self.assertLess((d_got - d_ref).abs().max().item(), 5e-6)
                self.assertLess((dt_got - dt_ref).abs().max().item(), 5e-6)

    def test_backward_matches_autograd_through_the_normalization(self) -> None:
        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            quaternion_normalize,
        )

        for lmax in range(1, 7):
            with self.subTest(lmax=lmax):
                raw, calc, fused = self._paths(lmax, 2048)
                g_d = torch.randn(
                    raw.shape[0], (lmax + 1) ** 2, (lmax + 1) ** 2, device="cuda"
                )
                g_dt = torch.randn_like(g_d)

                raw_ref = raw.clone().requires_grad_(True)
                torch.autograd.backward(
                    calc(quaternion_normalize(raw_ref)), [g_d, g_dt]
                )
                raw_got = raw.clone().requires_grad_(True)
                torch.autograd.backward(
                    fused(quaternion_normalize(raw_got)), [g_d, g_dt]
                )
                scale = raw_ref.grad.abs().max().item()
                rel = (raw_got.grad - raw_ref.grad).abs().max().item() / scale
                self.assertLess(rel, 5e-5)

    def test_make_fx_traces_the_operator(self) -> None:
        from torch.fx.experimental.proxy_tensor import (
            make_fx,
        )

        from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
            quaternion_normalize,
        )

        _, _, fused = self._paths(2, 64)

        def graph(quat: torch.Tensor, *tables: torch.Tensor) -> torch.Tensor:
            d_full, dt_full = torch.ops.deepmd.dpa4_wigner_dense(
                quaternion_normalize(quat), *tables, 2
            )
            return d_full.square().sum() + dt_full.square().sum()

        quat = torch.randn(64, 4, device="cuda")
        tables = fused.tables(quat.device)
        traced = make_fx(graph, tracing_mode="symbolic")(quat, *tables)
        self.assertTrue(
            any(
                "dpa4_wigner_dense" in str(node.target)
                for node in traced.graph.nodes
                if node.op == "call_function"
            )
        )
        got = traced(quat, *tables).item()
        self.assertLess(abs(got - graph(quat, *tables).item()), 1e-2)


@unittest.skipUnless(CUDA_RADIAL, "requires the CUDA dpa4_edge_radial operator")
class TestSeZMEdgeRadialCuda(unittest.TestCase):
    """Numerical contract of the fused cutoff envelope and radial basis."""

    RCUT = 6.0

    def _modules(self, basis_type: str, n_radial: int = 16) -> tuple:
        envelope = C3CutoffEnvelope(
            rcut=self.RCUT, exponent=5, dtype=torch.float32
        ).cuda()
        basis = RadialBasis(
            rcut=self.RCUT,
            basis_type=basis_type,
            n_radial=n_radial,
            exponent=7,
            dtype=torch.float32,
        ).cuda()
        return envelope, basis, make_cuda_edge_radial(envelope, basis)

    @staticmethod
    def _distances(n_edge: int, rcut: float) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        # Spanning past the cutoff exercises the clamped branch, where both the
        # envelope and its derivative vanish.
        r = torch.rand(n_edge, 1, device="cuda") * (rcut * 1.1) + 1e-2
        keep = (torch.rand(n_edge, 1, device="cuda") > 0.1).float()
        return r, keep

    def test_forward_matches_the_module_composition(self) -> None:
        for basis_type in ("bessel", "gaussian"):
            with self.subTest(basis=basis_type):
                envelope, basis, fused = self._modules(basis_type)
                self.assertIsNotNone(fused)
                r, keep = self._distances(4096, self.RCUT)
                got_env, got_rbf = fused(r, keep)
                want_env = envelope(r) * keep
                want_rbf = basis(r) * keep
                for got, want in ((got_env, want_env), (got_rbf, want_rbf)):
                    rel = ((got - want).abs().max() / want.abs().max()).item()
                    self.assertLess(rel, 5e-6)

    def test_backward_matches_autograd(self) -> None:
        for basis_type in ("bessel", "gaussian"):
            with self.subTest(basis=basis_type):
                envelope, basis, fused = self._modules(basis_type)
                r, keep = self._distances(4096, self.RCUT)
                r = r.requires_grad_(True)
                grad_env = torch.randn_like(r)
                grad_rbf = torch.randn(r.shape[0], 16, device="cuda")
                ((envelope(r) * keep) * grad_env).sum().backward(retain_graph=True)
                ((basis(r) * keep) * grad_rbf).sum().backward()
                env_series, rbf_series = fused.series(r.device)
                got = torch.ops.deepmd.dpa4_edge_radial_backward(
                    grad_env,
                    grad_rbf,
                    r.detach(),
                    keep,
                    basis.adam_freqs,
                    env_series,
                    rbf_series,
                    self.RCUT,
                    float(basis.gaussian_coeff),
                    0 if basis_type == "bessel" else 1,
                )
                # The Bessel derivative subtracts two terms that cancel to
                # leading order at large ``r f``, so its cotangent carries more
                # rounding than the values do.
                rel = (
                    (got.reshape(-1) - r.grad.reshape(-1)).abs().max()
                    / r.grad.abs().max()
                ).item()
                self.assertLess(rel, 5e-5)

    def test_declines_a_mismatched_cutoff(self) -> None:
        envelope = C3CutoffEnvelope(rcut=5.0, exponent=5, dtype=torch.float32)
        basis = RadialBasis(
            rcut=6.0,
            basis_type="bessel",
            n_radial=16,
            exponent=7,
            dtype=torch.float32,
        )
        self.assertIsNone(make_cuda_edge_radial(envelope, basis))


@unittest.skipUnless(_IMPORT_OK, "requires the pt_expt CUDA bindings")
class TestSeZMConvCudaGate(unittest.TestCase):
    """The factory declines shapes the operator does not serve."""

    @staticmethod
    def _conv(n_head: int):
        from deepmd.pt.model.descriptor.sezm_nn.so2 import (
            SO2Convolution,
        )

        return SO2Convolution(
            lmax=2,
            mmax=1,
            channels=64,
            n_focus=1,
            focus_dim=64,
            mixing_layers=3,
            radial_so2_mode="degree_channel",
            radial_so2_rank=1,
            n_atten_head=n_head,
            dtype=torch.float32,
            seed=7,
            trainable=False,
        )

    def test_supports_only_one_attention_head(self) -> None:
        from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv import (
            _is_supported,
        )

        self.assertTrue(_is_supported(self._conv(n_head=1)))
        self.assertFalse(_is_supported(self._conv(n_head=2)))

    def test_declines_an_unsupported_focus_width(self) -> None:
        from deepmd.pt_expt.kernels.cuda.dpa4 import (
            make_cuda_so2_conv,
        )

        class Stub:
            mmax = 1
            lmax = 2
            n_focus = 1
            mixing_layers = 3
            n_atten_head = 1
            # Instantiations exist for widths 32 and 64 only.
            so2_focus_dim = 48
            node_wise_grid_product = None
            attn_focus_mix = None
            use_so2_attn_res = False
            layer_scale = False
            focus_compete = False
            edge_cartesian = False
            radial_degree_mixer = None
            so2_inter_norms: tuple = ()
            so2_linears: tuple = ()
            non_linearities: tuple = ()

        self.assertIsNone(make_cuda_so2_conv(Stub()))


if __name__ == "__main__":
    unittest.main()
