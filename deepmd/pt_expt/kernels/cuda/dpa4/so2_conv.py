# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings and model entry for the fused DPA4 / SeZM SO(2) convolution.

The CUDA operator ``deepmd::dpa4_so2_conv`` (see
``source/op/pt/dpa4/so2_conv.cu``) evaluates the whole per-edge span of one
``SO2Convolution``: the attention logits and their envelope-gated segment
softmax, the Wigner rotation into the edge frame, the radial degree mixer, the
gated mixing stack, the inverse rotation, the attention weighting, the
destination reduction and the output-side head gate. It replaces the
composition of the attention-weight build, ``so2_rotate_mix``,
``so2_mixing_stack`` and ``flash_atten_aggregate``, and none of the per-edge
intermediates of that composition reach device memory.

Supported configuration
-----------------------
``mmax == 1``, degree 1 to 6, focus width 32 or 64, any focus-stream count, at
least 32 channels per attention head, an attention layout matching the value
stream, two or more mixing layers with an identity final layer, and a radial
mixer that is either absent or ``degree_channel`` of any rank. The kernels are
templated on degree and focus width only; every other dimension is a runtime
argument. The bridging-mode source gate reshapes the softmax normalization and
is declined at call time.

Usage and pitfalls
------------------
* The CSR views of both endpoints are built once per step with :func:`edge_csr`
  and cached on the edge cache; the source-major pair rides through the forward
  only so the autograd context can hand it to the backward. Stable sorting is
  what fixes the summation order and keeps the reductions bitwise reproducible.
* The operator computes the attention weights itself with an online softmax
  whose running maximum starts at the null-mass logit, matching the reference
  ``segment_envelope_gated_softmax`` exactly, and emits the finished weights as
  an output. The backward consumes those weights in the kernel and assembles
  the softmax, logit, query, key, radial-bias and envelope cotangents from
  plain tensor operations that the compile pipeline fuses.
* ``alpha``, ``pre_gate`` and ``z_all`` are auxiliary outputs and never receive
  a real gradient; ``set_materialize_grads(False)`` skips their zero fill.
* The stacked weights are assembled from the live parameters on every call and
  must not be cached: the first call may run inside a ``make_fx`` fake-tensor
  trace, and eager weights change when a checkpoint is loaded after
  construction. The assembly is a short chain of parameter-only aten ops that
  the compile pipeline constant-folds out of the hot path.
* The rotation contracts only the degree-block entries of the Wigner matrix.
  That is exact for the block-diagonal Wigner-D the model builds and is the same
  contract the Triton flash aggregation already uses for its inverse rotation;
  it differs from a dense random matrix.
* Cross-focus competition scales the finished weight outside the softmax, so it
  rides through the kernel as an optional per-``(edge, focus)`` multiplier and
  the backward splits its cotangent from the raw softmax weight.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch

__all__ = [
    "SO2ConvCuda",
    "ensure_registered",
    "make_cuda_so2_conv",
    "op_available",
]

_registered = False

_RUN_TABLE_CACHE: dict[int, tuple[torch.Tensor, ...]] = {}


def _reduced_rows(lmax: int) -> list[int]:
    """Row indices of the packed run: ``m = 0``, then ``m = -1``, then ``m = +1``."""
    rows = [l * l + l for l in range(lmax + 1)]
    rows += [l * l + l - 1 for l in range(1, lmax + 1)]
    rows += [l * l + l + 1 for l in range(1, lmax + 1)]
    return rows


def _monomial_exponents(degree: int) -> torch.Tensor:
    """Exponent tuples of every quaternion monomial of the given total degree.

    Returns
    -------
    torch.Tensor
        Exponents with shape (M, 4), int64.
    """
    exps = [
        (a, b, c, degree - a - b - c)
        for a in range(degree + 1)
        for b in range(degree + 1 - a)
        for c in range(degree + 1 - a - b)
    ]
    return torch.tensor(exps, dtype=torch.long, device="cpu")


def _monomials(q: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """Evaluate the monomial basis of quaternions, shape (E, M)."""
    out = torch.ones(q.shape[0], exps.shape[0], dtype=q.dtype, device=q.device)
    for i in range(4):
        powers = q[:, i : i + 1] ** torch.arange(
            int(exps[:, i].max()) + 1, device=q.device, dtype=q.dtype
        )
        out = out * powers[:, exps[:, i]]
    return out


def wigner_run_tables(lmax: int) -> tuple[torch.Tensor, ...]:
    """
    Polynomial tables that map a unit quaternion onto the packed Wigner run.

    Every entry of the packed block-diagonal run of degree ``l`` is a
    homogeneous polynomial of degree ``2 l`` in the quaternion; multiplying by
    powers of ``|q|^2 = 1`` lifts all entries onto the single degree
    ``2 lmax`` monomial basis. The coefficients are fitted once per degree in
    fp64 against the reference calculator (residual below 1e-11 across the
    supported degrees), the derivative tables follow by exact exponent
    manipulation, and the extension ambiguity off the unit sphere is
    immaterial because the quaternion normalization upstream projects the
    radial gradient component out.

    Parameters
    ----------
    lmax : int
        Maximum spherical-harmonic degree of the run.

    Returns
    -------
    tuple of torch.Tensor
        ``(mono_coeff, dmono_coeff, mono_exp, dmono_exp)`` on the CPU, with
        shapes (NW, M) fp32, (NW, 4, M') fp32, (M, 4) int8 and (M', 4) int8.
    """
    cached = _RUN_TABLE_CACHE.get(lmax)
    if cached is not None:
        return cached
    from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
        WignerDCalculator,
        quaternion_normalize,
    )

    calc = WignerDCalculator(lmax=lmax, dtype=torch.float64)
    # The calculator's constant buffers follow the package's default device, so
    # the fit runs there and the finished tables are kept on the CPU.
    device = next(calc.buffers()).device if any(True for _ in calc.buffers()) else "cpu"
    exps = _monomial_exponents(2 * lmax)
    dexps = _monomial_exponents(2 * lmax - 1)

    generator = torch.Generator().manual_seed(2026)
    n_fit = max(4 * exps.shape[0], 4096)
    q = quaternion_normalize(
        torch.randn(n_fit, 4, dtype=torch.float64, generator=generator, device="cpu")
    ).to(device)
    d_full, _ = calc(q)
    pieces = []
    for r in _reduced_rows(lmax):
        l = int(r**0.5)
        pieces.append(d_full[:, r, l * l : l * l + 2 * l + 1])
    run = torch.cat(pieces, dim=1)  # (n_fit, NW)
    coeff = torch.linalg.lstsq(
        _monomials(q, exps.to(device)), run
    ).solution.cpu()  # (M, NW)

    index = {tuple(int(v) for v in e): i for i, e in enumerate(dexps)}
    dcoeff = torch.zeros(
        4, dexps.shape[0], run.shape[1], dtype=torch.float64, device="cpu"
    )
    for m, e in enumerate(exps):
        e = [int(v) for v in e]
        for i in range(4):
            if e[i] > 0:
                lower = list(e)
                lower[i] -= 1
                dcoeff[i, index[tuple(lower)]] += e[i] * coeff[m]

    # The run coefficients are stored slot major so the in-kernel reduction
    # over the basis reads each row contiguously.
    tables = (
        coeff.float().t().contiguous(),
        dcoeff.float().permute(2, 0, 1).contiguous(),
        exps.to(torch.int8).contiguous(),
        dexps.to(torch.int8).contiguous(),
    )
    _RUN_TABLE_CACHE[lmax] = tables
    return tables


_SUPPORTED_FOCUS_DIMS = (32, 64)
_MAX_LMAX = 6


def edge_csr(key: torch.Tensor, n_node: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the CSR view of one endpoint array.

    Parameters
    ----------
    key : torch.Tensor
        Endpoint indices with shape (E,).
    n_node : int
        Number of nodes the endpoints index into.

    Returns
    -------
    tuple of torch.Tensor
        The stable sorting permutation with shape (E,) and the row pointer with
        shape (n_node + 1,). Stability fixes the within-segment edge order, which
        is what makes the operator's segment reductions bitwise reproducible.
    """
    order = torch.argsort(key, dim=0, stable=True)
    counts = torch.bincount(key, minlength=n_node)
    row_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
    return order, row_ptr


def op_available() -> bool:
    """Whether the C++ ``deepmd::dpa4_so2_conv`` op is loaded."""
    op = getattr(torch.ops.deepmd, "dpa4_so2_conv", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def _runs_fake(
    quat: torch.Tensor,
    mono_coeff: torch.Tensor,
    mono_exp: torch.Tensor,
    lmax: int,
) -> torch.Tensor:
    del mono_exp, lmax
    return quat.new_empty(quat.shape[0], mono_coeff.shape[0])


def _runs_backward_fake(
    grad_runs: torch.Tensor,
    quat: torch.Tensor,
    dmono_coeff: torch.Tensor,
    dmono_exp: torch.Tensor,
) -> torch.Tensor:
    del grad_runs, dmono_coeff, dmono_exp
    return torch.empty_like(quat)


def _runs_setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    quat, mono_coeff, mono_exp, lmax = inputs
    del mono_coeff, mono_exp
    ctx.lmax = int(lmax)
    ctx.save_for_backward(quat)
    ctx.set_materialize_grads(False)


def _runs_backward(ctx: Any, grad_runs: torch.Tensor) -> tuple:
    (quat,) = ctx.saved_tensors
    _, dmono_coeff, _, dmono_exp = wigner_run_tables(ctx.lmax)
    g_quat = torch.ops.deepmd.dpa4_wigner_runs_backward(
        grad_runs.contiguous(),
        quat,
        dmono_coeff.to(quat.device),
        dmono_exp.to(quat.device),
    )
    return g_quat, None, None, None


def _forward_fake(
    x: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    dst_order: torch.Tensor,
    dst_rowptr: torch.Tensor,
    src_order: torch.Tensor,
    src_rowptr: torch.Tensor,
    runs: torch.Tensor,
    kc: torch.Tensor,
    cb: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    gw: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    logit_w: torch.Tensor,
    null_logit: torch.Tensor,
    env: torch.Tensor,
    rad0: torch.Tensor,
    fscale: torch.Tensor,
    head_gate: torch.Tensor,
    rescale: torch.Tensor,
    lmax: int,
    focus_dim: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del src, dst, dst_order, dst_rowptr, src_order, src_rowptr
    del kc, cb, w1, gw, q, k, logit_w, null_logit, env, rad0, fscale
    del rescale, rank
    row = (3 * int(lmax) + 1) * int(focus_dim)
    n_focus = x.shape[2] // int(focus_dim)
    n_head = head_gate.shape[2]
    node = x.new_empty(x.shape[0], (int(lmax) + 1) ** 2, x.shape[2])
    return (
        node,
        x.new_empty(runs.shape[0], n_focus, n_head),
        torch.empty_like(node),
        x.new_empty(w0.shape[0], runs.shape[0], n_focus, row),
    )


def _backward_fake(
    grad_out: torch.Tensor,
    z_all: torch.Tensor,
    x: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    src_order: torch.Tensor,
    src_rowptr: torch.Tensor,
    runs: torch.Tensor,
    kc: torch.Tensor,
    cb: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    gw: torch.Tensor,
    alpha: torch.Tensor,
    head_gate: torch.Tensor,
    rescale: torch.Tensor,
    lmax: int,
    focus_dim: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_out, z_all, src, dst, src_order, src_rowptr
    del cb, w0, w1, gw, head_gate, rescale
    del lmax, focus_dim, rank
    return (
        torch.empty_like(x),
        torch.empty_like(runs),
        torch.empty_like(kc),
        torch.empty_like(alpha),
    )


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple) -> None:
    (
        x,
        src,
        dst,
        dst_order,
        dst_rowptr,
        src_order,
        src_rowptr,
        runs,
        kc,
        cb,
        w0,
        w1,
        gw,
        q,
        k,
        logit_w,
        null_logit,
        env,
        rad0,
        fscale,
        head_gate,
        rescale,
        lmax,
        focus_dim,
        rank,
    ) = inputs
    del dst_order, dst_rowptr, null_logit, rad0
    ctx.save_for_backward(
        output[1],
        output[2],
        output[3],
        x,
        src,
        dst,
        src_order,
        src_rowptr,
        runs,
        kc,
        cb,
        w0,
        w1,
        gw,
        q,
        k,
        logit_w,
        env,
        fscale,
        head_gate,
        rescale,
    )
    ctx.lmax = int(lmax)
    ctx.focus_dim = int(focus_dim)
    ctx.rank = int(rank)
    ctx.set_materialize_grads(False)


def _backward(
    ctx: Any, grad_out: torch.Tensor, grad_alpha: Any, grad_pre: Any, grad_z: Any
) -> tuple:
    del grad_alpha, grad_pre, grad_z
    (
        alpha,
        pre_gate,
        z_all,
        x,
        src,
        dst,
        src_order,
        src_rowptr,
        runs,
        kc,
        cb,
        w0,
        w1,
        gw,
        q,
        k,
        logit_w,
        env,
        fscale,
        head_gate,
        rescale,
    ) = ctx.saved_tensors
    grad_out = grad_out.contiguous()
    g_x, g_runs, g_kc, g_weight = torch.ops.deepmd.dpa4_so2_conv_backward(
        grad_out,
        z_all,
        x,
        src,
        dst,
        src_order,
        src_rowptr,
        runs,
        kc,
        cb,
        w0,
        w1,
        gw,
        alpha,
        head_gate,
        rescale,
        ctx.lmax,
        ctx.focus_dim,
        ctx.rank,
    )

    # === Step 1. Head-gate cotangent, a node-level reduction ===
    n_node, n_focus, n_head = head_gate.shape
    g_head_gate = (
        (grad_out * pre_gate).sum(1).reshape(n_node, n_focus, n_head, -1).sum(-1)
    )

    # === Step 2. Softmax and logit cotangents ===
    # The kernel differentiates through the weight it applied and the saved
    # weight carries the optional competition scale, so the raw softmax weight
    # is recovered before the Jacobian. The null mass keeps the weights from
    # summing to one but leaves the Jacobian form unchanged, because it does
    # not depend on any logit.
    if fscale.numel() > 0:
        fs = fscale.unsqueeze(-1)
        raw_alpha = alpha / fs.clamp_min(1e-30)
        g_alpha = g_weight * fs
        g_fscale = (g_weight * raw_alpha).sum(-1)
    else:
        raw_alpha = alpha
        g_alpha = g_weight
        g_fscale = None
    seg = alpha.new_zeros(n_node, n_focus, n_head)
    seg.index_add_(0, dst, raw_alpha * g_alpha)
    g_logit = raw_alpha * (g_alpha - seg.index_select(0, dst))  # (E, F, H)

    # === Step 3. Query, key, radial-bias and envelope cotangents ===
    head_dim = ctx.focus_dim // n_head
    n_edge = alpha.shape[0]
    inv = float(head_dim) ** -0.5
    q_heads = q.reshape(n_node, n_focus, n_head, head_dim)
    k_heads = k.reshape(n_node, n_focus, n_head, head_dim)
    gl = g_logit.unsqueeze(-1) * inv  # (E, F, H, 1)
    g_q = q.new_zeros(n_node, n_focus, n_head, head_dim)
    g_q.index_add_(0, dst, gl * k_heads.index_select(0, src))
    g_k = q.new_zeros(n_node, n_focus, n_head, head_dim)
    g_k.index_add_(0, src, gl * q_heads.index_select(0, dst))
    g_rad0 = torch.einsum(
        "efh,fih->efi", g_logit, logit_w.reshape(n_focus, ctx.focus_dim, n_head)
    ).reshape(n_edge, -1)
    env_flat = env.reshape(n_edge)
    positive = env_flat > 0
    g_env = torch.where(
        positive,
        g_logit.sum((1, 2)) * 2.0 / env_flat.clamp_min(1e-30),
        torch.zeros_like(env_flat),
    ).reshape(env.shape)

    return (
        g_x,
        None,
        None,
        None,
        None,
        None,
        None,
        g_runs,
        g_kc,
        None,
        None,
        None,
        None,
        g_q.reshape(q.shape),
        g_k.reshape(k.shape),
        None,
        None,
        g_env,
        g_rad0,
        g_fscale,
        g_head_gate,
        None,
        None,
        None,
        None,
    )


def ensure_registered() -> None:
    """Register fake and autograd implementations. Safe to call repeatedly."""
    global _registered
    if _registered or not op_available():
        return
    torch.library.register_fake("deepmd::dpa4_so2_conv")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4_so2_conv_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa4_so2_conv", _backward, setup_context=_setup_context
    )
    torch.library.register_fake("deepmd::dpa4_wigner_runs")(_runs_fake)
    torch.library.register_fake("deepmd::dpa4_wigner_runs_backward")(
        _runs_backward_fake
    )
    torch.library.register_autograd(
        "deepmd::dpa4_wigner_runs",
        _runs_backward,
        setup_context=_runs_setup_context,
    )
    _registered = True


class SO2ConvCuda:
    """Per-convolution entry driving the fused CUDA value path.

    The call contract replaces the reference ``so2_message(...,
    return_local=True)`` followed by the flash aggregation and the output head
    gate: it consumes the node features and returns the gated destination
    aggregate with shape ``(N, D, C_wide)``.
    """

    def __init__(self, conv: Any) -> None:
        self._conv = conv
        mixer = conv.radial_degree_mixer
        self._rank = 0 if mixer is None else int(mixer.rank)
        self._compete = bool(conv.focus_compete and conv.n_focus > 1)
        # Packed-run polynomial tables, fitted once per degree and materialized
        # on the compute device at the first call.
        self._tables_cpu = wigner_run_tables(conv.lmax)
        self._tables: tuple[torch.Tensor, ...] | None = None

    def run_tables(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The packed-run tables on the compute device."""
        if self._tables is None or self._tables[0].device != device:
            self._tables = tuple(t.to(device) for t in self._tables_cpu)
        return self._tables

    def edge_runs(self, edge_cache: Any) -> torch.Tensor:
        """
        Packed block-diagonal Wigner runs of every edge, built once per step.

        Every interaction block of a step shares one edge set and one degree,
        so the runs are cached next to the CSR views; autograd accumulates the
        run cotangents of all consumers before the single contraction back onto
        the quaternions.

        Parameters
        ----------
        edge_cache : Any
            The step's edge feature cache, holding the edge quaternions and the
            per-step tensor store.

        Returns
        -------
        torch.Tensor
            The packed runs with shape (E, NW). Entries ``l ** 2`` to
            ``(l + 1) ** 2`` of a row are the ``m = 0`` Wigner row of degree
            ``l``, which is also the zonal coupling the initial embedding needs.
        """
        ensure_registered()
        store = edge_cache.csr_cache if edge_cache.csr_cache is not None else {}
        key = f"runs:{self._conv.lmax}"
        runs = store.get(key)
        if runs is None:
            quat = edge_cache.edge_quat.contiguous()
            tables = self.run_tables(quat.device)
            runs = torch.ops.deepmd.dpa4_wigner_runs(
                quat, tables[0], tables[2], self._conv.lmax
            )
            store[key] = runs
        return runs

    def radial_features(self, radial_feat: torch.Tensor) -> torch.Tensor:
        """Project the per-edge radial features, as the reference path does."""
        conv = self._conv
        if conv.radial_hidden_proj is not None:
            return conv.radial_hidden_proj(radial_feat)
        return radial_feat

    def _degree_kernel(
        self, rad_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the compact per-edge degree kernel and its channel basis.

        Parameters
        ----------
        rad_feat : torch.Tensor
            Projected radial features with shape (E, lmax+1, C_wide).

        Returns
        -------
        tuple of torch.Tensor
            The compact kernel with shape (E, kc_len) and the channel basis with
            shape (rank, C_wide). Without a mixer the kernel is the radial
            feature itself and the basis is a placeholder.
        """
        mixer = self._conv.radial_degree_mixer
        if mixer is None:
            return rad_feat.reshape(rad_feat.shape[0], -1), rad_feat.new_zeros(1)
        kc = torch.matmul(rad_feat.reshape(rad_feat.shape[0], -1), mixer.weight)
        return kc, mixer.channel_basis.reshape(self._rank, -1)

    def _pack_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack the SO(2) block weights and gate projections per layer.

        Returns ``(w0, w1, gw)`` with shapes ``(n_layers, F, M0, M0)``,
        ``(n_layers, F, M1, M1)`` and ``(n_gated, F, Cf, lmax * Cf)``, all in
        the ``(in, out)`` convention the operator expects.
        """
        conv = self._conv
        m0 = (conv.lmax + 1) * conv.so2_focus_dim
        w0_list, w1_list, gw_list = [], [], []
        for layer, linear in enumerate(conv.so2_linears):
            weight = linear._build_so2_weight().detach().permute(1, 0, 2).contiguous()
            w0_list.append(weight[:, :m0, :m0])
            w1_list.append(weight[:, m0:, m0:])
            non_linear = conv.non_linearities[layer]
            if type(non_linear).__name__ == "GatedActivation":
                gw_list.append(
                    non_linear.gate_linear.weight.detach()
                    .view(
                        conv.so2_focus_dim,
                        conv.n_focus,
                        conv.lmax * conv.so2_focus_dim,
                    )
                    .permute(1, 0, 2)
                )
        return (
            torch.stack(w0_list).contiguous(),
            torch.stack(w1_list).contiguous(),
            torch.stack(gw_list).contiguous(),
        )

    def _focus_scale(
        self,
        x: torch.Tensor,
        edge_cache: Any,
        rad_feat: torch.Tensor,
        kc: torch.Tensor,
        cb: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-focus competition weights with shape (E, F).

        The competition reads the ``l = 0`` scalar row of the mixed local
        feature, which the fused operator computes internally. Reconstructing
        just that row costs one ``m = 0`` rotation, which is cheap next to the
        stack but not free; a dedicated kernel would avoid the gathered node
        feature it materializes.
        """
        conv = self._conv
        lmax, cf = conv.lmax, conv.so2_focus_dim
        n_deg = lmax + 1
        rows = [l * l + l for l in range(n_deg)]
        d_m0 = edge_cache.D_full[:, rows, :]  # (E, lmax+1, D)
        x_local = torch.bmm(d_m0, x.index_select(0, edge_cache.src))  # (E, L+1, C_wide)
        if self._rank == 0:
            scalar = x_local[:, 0, :] * rad_feat[:, 0, :]  # (E, C_wide)
        else:
            slots = [i * n_deg for i in range(n_deg)]
            sel = kc.reshape(kc.shape[0], -1, self._rank)[:, slots, :]  # (E, L+1, rank)
            keff = torch.einsum("eir,rc->eic", sel, cb)  # (E, L+1, C_wide)
            scalar = (keff * x_local).sum(1)  # (E, C_wide)
        gate_src = scalar.reshape(-1, conv.n_focus, cf)  # (E, F, Cf)
        return conv._focus_alpha(gate_src).to(dtype=x.dtype)

    def __call__(
        self,
        x: torch.Tensor,
        edge_cache: Any,
        rad_feat: torch.Tensor,
        q_node: torch.Tensor,
        k_node: torch.Tensor,
        head_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the fused convolution.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : Any
            Precomputed edge cache, providing ``src``, ``dst``, ``edge_quat``,
            ``edge_env`` and the CSR cache.
        rad_feat : torch.Tensor
            Projected radial features with shape (E, lmax+1, C_wide).
        q_node : torch.Tensor
            Attention queries with shape (N, F, Cf).
        k_node : torch.Tensor
            Attention keys with shape (N, F, Cf).
        head_gate : torch.Tensor
            Output-side head gate with shape (N, F, H).

        Returns
        -------
        torch.Tensor
            Gated destination aggregate with shape (N, D, C_wide).
        """
        ensure_registered()
        conv = self._conv
        n_node = x.shape[0]
        kc, cb = self._degree_kernel(rad_feat)
        if self._compete:
            fscale = self._focus_scale(x, edge_cache, rad_feat, kc, cb)  # (E, F)
        else:
            fscale = x.new_empty(0)
        w0, w1, gw = self._pack_weights()
        # Both interaction blocks share the edge set, so the two CSR views are
        # built once per step and kept in the edge cache. The source-major pair
        # only rides through the forward so the autograd context can hand it to
        # the backward.
        store = edge_cache.csr_cache if edge_cache.csr_cache is not None else {}
        if "dst" not in store:
            store["dst"] = edge_csr(edge_cache.dst, n_node)
            store["src"] = edge_csr(edge_cache.src, n_node)
        csr = store["dst"] + store["src"]
        runs = self.edge_runs(edge_cache)
        # The null mass enters the kernel in log space, matching the reference
        # softmax that seeds every segment maximum with it.
        null_logit = torch.log(
            torch.nn.functional.softplus(conv.adamw_attn_z_bias_raw) + float(conv.eps)
        ).reshape(conv.n_focus, conv.n_atten_head)
        logit_w = conv.adamw_attn_logit_w.permute(1, 0, 2).contiguous()  # (F, Cf, H)
        out, _, _, _ = torch.ops.deepmd.dpa4_so2_conv(
            x.contiguous(),
            edge_cache.src,
            edge_cache.dst,
            *csr,
            runs,
            kc.contiguous(),
            cb.contiguous(),
            w0,
            w1,
            gw,
            q_node.reshape(n_node, -1).contiguous(),
            k_node.reshape(n_node, -1).contiguous(),
            logit_w,
            null_logit,
            edge_cache.edge_env.reshape(-1).contiguous(),
            rad_feat[:, 0, :].contiguous(),
            fscale,
            head_gate.contiguous(),
            conv.rotate_inv_rescale_full,
            conv.lmax,
            conv.so2_focus_dim,
            self._rank,
        )
        return out


def _is_supported(conv: Any) -> bool:
    """Return whether ``conv`` matches the fused CUDA configuration."""
    mixer = conv.radial_degree_mixer
    non_linears = conv.non_linearities
    last = conv.mixing_layers - 1
    return (
        conv.mmax == 1
        and 1 <= conv.lmax <= _MAX_LMAX
        and conv.mixing_layers >= 2
        and conv.n_atten_head >= 1
        and conv.so2_focus_dim in _SUPPORTED_FOCUS_DIMS
        # The fused softmax assigns whole 32-lane channel slots to heads and
        # shares the attention layout with the value stream.
        and conv.so2_focus_dim % conv.n_atten_head == 0
        and conv.so2_focus_dim // conv.n_atten_head >= 32
        and conv.attn_n_focus == conv.n_focus
        and conv.attn_focus_dim == conv.so2_focus_dim
        # ``node_wise_grid_product`` couples into the local frame inside the
        # fused span; the message-node and node-Cartesian products act on the
        # aggregate afterwards and are therefore unconstrained here.
        and conv.node_wise_grid_product is None
        and conv.attn_focus_mix is None
        and not conv.use_so2_attn_res
        and not conv.layer_scale
        and not conv.edge_cartesian
        and conv.so2_linears[0].weight_m0.dtype is torch.float32
        # A ``degree`` mixer shares one kernel across channels, a layout the
        # compact per-edge buffer does not express.
        and (mixer is None or mixer.mode == "degree_channel")
        and all(type(norm).__name__ == "Identity" for norm in conv.so2_inter_norms)
        and all(linear.bias0 is None for linear in conv.so2_linears)
        and all(
            linear.in_channels == conv.so2_focus_dim
            and linear.out_channels == conv.so2_focus_dim
            for linear in conv.so2_linears
        )
        and all(
            type(non_linears[layer]).__name__ == "GatedActivation"
            and (
                getattr(non_linears[layer].scalar_act, "activation", None)
                or getattr(non_linears[layer], "activation_function", None)
            )
            == "silu"
            for layer in range(last)
        )
        and type(non_linears[last]).__name__ == "Identity"
    )


# Heaviest per-edge mixing-stack layer the fused convolution is worth taking
# over, in fused multiply-adds, calibrated on an RTX PRO 6000 Blackwell.
# The operator trades device traffic for float32 SIMT arithmetic, so its
# profit falls as that arithmetic grows. Measured as the end-to-end difference
# between taking the convolution over and leaving it on the Triton path:
# +43 % at 27.6 kFMA (``mini``) against -4 % at 112.6 kFMA (``neo``, whose
# second focus stream is why a width threshold misjudges it) and -9 % at
# 225.3 kFMA (``air``). The threshold sits between the measured signs.
_MAX_PROFITABLE_LAYER_FMA = 65536

# Ridge point (fp32 FLOP per DRAM byte) of the calibration part. The traffic
# the takeover saves is fixed by the shape while its cost is fp32 time, so the
# break-even arithmetic scales with the ridge of the executing device: an H20
# at roughly one seventh of this ridge admits no zoo checkpoint, matching the
# measured 0.79x of ``mini`` there.
_RIDGE_REF = 73.2

_ridge_scale: float | None = None


def _device_ridge_scale() -> float:
    """Ridge of the current device over the calibration part's."""
    global _ridge_scale
    if _ridge_scale is None:
        _ridge_scale = float(torch.ops.deepmd.dpa4_fp32_ridge()) / _RIDGE_REF
    return _ridge_scale


def _profitable(conv: Any) -> bool:
    """Whether the fused convolution is expected to beat the path it replaces.

    One mixing layer costs ``M0^2 + M1^2 + Cf * GATE`` multiply-adds per edge
    and focus stream, with ``M0 = (lmax + 1) Cf``, ``M1 = 2 lmax Cf`` and
    ``GATE = lmax * Cf``; every focus stream repeats it. That count sizes the
    float32 arithmetic the operator takes on and is what its profit turns on,
    normalized by how much of that arithmetic the executing device buys per
    byte of the traffic it saves.
    """
    cf = conv.so2_focus_dim
    m0 = (conv.lmax + 1) * cf
    m1 = 2 * conv.lmax * cf
    per_layer = m0 * m0 + m1 * m1 + cf * conv.lmax * cf
    budget = _MAX_PROFITABLE_LAYER_FMA * _device_ridge_scale()
    return per_layer * conv.n_focus <= budget


def make_cuda_so2_conv(conv: Any) -> SO2ConvCuda | None:
    """
    Build the fused CUDA value-path entry for a convolution block.

    Declines a block the operator does not serve, and also one it serves but
    would slow down: the fused convolution is float32 SIMT where the Triton
    composition it replaces is bandwidth bound, so it wins at the narrower
    degrees and loses once the per-edge arithmetic outgrows that advantage.
    Declining leaves the block on the Triton path, which is what
    ``DP_CUDA_INFER=1`` would have done, so raising the level never costs time.

    Parameters
    ----------
    conv : Any
        The ``SO2Convolution`` block to accelerate.

    Returns
    -------
    SO2ConvCuda or None
        The entry callable when the operator is loaded, ``conv`` matches the
        supported configuration and the substitution is expected to pay;
        otherwise ``None``, and the caller keeps the Triton or reference path.
    """
    if not op_available() or not _is_supported(conv) or not _profitable(conv):
        return None
    return SO2ConvCuda(conv)
