# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN202
"""Bindings and model entry for the fused SO(2) value path, training form.

The CUDA operator ``deepmd::sezm_so2_value_fwd`` (see
``source/op/pt/dpa4/so2_conv_train.cu``) evaluates the value stream of one
``SO2Convolution`` up to the attention aggregation in a single kernel: the
gather into the edge frame over the structural block-diagonal non-zeros of
the Wigner-D matrix, the edge-conditioned radial degree mixing, the
cross-focus competition weight from the ``l = 0`` scalars, every gated
mixing layer, and the final identity layer with its edge-major store. The
rotated input and all inter-layer activations live in shared memory for the
lifetime of a block; the only global surfaces are the operator outputs and
the backward anchors (the stacked pre-activations ``z_all``, the final gated
activation ``u_final``, and the competition weight ``alpha``).

The backward is one CUDA operator (``deepmd::sezm_so2_value_bwd``): the
rotated input is recomputed by the fused rotate-mix forward, the mixing
traversal runs with its weight contractions, the competition head is
differentiated in closed form from the stored ``alpha``, and the rotation
gradients flow through the fused rotation backward with the contention-free
CSR segment reduction. The weight contractions run only when a parameter
gradient is requested: the force pass (``autograd.grad(E, coord)``)
differentiates the coordinate chain alone, and its parameter-gradient GEMMs
would be discarded. The second order a force loss requires is likewise one
CUDA operator (``deepmd::sezm_so2_value_bwd2``), analytic for the
force-loss regime where the cotangent enters only through the node-feature
gradient. The training value path therefore never leaves the CUDA library;
it composes no Triton operator.

The attention span downstream (segmented softmax, flash aggregation, head
gate) runs as the Triton operator composition inside the traced graph,
where the compiler fuses it with its neighbours; a fused CUDA form of that
span was built, measured slower at equal memory, and removed
(``dpa4_cuda.md`` section 12).

Supported configuration
-----------------------
The Triton value-path constraints (``mmax == 1``, degree 1 to 6, gated stack
with an identity final layer, supported focus widths, radial mixer absent or
``degree_channel`` with rank at most 4), at most 256 wide channels, at most
4 focus streams, and an identity competition norm (``focus_norm=False``).
Unsupported blocks keep the narrower fused paths.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from torch import (
    Tensor,
)

if TYPE_CHECKING:
    from deepmd.pt.model.descriptor.sezm_nn.so2 import (
        SO2Convolution,
    )

__all__ = [
    "SO2ValueTrainCuda",
    "ensure_registered",
    "make_cuda_so2_value",
    "op_available",
]

_registered = False


def op_available() -> bool:
    """Return whether the fused value-path forward is loaded."""
    ops = getattr(torch.ops, "deepmd", None)
    return ops is not None and hasattr(ops, "sezm_so2_value_fwd")


def _alpha_dtype(working: torch.dtype) -> torch.dtype:
    """
    Precision the competition weight and its gradient are carried in.

    The weight is the backward's anchor for the whole competition head, which
    reconstructs the softmax from it (``p = (alpha - ls/F) / (1 - ls)``) and
    divides the traversal's weight gradient by it. Under bfloat16 that chain
    would lose about three decimal digits, which no later promotion recovers,
    so the anchor is kept in accumulator precision; being ``(E, F)`` scalars
    it costs nothing next to the ``(E, F, ROW)`` surfaces. Mirrors
    ``dpa4_sezm::alpha_dtype`` on the operator side.

    Parameters
    ----------
    working : torch.dtype
        Precision of the operator's surfaces.

    Returns
    -------
    torch.dtype
        ``torch.float64`` for a float64 pass, ``torch.float32`` otherwise.
    """
    return torch.float64 if working is torch.float64 else torch.float32


def _fwd_fake(
    x,
    src,
    wigner,
    kc,
    cb,
    w_fc,
    fc_bias,
    w0_all,
    w1_all,
    gw_all,
    lmax,
    n_focus,
    rank,
    apply_alpha,
    softmax_tau,
    label_smoothing,
):
    n_edge = src.shape[0]
    cf = x.shape[2] // n_focus
    row = (3 * lmax + 1) * cf
    return (
        x.new_empty((n_edge, n_focus, row)),
        x.new_empty((gw_all.shape[0], n_focus, n_edge, row)),
        x.new_empty((n_focus, n_edge, row)),
        x.new_empty((n_edge, n_focus), dtype=_alpha_dtype(x.dtype)),
    )


def _bwd_fake(
    grad_x_local,
    x,
    src,
    src_order,
    src_rowptr,
    wigner,
    kc,
    cb,
    w_fc,
    fc_bias,
    w0_all,
    w1_all,
    gw_all,
    x_local,
    z_all,
    u_final,
    alpha,
    h_z,
    h_uf,
    h_alpha,
    lmax,
    n_focus,
    rank,
    apply_alpha,
    softmax_tau,
    label_smoothing,
    keep_state,
    with_weights,
):
    # Every output of the operator is a fresh contiguous allocation;
    # ``new_empty`` (never ``empty_like``) keeps the fake from inheriting
    # the strides of a non-contiguous graph input.
    n_gated, n_focus_z, n_edge, row = z_all.shape
    lg = lmax * (row // (3 * lmax + 1))
    if keep_state:
        kept = (
            u_final.new_empty((n_focus_z, n_edge, row)),
            z_all.new_empty((n_gated, n_focus_z, n_edge, row)),
            z_all.new_empty((n_gated, n_focus_z, n_edge, row)),
            z_all.new_empty((n_gated, n_focus_z, n_edge, lg)),
        )
    else:
        kept = (
            x.new_empty(0),
            x.new_empty(0),
            x.new_empty(0),
            x.new_empty(0),
        )
    return (
        x.new_empty(x.shape),
        wigner.new_empty(wigner.shape),
        kc.new_empty(kc.shape),
        cb.new_empty(cb.shape) if rank > 0 else x.new_empty(0),
        (
            w_fc.new_empty(w_fc.shape)
            if (with_weights and w_fc is not None)
            else x.new_empty(0)
        ),
        (
            fc_bias.new_empty(fc_bias.shape)
            if (with_weights and fc_bias is not None)
            else x.new_empty(0)
        ),
        (w0_all.new_empty(w0_all.shape) if with_weights else x.new_empty(0)),
        (w1_all.new_empty(w1_all.shape) if with_weights else x.new_empty(0)),
        (gw_all.new_empty(gw_all.shape) if with_weights else x.new_empty(0)),
        *kept,
    )


def _bwd2_fake(
    h_gx,
    h_gwig,
    h_gkc,
    grad_x_local,
    x,
    src,
    src_order,
    src_rowptr,
    wigner,
    kc,
    cb,
    w_fc,
    fc_bias,
    w0_all,
    w1_all,
    gw_all,
    x_local,
    z_all,
    u_final,
    alpha,
    kept_grad_u0,
    kept_upstream,
    kept_grad_z,
    kept_grad_logit,
    lmax,
    n_focus,
    rank,
    apply_alpha,
    softmax_tau,
    label_smoothing,
):
    return (
        grad_x_local.new_empty(grad_x_local.shape),
        x.new_empty(x.shape),
        wigner.new_empty(wigner.shape),
        kc.new_empty(kc.shape),
        cb.new_empty(cb.shape) if rank > 0 else x.new_empty(0),
        w_fc.new_empty(w_fc.shape) if w_fc is not None else x.new_empty(0),
        (fc_bias.new_empty(fc_bias.shape) if fc_bias is not None else x.new_empty(0)),
        w0_all.new_empty(w0_all.shape),
        w1_all.new_empty(w1_all.shape),
        gw_all.new_empty(gw_all.shape),
        x_local.new_empty(x_local.shape) if apply_alpha else x.new_empty(0),
        (
            alpha.new_empty(alpha.shape)
            if apply_alpha
            else alpha.new_empty((0, alpha.shape[1]))
        ),
        z_all.new_empty(z_all.shape),
        # The force-regime first order never reads ``u_final`` (its weight
        # contractions are skipped and the alpha gradient contracts against
        # the stored output), so its curvature slot stays a placeholder.
        x.new_empty(0),
    )


def ensure_registered() -> None:
    """Register the fake implementations the compile pipeline requires."""
    global _registered
    if _registered or not op_available():
        return
    torch.library.register_fake("deepmd::sezm_so2_value_fwd")(_fwd_fake)
    torch.library.register_fake("deepmd::sezm_so2_value_bwd")(_bwd_fake)
    torch.library.register_fake("deepmd::sezm_so2_value_bwd2")(_bwd2_fake)
    _registered = True


def _value_train_impl(
    x: Tensor,
    src: Tensor,
    src_order: Tensor,
    src_rowptr: Tensor,
    wigner: Tensor,
    kc: Tensor,
    cb: Tensor,
    w_fc: Tensor | None,
    fc_bias: Tensor | None,
    w0_all: Tensor,
    w1_all: Tensor,
    gw_all: Tensor,
    lmax: int,
    n_focus: int,
    rank: int,
    apply_alpha: bool,
    softmax_tau: float,
    label_smoothing: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the fused value-path forward.

    The source CSR view rides through untouched so the autograd context can
    hand it to the backward's segment reduction.

    Returns ``(x_local, z_all, u_final, alpha)`` with ``x_local`` edge-major
    ``(E, F, ROW)`` and the remaining three the backward anchors.
    """
    del src_order, src_rowptr
    return torch.ops.deepmd.sezm_so2_value_fwd(
        x.contiguous(),
        src,
        wigner,
        kc.contiguous(),
        cb.contiguous(),
        w_fc.to(x.dtype) if w_fc is not None else None,
        fc_bias.to(x.dtype) if fc_bias is not None else None,
        w0_all,
        w1_all,
        gw_all,
        int(lmax),
        int(n_focus),
        int(rank),
        bool(apply_alpha),
        float(softmax_tau),
        float(label_smoothing),
    )


_value_train_op = torch.library.custom_op(
    "deepmd_cuda::so2_value_train",
    _value_train_impl,
    mutates_args=(),
)


@_value_train_op.register_fake
def _(
    x,
    src,
    src_order,
    src_rowptr,
    wigner,
    kc,
    cb,
    w_fc,
    fc_bias,
    w0_all,
    w1_all,
    gw_all,
    lmax,
    n_focus,
    rank,
    apply_alpha,
    softmax_tau,
    label_smoothing,
):
    return _fwd_fake(
        x,
        src,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        lmax,
        n_focus,
        rank,
        apply_alpha,
        softmax_tau,
        label_smoothing,
    )


def _value_train_bwd_impl(
    grad_x_local: Tensor,
    x: Tensor,
    src: Tensor,
    src_order: Tensor,
    src_rowptr: Tensor,
    wigner: Tensor,
    kc: Tensor,
    cb: Tensor,
    w_fc: Tensor | None,
    fc_bias: Tensor | None,
    w0_all: Tensor,
    w1_all: Tensor,
    gw_all: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    alpha: Tensor,
    h_z: Tensor | None,
    h_uf: Tensor | None,
    h_alpha: Tensor | None,
    lmax: int,
    n_focus: int,
    rank: int,
    apply_alpha: bool,
    softmax_tau: float,
    label_smoothing: float,
    keep_state: bool,
    with_weights: bool,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    """First order of the fused value path, one CUDA operator call.

    Under ``keep_state`` (the force regime) the mixing traversal's per-layer
    surfaces and the total input gradient ride out as trailing outputs; the
    second order consumes them and replays nothing. The weight contractions
    run only under ``with_weights``.
    """
    return torch.ops.deepmd.sezm_so2_value_bwd(
        grad_x_local,
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
        h_z,
        h_uf,
        h_alpha,
        int(lmax),
        int(n_focus),
        int(rank),
        bool(apply_alpha),
        float(softmax_tau),
        float(label_smoothing),
        bool(keep_state),
        bool(with_weights),
    )


_value_train_bwd_op = torch.library.custom_op(
    "deepmd_cuda::so2_value_train_bwd",
    _value_train_bwd_impl,
    mutates_args=(),
)


@_value_train_bwd_op.register_fake
def _(
    grad_x_local,
    x,
    src,
    src_order,
    src_rowptr,
    wigner,
    kc,
    cb,
    w_fc,
    fc_bias,
    w0_all,
    w1_all,
    gw_all,
    x_local,
    z_all,
    u_final,
    alpha,
    h_z,
    h_uf,
    h_alpha,
    lmax,
    n_focus,
    rank,
    apply_alpha,
    softmax_tau,
    label_smoothing,
    keep_state,
    with_weights,
):
    return _bwd_fake(
        grad_x_local,
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
        h_z,
        h_uf,
        h_alpha,
        lmax,
        n_focus,
        rank,
        apply_alpha,
        softmax_tau,
        label_smoothing,
        keep_state,
        with_weights,
    )


def _value_train_bwd_setup_context(ctx, inputs, output):
    (
        grad_x_local,
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
        h_z,
        h_uf,
        h_alpha,
        lmax,
        n_focus,
        rank,
        apply_alpha,
        softmax_tau,
        label_smoothing,
        keep_state,
        with_weights,
    ) = inputs
    kept = output[9:13] if keep_state else (None, None, None, None)
    ctx.save_for_backward(
        grad_x_local,
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
        *kept,
    )
    ctx.set_materialize_grads(False)
    ctx.had_upstream = h_z is not None or h_uf is not None or h_alpha is not None
    ctx.keep_state = keep_state
    ctx.lmax = lmax
    ctx.n_focus = n_focus
    ctx.rank = rank
    ctx.apply_alpha = apply_alpha
    ctx.softmax_tau = softmax_tau
    ctx.label_smoothing = label_smoothing


def _value_train_bwd_backward(ctx, h_gx, *h_rest: Tensor | None):
    """Analytic second order, force-loss regime.

    The force graph sends cotangents through the node-feature, Wigner and
    degree-kernel gradients (whose producers precede this operator on the
    coordinate graph); the parameter gradients feed the optimizer and carry
    none. The whole linearization runs as one CUDA operator call.
    """
    h_gwig, h_gkc = h_rest[0], h_rest[1]
    if h_gx is None and all(h is None for h in h_rest):
        return (None,) * 28
    if any(h is not None for h in h_rest[2:]) or ctx.had_upstream:
        raise NotImplementedError(
            "sezm_so2_value_bwd second order supports the force-loss regime "
            "only: cotangents on parameter gradients are not implemented"
        )
    (
        grad_x_local,
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
        kept_grad_u0,
        kept_upstream,
        kept_grad_z,
        kept_grad_logit,
    ) = ctx.saved_tensors
    apply_alpha = bool(ctx.apply_alpha)
    rank = int(ctx.rank)
    (
        grad_grad_x_local,
        gx2,
        gwig2,
        gkc2,
        gcb2,
        gwfc2,
        gbias2,
        gw02,
        gw12,
        ggw2,
        gxl2,
        galpha2,
        gz2,
        _guf2,
    ) = torch.ops.deepmd.sezm_so2_value_bwd2(
        h_gx.contiguous() if h_gx is not None else torch.zeros_like(x),
        h_gwig,
        h_gkc,
        grad_x_local,
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
        kept_grad_u0,
        kept_upstream,
        kept_grad_z,
        kept_grad_logit,
        int(ctx.lmax),
        int(ctx.n_focus),
        rank,
        apply_alpha,
        float(ctx.softmax_tau),
        float(ctx.label_smoothing),
    )
    # inputs: grad_x_local, x, src, src_order, src_rowptr, wigner, kc, cb,
    # w_fc, fc_bias, w0_all, w1_all, gw_all, x_local, z_all, u_final, alpha,
    # h_z, h_uf, h_alpha, lmax, n_focus, rank, apply_alpha, softmax_tau,
    # label_smoothing, keep_state, with_weights.
    return (
        grad_grad_x_local,
        gx2,
        None,
        None,
        None,
        gwig2,
        gkc2,
        gcb2 if rank > 0 else None,
        gwfc2 if apply_alpha else None,
        gbias2 if (apply_alpha and fc_bias is not None) else None,
        gw02,
        gw12,
        ggw2,
        gxl2 if apply_alpha else None,
        gz2,
        # The first order never reads ``u_final`` in this regime; ``guf2``
        # is a zero-sized placeholder and its cotangent stays ``None``.
        None,
        galpha2 if apply_alpha else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_value_train_bwd_op.register_autograd(
    _value_train_bwd_backward, setup_context=_value_train_bwd_setup_context
)


def _value_train_setup_context(ctx, inputs, output):
    (
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        lmax,
        n_focus,
        rank,
        apply_alpha,
        softmax_tau,
        label_smoothing,
    ) = inputs
    x_local, z_all, u_final, alpha = output
    ctx.save_for_backward(
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
    )
    # The anchors have no consumer outside this context; their cotangents
    # must stay ``None`` rather than materialize as zero surfaces.
    ctx.set_materialize_grads(False)
    ctx.lmax = lmax
    ctx.n_focus = n_focus
    ctx.rank = rank
    ctx.apply_alpha = apply_alpha
    ctx.softmax_tau = softmax_tau
    ctx.label_smoothing = label_smoothing


def _value_train_backward(ctx, grad_x_local, h_z, h_uf, h_alpha):
    """First order of the fused value path, one CUDA operator call."""
    (
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
    ) = ctx.saved_tensors
    apply_alpha = bool(ctx.apply_alpha)
    rank = int(ctx.rank)
    grad_x_local = (
        grad_x_local.contiguous()
        if grad_x_local is not None
        else torch.zeros_like(x_local)
    )
    # Under an ambient grad mode a second differentiation is coming (the
    # force regime), so the traversal retains its linearization surfaces
    # and the second order replays nothing.
    keep_state = torch.is_grad_enabled()
    # The force pass differentiates the coordinate chain alone; the
    # parameter-gradient contractions run only when some parameter slot
    # actually requests a gradient.
    needs = ctx.needs_input_grad
    with_weights = any(needs[i] for i in (7, 8, 9, 10, 11))
    (
        grad_x,
        grad_wigner,
        grad_kc,
        grad_cb,
        grad_w_fc,
        grad_bias,
        grad_w0,
        grad_w1,
        grad_gw,
    ) = _value_train_bwd_op(
        grad_x_local,
        x,
        src,
        src_order,
        src_rowptr,
        wigner,
        kc,
        cb,
        w_fc,
        fc_bias,
        w0_all,
        w1_all,
        gw_all,
        x_local,
        z_all,
        u_final,
        alpha,
        h_z,
        h_uf,
        h_alpha,
        int(ctx.lmax),
        int(ctx.n_focus),
        rank,
        apply_alpha,
        float(ctx.softmax_tau),
        float(ctx.label_smoothing),
        keep_state,
        with_weights,
    )[:9]
    # inputs: x, src, src_order, src_rowptr, wigner, kc, cb, w_fc, fc_bias,
    # w0_all, w1_all, gw_all, lmax, n_focus, rank, apply_alpha, softmax_tau,
    # label_smoothing.
    return (
        grad_x,
        None,
        None,
        None,
        grad_wigner,
        grad_kc,
        grad_cb if rank > 0 else None,
        grad_w_fc if (with_weights and apply_alpha) else None,
        (grad_bias if (with_weights and apply_alpha and fc_bias is not None) else None),
        grad_w0 if with_weights else None,
        grad_w1 if with_weights else None,
        grad_gw if with_weights else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_value_train_op.register_autograd(
    _value_train_backward, setup_context=_value_train_setup_context
)

# Under autocast the node features arrive in bfloat16 while the Wigner
# buffer and the parameters are float32, a mix the kernel cannot consume.
# Align every floating-point input to the autocast dtype exactly as the
# built-in matmuls do; the casts are recorded by autograd, so parameters
# still accumulate float32 gradients. Inert outside an autocast region.
_value_train_op.register_autocast("cuda", torch.bfloat16)
_value_train_bwd_op.register_autocast("cuda", torch.bfloat16)


class SO2ValueTrainCuda:
    """Per-convolution entry running the value path through the fused kernel.

    The call contract mirrors ``_TritonSO2ValuePath``: it returns the
    post-focus-compete local features ``(E, F, D_m, Cf)`` and the projected
    radial features whose ``l = 0`` slice feeds the attention aggregation.

    The stacked weights are assembled from the live parameters on every call
    and must not be cached across calls: the first call may run inside a
    ``make_fx`` fake-tensor trace, where a cache would capture fake weights,
    and eager weights may change when a checkpoint is loaded after
    construction.
    """

    def __init__(self, conv: SO2Convolution) -> None:
        self._conv = conv

    def _pack_weights(self, *, differentiable: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Stack the SO(2) block weights and gate projections per layer.

        Mirrors the Triton value path: ``(w0_all, w1_all, gw_all)`` with
        shapes ``(n_layers, F, M0, M0)``, ``(n_layers, F, M1, M1)`` and
        ``(n_gated, F, Cf, lmax * Cf)``, all in the ``(in, out)`` convention.
        """
        conv = self._conv
        m0 = (conv.lmax + 1) * conv.so2_focus_dim
        w0_list, w1_list, gw_list = [], [], []
        for layer, linear in enumerate(conv.so2_linears):
            weight = linear._build_so2_weight()
            if not differentiable:
                weight = weight.detach()
            weight = weight.permute(1, 0, 2).contiguous()
            w0_list.append(weight[:, :m0, :m0])
            w1_list.append(weight[:, m0:, m0:])
            non_linear = conv.non_linearities[layer]
            if type(non_linear).__name__ == "GatedActivation":
                gate = non_linear.gate_linear.weight
                if not differentiable:
                    gate = gate.detach()
                gw_list.append(
                    gate.view(
                        conv.so2_focus_dim,
                        conv.n_focus,
                        conv.lmax * conv.so2_focus_dim,
                    ).permute(1, 0, 2)
                )
        return (
            torch.stack(w0_list).contiguous(),
            torch.stack(w1_list).contiguous(),
            torch.stack(gw_list).contiguous(),
        )

    def __call__(
        self,
        x: Tensor,
        edge_cache: Any,
        radial_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the SO(2) local features and radial features.

        Parameters
        ----------
        x : Tensor
            Node features with shape (N, D, C_wide).
        edge_cache : EdgeCache
            Precomputed edge cache (provides ``src`` and the Wigner
            ``D_full``).
        radial_feat : Tensor
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        x_local : Tensor
            Post-focus-compete local features with shape (E, F, D_m, Cf).
        rad_feat : Tensor
            Projected radial features with shape (E, lmax+1, C_wide).
        """
        conv = self._conv
        src = edge_cache.src
        ensure_registered()
        w0_all, w1_all, gw_all = self._pack_weights(differentiable=conv.training)

        rad_feat = (
            conv.radial_hidden_proj(radial_feat)
            if conv.radial_hidden_proj is not None
            else radial_feat
        )
        mixer = conv.radial_degree_mixer
        if mixer is None:
            kc = rad_feat
            cb = rad_feat.new_zeros(1)
            rank = 0
        else:
            kc = torch.matmul(rad_feat.reshape(rad_feat.shape[0], -1), mixer.weight)
            cb = mixer.channel_basis.reshape(-1)
            rank = mixer.rank

        # The source CSR view is built once per step and kept on the edge
        # cache (the caller normally pre-populates it); a cache-less caller
        # pays for its own.
        store = getattr(edge_cache, "csr_cache", None)
        csr = None if store is None else store.get("src")
        if csr is None:
            src_order = torch.argsort(src, dim=0, stable=True)
            counts = src.new_zeros(x.shape[0]).scatter_add(0, src, torch.ones_like(src))
            src_rowptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
        else:
            src_order, src_rowptr = csr
        apply_alpha = bool(conv.focus_compete and conv.n_focus > 1)
        x_local, _z_all, _u_final, _alpha = _value_train_op(
            x,
            src,
            src_order,
            src_rowptr,
            edge_cache.D_full,
            kc,
            cb,
            conv.adamw_focus_compete_w if apply_alpha else None,
            conv.focus_compete_bias if apply_alpha else None,
            w0_all,
            w1_all,
            gw_all,
            conv.lmax,
            conv.n_focus,
            rank,
            apply_alpha,
            float(conv.focus_softmax_tau),
            float(conv.focus_label_smoothing),
        )
        n_edge = src.shape[0]
        reduced_dim = 3 * conv.lmax + 1
        return (
            x_local.view(n_edge, conv.n_focus, reduced_dim, conv.so2_focus_dim),
            rad_feat,
        )


def make_cuda_so2_value(conv: SO2Convolution) -> SO2ValueTrainCuda | None:
    """Build the fused CUDA value-path entry for a convolution block.

    Returns ``None`` unless the CUDA operator is loaded and ``conv`` matches
    the supported configuration; the caller then keeps the narrower fused
    paths. The Triton value-path constraints are reused as the base
    admission, with the kernel-specific bounds on top.
    """
    if not op_available():
        return None
    from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
        _is_supported,
    )

    if not _is_supported(conv):
        return None
    if conv.n_focus * conv.so2_focus_dim > 256 or conv.n_focus > 4:
        return None
    if conv.focus_compete and conv.n_focus > 1:
        # The identity competition norm is spelled ``nn.Identity`` on the pt
        # backend and an unbound (``None``) hook on the dpmodel/pt_expt one.
        norm = conv.focus_compete_norm
        if norm is not None and type(norm).__name__ != "Identity":
            return None
    ensure_registered()
    return SO2ValueTrainCuda(conv)
