# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused energy fitting operator of the graph lower.

The CUDA operator ``deepmd::graph_fitting`` (see
``source/op/pt/graph_fitting.cu``) evaluates the whole energy fitting
network on the flat node axis -- cuBLAS GEMMs with the bias / activation /
timestep / residual epilogues fused into single elementwise kernels -- and
returns the per-atom energy in fp64. The registered backward chains the layer
dgrads from the saved activation derivatives, exposing the descriptor
gradient that the force / virial assembly differentiates through.

The operator is descriptor-agnostic: any graph-lowered energy model whose
fitting is a plain MLP over the flat node axis (see
:func:`fitting_eligible`) dispatches here, regardless of which descriptor
produced the input.

Usage and pitfalls
------------------
* The forward's second output packs the layer activation derivatives as one
  flat buffer of ``adot chunks`` (the activations themselves are a forward-only
  transient); it is an autograd save, never a user-facing value, and receives
  no gradient (``set_materialize_grads(False)``).
* The backward infers the node count and descriptor width from the saved
  derivative buffer and first weight. It deliberately does not retain the
  descriptor tensor, allowing inference memory planners to reuse descriptor
  storage for its gradient after the fitting forward.
* The head bias is passed as a device tensor, not a Python float: reading a
  value host-side (``.item()``) inside the dispatch path would fail under
  symbolic tracing (``GuardOnDataDependentSymNode``) and force a GPU sync per
  step in eager mode.
* ``Tensor[]`` op inputs (weights / biases / timesteps) are pytree list
  nodes: the backward must return a matching ``list`` of ``None`` for each,
  while ``int[]`` inputs are single leaves taking a single ``None``.
* The eligibility gate (:func:`fitting_eligible`) requires float4-aligned
  hidden widths, one tanh / silu activation on every hidden layer, a linear
  scalar head and no frame / atomic parameters.
"""

from typing import (
    Any,
)

import torch

from deepmd.kernels.triton.dpa1.activation import (
    ACT_CODES,
)

__all__ = [
    "ensure_registered",
    "fitting_eligible",
    "graph_fitting",
    "op_available",
]

_registered = False


def op_available() -> bool:
    """Whether the C++ ``deepmd::graph_fitting`` op is loaded."""
    op = getattr(torch.ops.deepmd, "graph_fitting", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def fitting_eligible(fit: Any) -> bool:
    """Whether the fused fitting operator can serve this network.

    Requires a single mixed-types energy net with tanh / silu on every hidden
    layer, fp32 weights and hidden-layer parameters, a linear scalar head
    without timestep or residual, float4-aligned hidden widths, and no frame /
    atomic parameters, case embedding or type exclusion. Hidden identity
    residuals are supported; width-doubling residuals use the reference path.

    Parameters
    ----------
    fit : EnergyFittingNet
        The pt_expt fitting module.

    Returns
    -------
    bool
        ``True`` when the fused operator reproduces the reference forward.
    """
    if fit.numb_fparam or fit.numb_aparam or fit.dim_case_embd:
        return False
    if getattr(fit, "exclude_types", None):
        return False
    if not fit.mixed_types or len(fit.nets._networks) != 1:
        return False
    layers = fit.nets[0].layers
    if len(layers) < 2:
        return False
    *hidden, head = layers
    acts = {str(layer.activation_function).lower() for layer in hidden}
    if len(acts) != 1 or acts.pop() not in ACT_CODES:
        return False
    if str(head.activation_function).lower() not in ("none", "linear"):
        return False
    if head.w.shape[1] != 1 or head.idt is not None or head.resnet:
        return False
    tensors = [
        tensor
        for layer in layers
        for tensor in (layer.w, layer.b, layer.idt)
        if tensor is not None
    ]
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        return False
    if any(
        layer.resnet and layer.w.shape[1] == 2 * layer.w.shape[0] for layer in hidden
    ):
        return False
    return all(int(layer.w.shape[1]) % 4 == 0 for layer in hidden)


# ======================================================================
# Fake (meta) implementations and the autograd bridge
# ======================================================================
def _forward_fake(
    x: torch.Tensor,
    atype: torch.Tensor,
    ws: list[torch.Tensor],
    bs: list[torch.Tensor],
    idts: list[torch.Tensor],
    resnets: list[int],
    w_head: torch.Tensor,
    b_head: torch.Tensor,
    bias_atom_e: torch.Tensor,
    act: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_node = x.shape[0]
    total_width = sum(int(w.shape[1]) for w in ws)
    return (
        x.new_empty(n_node, 1, dtype=torch.float64),
        x.new_empty(n_node * total_width, dtype=torch.float32),
    )


def _backward_fake(
    d_e: torch.Tensor,
    saved: torch.Tensor,
    ws: list[torch.Tensor],
    resnets: list[int],
    w_head: torch.Tensor,
) -> torch.Tensor:
    total_width = sum(int(w.shape[1]) for w in ws)
    n_node = saved.shape[0] // total_width
    return saved.new_empty(n_node, ws[0].shape[0])


def _setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
    x, atype, ws, bs, idts, resnets, w_head, b_head, bias_atom_e, act = inputs
    _e, saved = output
    ctx.save_for_backward(saved, w_head, *ws)
    ctx.n_layers = len(ws)
    ctx.resnets = resnets
    ctx.set_materialize_grads(False)


def _backward(ctx: Any, d_e: torch.Tensor, d_saved: Any) -> tuple:
    saved, w_head, *ws = ctx.saved_tensors
    d_x = torch.ops.deepmd.graph_fitting_backward(
        d_e, saved, list(ws), ctx.resnets, w_head
    )
    none_list = [None] * ctx.n_layers
    return (d_x, None, none_list, none_list, none_list, None, None, None, None, None)


# ======================================================================
# CPU reference implementations
# ======================================================================
def _cpu_forward(
    x: torch.Tensor,
    atype: torch.Tensor,
    ws: list[torch.Tensor],
    bs: list[torch.Tensor],
    idts: list[torch.Tensor],
    resnets: list[int],
    w_head: torch.Tensor,
    b_head: torch.Tensor,
    bias_atom_e: torch.Tensor,
    act: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    adots = []
    cur = x.to(torch.float32)
    for w, b, idt, res in zip(ws, bs, idts, resnets):
        pre = cur @ w
        if b.numel():
            pre = pre + b
        a = torch.tanh(pre) if act == 0 else torch.nn.functional.silu(pre)
        if act == 0:
            adot = 1.0 - a * a
        else:
            s = torch.sigmoid(pre)
            adot = s * (1.0 + pre * (1.0 - s))
        if idt.numel():
            a = a * idt
            adot = adot * idt
        adots.append(adot)
        cur = a + cur if (res and w.shape[0] == w.shape[1]) else a
    e = (cur @ w_head[:, None]).to(torch.float64)
    if b_head.numel():
        e = e + b_head.to(torch.float64)
    e = e + bias_atom_e[atype][:, None]
    # Chunk layout mirrors the CUDA op: adot chunks only, each a contiguous
    # row-major (N, w_l) block; the backward needs only these derivatives.
    saved = torch.cat([t.reshape(-1) for t in adots])
    return e, saved


def _cpu_backward(
    d_e: torch.Tensor,
    saved: torch.Tensor,
    ws: list[torch.Tensor],
    resnets: list[int],
    w_head: torch.Tensor,
) -> torch.Tensor:
    total_width = sum(int(w.shape[1]) for w in ws)
    n_node = saved.shape[0] // total_width
    offset = [0]
    for w in ws:
        offset.append(offset[-1] + int(w.shape[1]))
    adots = [
        saved[offset[l] * n_node : offset[l + 1] * n_node].reshape(
            n_node, int(ws[l].shape[1])
        )
        for l in range(len(ws))
    ]
    dh = d_e.to(torch.float32) * w_head
    for l in range(len(ws) - 1, -1, -1):
        dpre = dh * adots[l]
        dx = dpre @ ws[l].t()
        if resnets[l] and ws[l].shape[0] == ws[l].shape[1]:
            dx = dx + dh
        dh = dx
    return dh


# ======================================================================
# Registration and the public wrapper
# ======================================================================
_cpu_library: torch.library.Library | None = None


def ensure_registered() -> None:
    """Register the fake / backward / CPU implementations for the ops.

    Idempotent; a no-op when the C++ operator library is not loaded.
    """
    global _registered, _cpu_library
    if _registered or not op_available():
        return
    torch.library.register_fake("deepmd::graph_fitting")(_forward_fake)
    torch.library.register_fake("deepmd::graph_fitting_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::graph_fitting", _backward, setup_context=_setup_context
    )
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("graph_fitting", _cpu_forward, "CPU")
    _cpu_library.impl("graph_fitting_backward", _cpu_backward, "CPU")
    _registered = True


def graph_fitting(
    fit: Any,
    descriptor: torch.Tensor,
    atype: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fused energy fitting on the flat node axis.

    Drop-in for ``GeneralFitting.call_graph`` on the configuration accepted
    by :func:`fitting_eligible`; the descriptor gradient flows through the
    registered backward so the force / virial assembly differentiates
    end-to-end.

    Parameters
    ----------
    fit : EnergyFittingNet
        The pt_expt fitting module.
    descriptor : torch.Tensor
        Flat descriptor with shape (N, nd).
    atype : torch.Tensor
        Flat node atom types with shape (N,), int64.

    Returns
    -------
    dict[str, torch.Tensor]
        ``{fit.var_name: energy}`` with energy shape (N, 1), fp64.
    """
    ensure_registered()
    *hidden, head = fit.nets[0].layers
    empty = hidden[0].w.new_empty(0)
    e, _saved = torch.ops.deepmd.graph_fitting(
        descriptor.to(torch.float32).contiguous(),
        atype.contiguous(),
        [layer.w.contiguous() for layer in hidden],
        [layer.b.contiguous() if layer.b is not None else empty for layer in hidden],
        [
            layer.idt.contiguous() if layer.idt is not None else empty
            for layer in hidden
        ],
        [1 if layer.resnet else 0 for layer in hidden],
        head.w.reshape(-1).contiguous(),
        (
            head.b.reshape(-1).to(torch.float32).contiguous()
            if head.b is not None
            else empty
        ),
        fit.bias_atom_e.to(torch.float64).reshape(-1, 1)[:, 0].contiguous(),
        ACT_CODES[str(hidden[0].activation_function).lower()],
    )
    return {fit.var_name: e}
