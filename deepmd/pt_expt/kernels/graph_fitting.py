# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused energy fitting operator of the graph lower.

The operator ``deepmd::graph_fitting`` evaluates the whole energy fitting
network on the flat node axis -- GEMMs with the bias / activation / residual
epilogues fused into the surrounding elementwise pass -- and returns the
per-atom energy in fp64. The registered backward chains the layer dgrads from
the saved pre-activations, exposing the descriptor gradient that the force /
virial assembly differentiates through. The CUDA kernel is
``source/op/pt/graph_fitting.cu`` and the CPU kernel
``source/op/pt/graph_fitting_cpu.cc``.

The operator is descriptor-agnostic: any graph-lowered energy model whose
fitting is a plain MLP over the flat node axis (see
:func:`fitting_eligible`) dispatches here, regardless of which descriptor
produced the input.

Usage and pitfalls
------------------
* The forward's second output packs the layer pre-activations as one flat
  buffer, chunk per layer, written by the GEMMs themselves; the activations are
  a forward-only transient in a two-slot ping-pong. The buffer is an autograd
  save, never a user-facing value, and is marked non-differentiable.
* The backward re-derives each activation derivative from the saved
  pre-activation and the layer bias, so it takes the biases and the activation
  code as arguments. It infers the node count and descriptor width from the
  saved buffer and first weight, deliberately not retaining the descriptor
  tensor, which allows inference memory planners to reuse descriptor storage
  for its gradient after the fitting forward.
* The head bias is passed as a device tensor, not a Python float: reading a
  value host-side (``.item()``) inside the dispatch path would fail under
  symbolic tracing (``GuardOnDataDependentSymNode``) and force a GPU sync per
  step in eager mode.
* ``Tensor[]`` op inputs (weights / biases) are pytree list nodes: the
  backward must return a matching ``list`` of ``None`` for each, while
  ``int[]`` inputs are single leaves taking a single ``None``.
* The operator represents exactly the networks :func:`fitting_eligible`
  accepts and silently computes something else for any other network, so
  every entry point builds its arguments through
  :func:`fitting_operator_arguments`, which validates before it converts.
"""

import os
from dataclasses import (
    dataclass,
)
from functools import (
    cache,
)
from typing import (
    Any,
)

import torch

from deepmd.pt_expt.kernels.triton.dpa1.activation import (
    ACT_CODES,
)
from deepmd.pt_expt.kernels.utils import (
    operator_available,
)

__all__ = [
    "FittingArguments",
    "energy_and_input_gradient",
    "ensure_registered",
    "fitting_eligible",
    "fitting_operator_arguments",
    "graph_fitting",
    "node_tile",
    "op_available",
]

#: Nodes evaluated per run of a tiled inference pipeline. The tile bounds
#: every node-scale allocation that a run retires, and at this size the extra
#: launches are a fraction of a percent of a saturated step while the GEMM
#: shape stays far above the width at which cuBLAS loses efficiency.
#: ``DP_NODE_TILE`` overrides it; zero evaluates the whole node axis at once.
_DEFAULT_TILE = 131072


def node_tile() -> int:
    """Return the configured node tile of the tiled inference pipeline."""
    value = os.environ.get("DP_NODE_TILE")
    return _DEFAULT_TILE if value is None else int(value)


def op_available() -> bool:
    """Whether the backend device carries every fitting operator used here."""
    return all(
        operator_available(name)
        for name in (
            "graph_fitting",
            "graph_fitting_backward",
            "graph_fitting_energy_gradient",
        )
    )


def fitting_eligible(fit: Any) -> bool:
    """Whether the fused fitting operator can serve this network.

    Requires a single mixed-types energy net with tanh / silu on every hidden
    layer, fp32 weights and biases, no layer timestep, a linear scalar head
    without residual, float4-aligned hidden widths of at most 4096, and no
    frame / atomic parameters, case embedding or type exclusion. Hidden
    identity residuals are supported; width-doubling residuals use the
    reference path.

    A per-layer timestep is a supported configuration of the reference
    network but not of this operator. Nothing prevents it technically -- the
    scale is a length-``dout`` vector, so the backward could form
    ``act'(pre + b) * idt`` from the same saved pre-activation -- but the
    deployed models do not use one, and carrying it would add a load and a
    multiply per element to the forward and to both backward epilogues.

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
    if any(layer.idt is not None for layer in hidden):
        return False
    tensors = [
        tensor
        for layer in layers
        for tensor in (layer.w, layer.b)
        if tensor is not None
    ]
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        return False
    if any(
        layer.resnet and layer.w.shape[1] == 2 * layer.w.shape[0] for layer in hidden
    ):
        return False
    # The elementwise epilogues map one float4 lane of a row to threadIdx.x,
    # so a row must be non-empty and fit the 1024-thread block limit at four
    # values per lane.
    return all(
        int(layer.w.shape[1]) % 4 == 0 and 0 < int(layer.w.shape[1]) <= 4096
        for layer in hidden
    )


@dataclass(frozen=True)
class FittingArguments:
    """Fused operator arguments derived from one fitting network.

    Attributes
    ----------
    weights : list[torch.Tensor]
        Hidden layer weights, each with shape (din, dout).
    biases : list[torch.Tensor]
        Hidden layer biases with shape (dout,), empty where a layer has none.
    residuals : list[int]
        One flag per hidden layer marking an identity residual.
    head_weight : torch.Tensor
        Flattened linear head weight with shape (width,).
    head_bias : torch.Tensor
        Scalar head bias with shape (1,), empty where the head has none.
    activation : int
        Hidden activation code, see ``ACT_CODES``.
    """

    weights: list[torch.Tensor]
    biases: list[torch.Tensor]
    residuals: list[int]
    head_weight: torch.Tensor
    head_bias: torch.Tensor
    activation: int


def fitting_operator_arguments(fit: Any) -> FittingArguments:
    """Convert a fitting network into fused operator arguments.

    The operator represents exactly the networks :func:`fitting_eligible`
    accepts. For anything else it does not fail; it evaluates a different
    network, dropping whatever it cannot represent. Validation therefore
    belongs at every boundary that converts a module into operator arguments,
    which is what this function is.

    Parameters
    ----------
    fit : EnergyFittingNet
        The pt_expt fitting module.

    Returns
    -------
    FittingArguments
        Contiguous tensors and codes ready for ``deepmd::graph_fitting``.

    Raises
    ------
    ValueError
        If the fused operator cannot reproduce this network.
    """
    if not fitting_eligible(fit):
        raise ValueError(
            "the fused fitting operator cannot reproduce this network; test "
            "fitting_eligible() first and fall back to the reference path"
        )
    *hidden, head = fit.nets[0].layers
    empty = hidden[0].w.new_empty(0)
    return FittingArguments(
        weights=[layer.w.contiguous() for layer in hidden],
        biases=[
            layer.b.contiguous() if layer.b is not None else empty for layer in hidden
        ],
        residuals=[1 if layer.resnet else 0 for layer in hidden],
        head_weight=head.w.reshape(-1).contiguous(),
        head_bias=(
            head.b.reshape(-1).to(torch.float32).contiguous()
            if head.b is not None
            else empty
        ),
        activation=ACT_CODES[str(hidden[0].activation_function).lower()],
    )


# ======================================================================
# Fake (meta) implementations and the autograd bridge
# ======================================================================
def _forward_fake(
    x: torch.Tensor,
    atype: torch.Tensor,
    ws: list[torch.Tensor],
    bs: list[torch.Tensor],
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


def _energy_gradient_fake(
    x: torch.Tensor,
    atype: torch.Tensor,
    ws: list[torch.Tensor],
    bs: list[torch.Tensor],
    resnets: list[int],
    w_head: torch.Tensor,
    b_head: torch.Tensor,
    bias_atom_e: torch.Tensor,
    act: int,
    seed: torch.Tensor,
    tile: int,
) -> torch.Tensor:
    return x.new_empty(x.shape[0], 1, dtype=torch.float64)


def _backward_fake(
    d_e: torch.Tensor,
    saved: torch.Tensor,
    ws: list[torch.Tensor],
    bs: list[torch.Tensor],
    resnets: list[int],
    w_head: torch.Tensor,
    act: int,
) -> torch.Tensor:
    total_width = sum(int(w.shape[1]) for w in ws)
    n_node = saved.shape[0] // total_width
    return saved.new_empty(n_node, ws[0].shape[0])


def _setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
    x, atype, ws, bs, resnets, w_head, b_head, bias_atom_e, act = inputs
    _e, saved = output
    # The saved buffer is an internal autograd artifact; the backward consumes
    # it as data and produces no cotangent for it.
    ctx.mark_non_differentiable(saved)
    ctx.save_for_backward(saved, w_head, *ws, *bs)
    ctx.n_layers = len(ws)
    ctx.resnets = resnets
    ctx.act = act
    ctx.set_materialize_grads(False)


def _backward(ctx: Any, d_e: torch.Tensor, d_saved: Any) -> tuple:
    saved, w_head, *rest = ctx.saved_tensors
    ws, bs = rest[: ctx.n_layers], rest[ctx.n_layers :]
    d_x = torch.ops.deepmd.graph_fitting_backward(
        d_e, saved, list(ws), list(bs), ctx.resnets, w_head, ctx.act
    )
    none_list = [None] * ctx.n_layers
    return (d_x, None, none_list, none_list, None, None, None, None, None)


# ======================================================================
# CPU reference implementations
# ======================================================================
# ======================================================================
# Registration and the public wrapper
# ======================================================================
@cache
def _register_ops() -> None:
    """Register the meta and autograd implementations for the ops once.

    Both devices implement the network in C++, so only the shapes and the
    autograd rule are described here.
    """
    torch.library.register_fake("deepmd::graph_fitting")(_forward_fake)
    torch.library.register_fake("deepmd::graph_fitting_backward")(_backward_fake)
    torch.library.register_fake("deepmd::graph_fitting_energy_gradient")(
        _energy_gradient_fake
    )
    torch.library.register_autograd(
        "deepmd::graph_fitting", _backward, setup_context=_setup_context
    )


def ensure_registered() -> None:
    """Register meta and autograd implementations when the ops are available."""
    if op_available():
        _register_ops()


def energy_and_input_gradient(
    fit: Any,
    descriptor: torch.Tensor,
    atype: torch.Tensor,
    ownership: torch.Tensor,
    atom_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node energy and descriptor cotangent of an inference step.

    Inference seeds the fitting backward with the ownership mask, which is
    known before the forward runs, so both directions are evaluated together
    over node tiles and the pre-activations never reach node scale. The
    cotangent is returned in the descriptor's own storage, which inference no
    longer needs once the forward has consumed it.

    Parameters
    ----------
    fit : EnergyFittingNet
        The pt_expt fitting module.
    descriptor : torch.Tensor
        Flat descriptor with shape (N, nd), fp32 and contiguous.
    atype : torch.Tensor
        Flat node atom types with shape (N,), int64.
    ownership : torch.Tensor
        Mask selecting energy-contributing nodes with shape (N,).
    atom_bias : torch.Tensor
        Combined atomic energy bias with shape (ntypes,).

    Returns
    -------
    atom_energy : torch.Tensor
        Per-node energy with shape (N, 1), fp64, before the ownership mask.
    descriptor_gradient : torch.Tensor
        Cotangent of the descriptor with shape (N, nd), fp32. This is the
        descriptor tensor itself, overwritten in place.
    """
    ensure_registered()
    network = fitting_operator_arguments(fit)
    energy = torch.ops.deepmd.graph_fitting_energy_gradient(
        descriptor,
        atype,
        network.weights,
        network.biases,
        network.residuals,
        network.head_weight,
        network.head_bias,
        atom_bias.to(torch.float64).contiguous(),
        network.activation,
        ownership.to(torch.float64).reshape(-1).contiguous(),
        node_tile(),
    )
    return energy, descriptor


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
    arguments = fitting_operator_arguments(fit)
    e, _saved = torch.ops.deepmd.graph_fitting(
        descriptor.to(torch.float32).contiguous(),
        atype.contiguous(),
        arguments.weights,
        arguments.biases,
        arguments.residuals,
        arguments.head_weight,
        arguments.head_bias,
        fit.bias_atom_e.to(torch.float64).reshape(-1, 1)[:, 0].contiguous(),
        arguments.activation,
    )
    return {fit.var_name: e}
