# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training loop for the pt_expt backend.

Uses ``DeepmdDataSystem`` (numpy-based batch provider) instead of the
pt backend's ``DpLoaderSet`` + ``DataLoader``.  NumPy batches are
converted to torch tensors at the boundary.
"""

import functools
import logging
import os
import time
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import torch
import torch.distributed as dist

from deepmd.dpmodel.train import (
    DEFAULT_TASK_KEY,
    AbstractTrainer,
    RankContext,
    TrainerConfig,
    TrainingTask,
    TrainingTaskCollection,
    TrainStepResult,
    change_model_out_bias,
    change_model_out_bias_by_task,
)
from deepmd.dpmodel.utils.batch import (
    normalize_batch,
    split_batch,
)
from deepmd.dpmodel.utils.learning_rate import (
    make_learning_rate_schedule,
)
from deepmd.pt.train.utils import (
    resolve_best_checkpoint_dir,
)
from deepmd.pt.train.validation import (
    FullValidator,
    resolve_full_validation_start_step,
)
from deepmd.pt.utils.compile_compat import next_safe_prime as _next_safe_prime
from deepmd.pt.utils.compile_compat import rebuild_graph_module as _rebuild_graph_module
from deepmd.pt.utils.compile_compat import (
    strip_saved_tensor_detach as _strip_saved_tensor_detach,
)
from deepmd.pt.utils.compile_compat import trace_pad_dim as _trace_pad_dim
from deepmd.pt_expt.loss import (
    DOSLoss,
    EnergyLoss,
    EnergySpinLoss,
    PropertyLoss,
    TensorLoss,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.pt_expt.utils.stat import (
    make_stat_input,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    warn_configuration_mismatch_during_finetune,
)
from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)

# Buffer names in atomic_model that are per-task (energy/output statistics).
# These live one level above the fitting net and are not reached by
# fitting-net share_params.  They are always promoted to FX placeholders
# because model_change_out_bias may replace them out-of-place after
# compilation, so the compiled forward must read them fresh each call.
_ATOMIC_MODEL_TASK_BUFFER_NAMES: tuple[str, ...] = ("out_bias", "out_std")

# Prefix used in task_buf_order keys to distinguish atomic_model buffers
# from fitting-net buffers.
_AM_PREFIX = "am/"


def _detect_task_buffers(
    model: torch.nn.Module,
    group_models: list["torch.nn.Module"],
) -> dict[str, torch.Tensor]:
    """Collect per-task buffers to promote to FX placeholders.

    Fitting-net buffers are auto-detected by identity diff across
    *group_models* (all tasks that share this model's structure key after
    ``share_params``).  Any buffer that is a *different* Python object in at
    least one other group member is task-specific and gets promoted.

    Atomic-model buffers listed in ``_ATOMIC_MODEL_TASK_BUFFER_NAMES`` are
    always promoted because ``model_change_out_bias`` may replace them
    out-of-place after compilation.
    """
    result: dict[str, torch.Tensor] = {}

    # Auto-detect fitting-net task buffers by identity diff across the group.
    try:
        fitting = model.get_fitting_net()
        for name, val in fitting._buffers.items():
            if val is None or not torch.is_tensor(val):
                continue
            for other in group_models:
                if other is model:
                    continue
                try:
                    other_val = other.get_fitting_net()._buffers.get(name)
                    if other_val is not val:
                        result[name] = val.detach().clone()
                        break
                except AttributeError:
                    pass
    except AttributeError:
        pass

    # Atomic-model task buffers (always promote).
    try:
        am = model.atomic_model
        for name in _ATOMIC_MODEL_TASK_BUFFER_NAMES:
            val = am._buffers.get(name)
            if val is not None and torch.is_tensor(val):
                result[_AM_PREFIX + name] = val.detach().clone()
    except AttributeError:
        pass

    return result


def _get_model_structure_key(model: torch.nn.Module) -> tuple[int, ...]:
    """Return a key that is identical iff two tasks can safely share a compiled graph.

    The key captures both the descriptor identity and the fitting-net
    structure so that tasks sharing a fitting net but using *different*
    descriptors (which bake distinct descriptor constants into the traced
    graph) are never assigned the same compiled graph.

    Descriptor identity uses the id of the first shared parameter tensor.
    ``share_params`` makes descriptor *parameters* the same Python objects
    across tasks while the descriptor modules remain distinct.  Two
    descriptors sharing params therefore collapse to the same key here.
    Partial sharing (shared_level=1, type-embedding only) is detected in
    ``_compile_model`` and raises an explicit error rather than silently
    producing a wrong compiled graph.

    After ``share_params``, the fitting net's child sub-modules are the same
    Python objects across tasks, so ``id(first_child)`` is equal for all
    shared tasks and unique across unrelated models.
    """
    descriptor_id: int = 0
    try:
        desc = model.get_descriptor()
        for _, p in desc.named_parameters():
            descriptor_id = id(p)
            break
        else:
            descriptor_id = id(desc)
    except AttributeError:
        pass

    try:
        fitting = model.get_fitting_net()
        for _, child in fitting.named_children():
            return (descriptor_id, id(child))
    except AttributeError:
        pass
    return (descriptor_id, id(model))


# ---------------------------------------------------------------------------
# Helper: loss factory (reused from pt)
# ---------------------------------------------------------------------------


def get_loss(
    loss_params: dict[str, Any],
    start_lr: float,
    _ntypes: int,
    _model: Any,
) -> EnergyLoss:
    loss_type = loss_params.get("type", "ener")
    if loss_type == "ener":
        loss_params["starter_learning_rate"] = start_lr
        return EnergyLoss(**loss_params)
    elif loss_type == "dos":
        loss_params["starter_learning_rate"] = start_lr
        loss_params["numb_dos"] = _model.model_output_def()["dos"].output_size
        return DOSLoss(**loss_params)
    elif loss_type == "ener_spin":
        loss_params["starter_learning_rate"] = start_lr
        return EnergySpinLoss(**loss_params)
    elif loss_type == "tensor":
        model_output_type = _model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        tensor_name = model_output_type[0]
        loss_params["tensor_size"] = _model.model_output_def()[tensor_name].output_size
        loss_params["label_name"] = tensor_name
        if tensor_name == "polarizability":
            tensor_name = "polar"
        loss_params["tensor_name"] = tensor_name
        return TensorLoss(**loss_params)
    elif loss_type == "property":
        task_dim = _model.get_task_dim()
        var_name = _model.get_var_name()
        intensive = _model.get_intensive()
        loss_params["task_dim"] = task_dim
        loss_params["var_name"] = var_name
        loss_params["intensive"] = intensive
        return PropertyLoss(**loss_params)
    else:
        raise ValueError(f"Unsupported loss type for pt_expt: {loss_type}")


def get_additional_data_requirement(_model: Any) -> list[DataRequirementItem]:
    additional_data_requirement: list[DataRequirementItem] = []
    if _model.get_dim_fparam() > 0:
        has_default_fparam = _model.has_default_fparam()
        fparam_default = (
            np.asarray(_model.get_default_fparam()) if has_default_fparam else 0.0
        )
        additional_data_requirement.append(
            DataRequirementItem(
                "fparam",
                _model.get_dim_fparam(),
                atomic=False,
                must=not has_default_fparam,
                default=fparam_default,
            )
        )
    if _model.get_dim_aparam() > 0:
        additional_data_requirement.append(
            DataRequirementItem(
                "aparam", _model.get_dim_aparam(), atomic=True, must=True
            )
        )
    if _model.has_chg_spin_ebd():
        has_default_cs = _model.has_default_chg_spin()
        if has_default_cs:
            default_cs = _model.get_default_chg_spin()
            if hasattr(default_cs, "cpu"):
                default_cs = default_cs.cpu().numpy()
            else:
                default_cs = np.asarray(default_cs)
        else:
            default_cs = 0.0
        additional_data_requirement.append(
            DataRequirementItem(
                "charge_spin",
                ndof=2,
                atomic=False,
                must=not has_default_cs,
                default=default_cs,
            )
        )
    return additional_data_requirement


def _as_task_map(
    value: Any,
    *,
    multi_task: bool,
    model_keys: list[str],
) -> dict[str, Any]:
    """Return a task-keyed mapping, wrapping single-task values as Default."""
    if multi_task:
        return {model_key: value[model_key] for model_key in model_keys}
    return {DEFAULT_TASK_KEY: value}


def _replace_latest_checkpoint_link(latest: Path, ckpt_path: Path) -> None:
    """Point latest to ckpt_path using a target relative to latest's directory."""
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.symlink_to(os.path.relpath(ckpt_path, latest.parent))


# ---------------------------------------------------------------------------
# torch.compile helpers
# ---------------------------------------------------------------------------


def _forbidden_dims_from_model(
    model: torch.nn.Module,
    task_buf_vals: tuple[torch.Tensor, ...],
) -> set[int]:
    """Prime-collision set for trace-dim selection.

    Collects every ``> 1`` dim of the model's parameters/buffers (so
    ``_next_safe_prime`` never aliases an internal dim like ``g2_dim`` /
    ``axis_neuron`` / ``attn_head`` without a hardcoded list), plus
    ``dim_fparam``/``dim_aparam`` and the task-buffer dims.  Shared by the dense
    :func:`_trace_and_compile` and the graph :func:`_trace_and_compile_graph`;
    each caller adds its path-specific dims (nall/nloc/nsel for dense,
    charge_spin for both) on top of this base set.
    """
    forbidden: set[int] = {
        int(_d)
        for _src in (model.parameters(), model.buffers())
        for _p in _src
        for _d in _p.shape
        if _d > 1
    }
    for _getter in (model.get_dim_fparam, model.get_dim_aparam):
        try:
            _dim = _getter()
            if _dim > 1:
                forbidden.add(int(_dim))
        except Exception:
            pass  # best-effort: dim unavailable -> nothing to forbid
    for _tbv in task_buf_vals:
        for _d in _tbv.shape:
            if _d > 1:
                forbidden.add(int(_d))
    return forbidden


def _trace_and_compile(
    model: torch.nn.Module,
    ext_coord: torch.Tensor,
    ext_atype: torch.Tensor,
    nlist: torch.Tensor,
    mapping: torch.Tensor,
    fparam: torch.Tensor | None,
    aparam: torch.Tensor | None,
    compile_opts: dict[str, Any] | None = None,
    charge_spin: torch.Tensor | None = None,
    task_buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, tuple[str, ...]]:
    """Symbolic-trace ``forward_lower`` and compile with inductor + dynamic=True.

    Parameters
    ----------
    model : torch.nn.Module
        The (uncompiled) model.
    ext_coord, ext_atype, nlist, mapping, fparam, aparam
        Sample tensors used to seed the symbolic tracer.
    compile_opts : dict or None
        User-supplied inductor options.  These are merged on top of the
        built-in defaults (user values take precedence).
    task_buffers : dict or None
        Per-task buffers (e.g. ``bias_atom_e``, ``case_embd``, ``out_bias``,
        ``out_std``) detected by ``_detect_task_buffers``.  These are promoted
        to explicit FX ``placeholder`` nodes so the compiled graph is reusable
        across tasks that share the same structure key.

    Returns
    -------
    compiled : torch.nn.Module
        The compiled ``forward_lower`` callable.
    task_buf_order : tuple[str, ...]
        Ordered names of the promoted buffers (empty when none).
    """
    from torch.fx.experimental.proxy_tensor import (
        make_fx,
    )

    was_training = model.training
    # Trace in train mode so that create_graph=True is captured inside
    # task_deriv_one.  Without this, the autograd.grad that computes
    # forces is traced with create_graph=False (eval mode), producing
    # force tensors that are detached from model parameters — force loss
    # backprop cannot reach the weights and force RMSE never decreases.
    model.train()

    task_buf_order: tuple[str, ...] = tuple(task_buffers.keys()) if task_buffers else ()
    task_buf_vals_trace: tuple[torch.Tensor, ...] = (
        tuple(task_buffers[k] for k in task_buf_order) if task_buffers else ()
    )

    # Resolve fitting net and atomic_model once for buffer patching inside fn.
    _fitting: torch.nn.Module | None = None
    _atomic_model: torch.nn.Module | None = None
    if task_buf_order:
        try:
            _fitting = model.get_fitting_net()
        except AttributeError:
            pass  # no fitting net → no fitting-net buffers to patch
        try:
            _atomic_model = model.atomic_model
        except AttributeError:
            pass  # no atomic_model → no atomic-model buffers to patch

    def fn(
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
        *task_buf_vals: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        extended_coord = extended_coord.detach().requires_grad_(True)
        # Temporarily patch task-specific buffers with the proxy tensors so
        # make_fx records them as FX placeholders rather than baked-in constants.
        # Keys prefixed with _AM_PREFIX are atomic_model buffers; the rest are
        # fitting-net buffers.
        originals: dict[str, torch.Tensor | None] = {}
        if task_buf_order:
            for name, val in zip(task_buf_order, task_buf_vals, strict=True):
                if name.startswith(_AM_PREFIX):
                    actual = name[len(_AM_PREFIX) :]
                    if _atomic_model is not None:
                        originals[name] = _atomic_model._buffers.get(actual)
                        _atomic_model._buffers[actual] = val
                else:
                    if _fitting is not None:
                        originals[name] = _fitting._buffers.get(name)
                        _fitting._buffers[name] = val
        try:
            return model.forward_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
            )
        finally:
            for name, orig in originals.items():
                if name.startswith(_AM_PREFIX):
                    actual = name[len(_AM_PREFIX) :]
                    if _atomic_model is not None:
                        _atomic_model._buffers[actual] = orig
                else:
                    if _fitting is not None:
                        _fitting._buffers[name] = orig

    # Pad nf to a safe prime; keep real nloc and nall from the data.
    #
    # make_fx (tracing_mode="symbolic") unifies dimension symbols that share
    # the same concrete value at trace time (duck-shape merging).  We take
    # one frame ([:1]) to normalise nf, then pad it to a prime so PyTorch
    # does not specialise it as the constant 1.  nloc and nall come from
    # real data, so they are already too
    # large to alias with any architecture dim and need no adjustment.
    #
    # The prime for nf is chosen by enumerating every dimension that appears
    # in the model's parameters and buffers (see _forbidden_dims_from_model),
    # then calling _next_safe_prime to find the first prime that doesn't collide
    # with any of them -- catching internal dims like g2_dim/axis_neuron/
    # attn_head without a hardcoded list.  Add the dense-path dims on top.
    _forbidden = _forbidden_dims_from_model(model, task_buf_vals_trace)
    # Also add the real nloc and nall so trace_nf never aliases them.
    _forbidden.add(int(ext_coord.shape[1]))  # nall
    _forbidden.add(int(ext_atype.shape[1]))  # nall (same tensor, defensive)
    _forbidden.add(int(nlist.shape[1]))  # nloc
    # nsel stays at its real value; add it to forbidden for the same reason.
    _nsel = int(nlist.shape[2])
    if _nsel > 1:
        _forbidden.add(_nsel)
    if charge_spin is not None:
        _dim_cs = int(charge_spin.shape[1])
        if _dim_cs > 1:
            _forbidden.add(_dim_cs)

    trace_nf = _next_safe_prime(5, _forbidden)

    # Pad nf only; nloc and nall retain their real values (no clamping needed).
    ext_coord = _trace_pad_dim(ext_coord[:1], 0, trace_nf)
    ext_atype = _trace_pad_dim(ext_atype[:1], 0, trace_nf)
    nlist = _trace_pad_dim(nlist[:1], 0, trace_nf)
    mapping = _trace_pad_dim(mapping[:1], 0, trace_nf)
    if fparam is not None:
        fparam = _trace_pad_dim(fparam[:1], 0, trace_nf)
    if aparam is not None:
        aparam = _trace_pad_dim(aparam[:1], 0, trace_nf)
    if charge_spin is not None:
        charge_spin = _trace_pad_dim(charge_spin[:1], 0, trace_nf)

    # Decompose silu_backward into primitive ops (sigmoid + mul + ...)
    # so that inductor can compile the graph without requiring a
    # higher-order derivative that PyTorch does not register for
    # the fused silu backward kernel.
    from torch._decomp import (
        get_decompositions,
    )

    decomp_table = get_decompositions([torch.ops.aten.silu_backward.default])

    traced_lower = make_fx(
        fn,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
        decomposition_table=decomp_table,
    )(
        ext_coord,
        ext_atype,
        nlist,
        mapping,
        fparam,
        aparam,
        charge_spin,
        *task_buf_vals_trace,
    )

    return (
        _finalize_compiled_lower(traced_lower, model, was_training, compile_opts),
        task_buf_order,
    )


def _finalize_compiled_lower(
    traced_lower: "torch.fx.GraphModule",
    model: torch.nn.Module,
    was_training: bool,
    compile_opts: dict[str, Any] | None,
    extra_options: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Shared post-``make_fx`` tail: strip detach, rebuild, inductor-compile.

    Used by both the dense :func:`_trace_and_compile` and the graph
    :func:`_trace_and_compile_graph` so the second-order-gradient handling
    (detach removal) and inductor options stay identical on both paths.
    """
    # make_fx inserts aten.detach.default for saved tensors used in the
    # decomposed autograd.grad backward ops.  These detach nodes break
    # second-order gradient flow (d(force)/d(params) for force training).
    # The training trace is fed already-detached, grad-enabled inputs, so
    # every detach is removed unconditionally to restore the gradient path.
    _strip_saved_tensor_detach(traced_lower, remove_all=True)
    # Rebuild into a fresh graph to eliminate stale C-level node pointers
    # left by erase_node(), which can cause segfaults during dynamo re-trace.
    traced_lower = _rebuild_graph_module(traced_lower)

    if not was_training:
        model.eval()

    # Inductor defaults tuned for second-order-gradient training graphs.
    # User-supplied compile_opts override these on a per-key basis.
    inductor_options: dict[str, Any] = {
        "max_autotune": False,
        "shape_padding": True,
        "epilogue_fusion": False,
        "triton.cudagraphs": False,
        "max_fusion_size": 8,
        # NOTE: On GPU with PyTorch <=2.11, consider adding
        # "triton.mix_order_reduction": False to work around
        # pytorch/pytorch#174379, #178080, #179494 under
        # data-dependent symbolic shapes.
    }
    if extra_options:
        inductor_options.update(extra_options)
    if compile_opts:
        inductor_options.update(compile_opts)

    return torch.compile(
        traced_lower,
        backend="inductor",
        dynamic=True,
        options=inductor_options,
    )


def _model_uses_graph_lower(model: torch.nn.Module) -> bool:
    """Whether ``model``'s eager default-flip routes through the GRAPH lower.

    Mirrors the predicate in
    :meth:`~deepmd.pt_expt.model.make_model.make_model.<locals>.CM._resolve_graph_method`
    for ``neighbor_graph_method is None`` (the training default): a model is
    graph-eligible iff it is ``mixed_types`` AND its single descriptor reports
    ``uses_graph_lower() == True`` (dpa1/se_atten with concat type embedding;
    attention layers included).

    When True the compiled lower must be the GRAPH ``forward_common_lower_graph``
    so the compiled path matches eager training (which already default-flips to
    the carry-all graph forward); when False the dense ``forward_lower`` is
    compiled (se_e2_a / dpa2 / dpa3 / linear / zbl).

    ASSUMPTION: training uses the default ``neighbor_graph_method`` (None). If a
    user-facing ``"legacy"`` opt-out is ever plumbed into the trainer, this gate
    must also honor it (else eager would run dense while the compiled path runs
    the graph lower, re-introducing the eager!=compiled divergence this fixes).
    """
    if not hasattr(model, "mixed_types"):
        return False
    try:
        if not model.mixed_types():
            return False
    except (AttributeError, NotImplementedError):
        return False
    # Linear / ZBL atomic models have no single ``descriptor`` -> dense.
    descriptor = getattr(getattr(model, "atomic_model", None), "descriptor", None)
    uses_graph = getattr(descriptor, "uses_graph_lower", None)
    if uses_graph is None:
        return False
    try:
        return bool(uses_graph())
    except (AttributeError, NotImplementedError):
        return False


def _trace_and_compile_graph(
    model: torch.nn.Module,
    fparam: torch.Tensor | None,
    aparam: torch.Tensor | None,
    charge_spin: torch.Tensor | None,
    compile_opts: dict[str, Any] | None = None,
    task_buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, tuple[str, ...]]:
    """Symbolic-trace ``forward_common_lower_graph`` and inductor-compile it.

    The GRAPH analogue of :func:`_trace_and_compile`.  Builds a small synthetic
    NeighborGraph with prime-controlled ``nf`` / ``N`` / ``E`` axes (so make_fx's
    duck-shape unification keeps the three dynamic dims as distinct symbols),
    traces ``model.forward_common_lower_graph`` with ``edge_vec`` as the autograd
    leaf, and translates the internal fitting keys to the public energy-model
    keys (``atom_energy`` / ``energy`` / ``force`` / ``virial``).  The compiled
    callable accepts the positional graph tensors plus the promoted task buffers
    and returns those public keys on the FLAT node axis (``N == sum(n_node)``);
    the caller (:meth:`_CompiledModel.forward`) unravels them to ``(nf, nloc, *)``.

    Parameters
    ----------
    model
        The (uncompiled) graph-eligible energy model.
    fparam, aparam, charge_spin
        Representative optional inputs (or ``None``) so the traced branch
        matches what :meth:`_CompiledModel.forward` passes at run time.
    compile_opts
        User-supplied inductor options (merged over the built-in defaults).
    task_buffers
        Per-task buffers promoted to FX placeholders (see
        :func:`_detect_task_buffers`).
    """
    import math

    from torch._decomp import (
        get_decompositions,
    )
    from torch.fx.experimental.proxy_tensor import (
        make_fx,
    )

    from deepmd.pt_expt.model.ener_model import (
        _translate_energy_keys,
    )

    was_training = model.training
    # Trace in train mode so create_graph=True is captured inside the graph
    # force backward (forward_common_lower_graph passes create_graph=self.training).
    model.train()

    task_buf_order: tuple[str, ...] = tuple(task_buffers.keys()) if task_buffers else ()
    task_buf_vals_trace: tuple[torch.Tensor, ...] = (
        tuple(task_buffers[k] for k in task_buf_order) if task_buffers else ()
    )

    _fitting: torch.nn.Module | None = None
    _atomic_model: torch.nn.Module | None = None
    if task_buf_order:
        try:
            _fitting = model.get_fitting_net()
        except AttributeError:
            pass  # optional accessor; a model without a fitting net keeps None
        try:
            _atomic_model = model.atomic_model
        except AttributeError:
            pass  # optional attribute; a model without an atomic model keeps None

    do_grad_r = model.do_grad_r("energy")
    do_grad_c = model.do_grad_c("energy")

    # ------------------------------------------------------------------
    # Build the trace-time NeighborGraph with prime-distinct nf / N / E.
    #
    # make_fx (tracing_mode="symbolic") unifies dimension symbols that share a
    # concrete value (duck-shape merging).  The three dynamic axes of the graph
    # lower must stay distinct symbols, otherwise the per-frame segment_sum
    # (N -> nf) and the per-edge scatter (E -> N) bake in a false equality:
    #   * nf  = n_node.shape[0]      (per-frame reductions)
    #   * N   = atype.shape[0]       (flat node axis = sum(n_node))
    #   * E   = edge_vec.shape[0]    (edge axis)
    # They are chosen as collision-free primes vs every parameter/buffer dim
    # (see _forbidden_dims_from_model) plus charge_spin.
    # ------------------------------------------------------------------
    _forbidden = _forbidden_dims_from_model(model, task_buf_vals_trace)
    if charge_spin is not None and charge_spin.shape[-1] > 1:
        _forbidden.add(int(charge_spin.shape[-1]))

    trace_nf = _next_safe_prime(5, _forbidden)
    # nloc such that N = trace_nf * nloc is collision-free (and != trace_nf).
    nloc_trace = 7
    while (trace_nf * nloc_trace) in (_forbidden | {trace_nf}):
        nloc_trace += 1
    trace_N = trace_nf * nloc_trace
    # Static edge capacity, prime-padded to stay distinct from nf and N.
    nnei = sum(model.get_sel())
    e_max_base = max(math.ceil(1.25 * nloc_trace * nnei), 7)
    e_max = _next_safe_prime(e_max_base, _forbidden | {trace_nf, trace_N})

    # Shared with the .pt2 export trace (serialization.py) so the two graph
    # traces can never desync on the input schema.  Training uses the run-time
    # float precision and device; optional tensors match the actual call.
    from deepmd.pt_expt.utils.serialization import (
        build_synthetic_graph_inputs,
        check_graph_trace_torch_version,
    )

    check_graph_trace_torch_version(model)
    sample = build_synthetic_graph_inputs(
        model,
        e_max=e_max,
        nframes=trace_nf,
        nloc=nloc_trace,
        dtype=GLOBAL_PT_FLOAT_PRECISION,
        device=DEVICE,
        want_fparam=fparam is not None,
        want_aparam=aparam is not None,
        want_charge_spin=charge_spin is not None,
    )
    (
        s_atype,
        s_n_node,
        s_edge_index,
        s_edge_vec,
        s_edge_mask,
        s_fparam,
        s_aparam,
        s_charge_spin,
    ) = sample

    def fn(
        atype: torch.Tensor,
        n_node: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
        *task_buf_vals: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Patch task-specific buffers with the proxy tensors so make_fx records
        # them as FX placeholders (mirrors the dense ``_trace_and_compile``).
        originals: dict[str, torch.Tensor | None] = {}
        if task_buf_order:
            for name, val in zip(task_buf_order, task_buf_vals, strict=True):
                if name.startswith(_AM_PREFIX):
                    actual = name[len(_AM_PREFIX) :]
                    if _atomic_model is not None:
                        originals[name] = _atomic_model._buffers.get(actual)
                        _atomic_model._buffers[actual] = val
                else:
                    if _fitting is not None:
                        originals[name] = _fitting._buffers.get(name)
                        _fitting._buffers[name] = val
        try:
            # forward_common_lower_graph makes edge_vec the autograd leaf
            # internally, so no outer detach/requires_grad_ here.
            model_ret = model.forward_common_lower_graph(
                atype,
                n_node,
                edge_index,
                edge_vec,
                edge_mask,
                do_atomic_virial=False,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
            )
            return _translate_energy_keys(
                model_ret,
                do_grad_r=do_grad_r,
                do_grad_c=do_grad_c,
                do_atomic_virial=False,
                local=True,
            )
        finally:
            for name, orig in originals.items():
                if name.startswith(_AM_PREFIX):
                    actual = name[len(_AM_PREFIX) :]
                    if _atomic_model is not None:
                        _atomic_model._buffers[actual] = orig
                else:
                    if _fitting is not None:
                        _fitting._buffers[name] = orig

    decomp_table = get_decompositions([torch.ops.aten.silu_backward.default])

    traced_lower = make_fx(
        fn,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
        decomposition_table=decomp_table,
    )(
        s_atype,
        s_n_node,
        s_edge_index,
        s_edge_vec,
        s_edge_mask,
        s_fparam,
        s_aparam,
        s_charge_spin,
        *task_buf_vals_trace,
    )

    # The per-frame virial reduction scatters E edges into the (nf, 3, 3) virial
    # via an atomic_add; inductor's CPU vectorizer asserts on that scatter's
    # scalar index (``index.is_vec``).  Disable CPU SIMD for the graph lower so
    # the scatter is emitted scalar — numerically this only removes a
    # reduction-order source, keeping eager==compiled within fp64 tolerance.
    return (
        _finalize_compiled_lower(
            traced_lower,
            model,
            was_training,
            compile_opts,
            extra_options={"cpp.simdlen": 0},
        ),
        task_buf_order,
    )


class _CompiledModel(torch.nn.Module):
    """Coord extension (eager) -> compiled forward_lower (dynamic shapes).

    Compilation is lazy: ``_trace_and_compile`` is called on the first real
    ``forward()`` invocation using that batch's tensors, so no extra
    ``get_data()`` call is needed during ``__init__``.  Tasks that share the
    same model structure reuse the compiled graph via ``compiled_by_structure``.
    """

    def __init__(
        self,
        original_model: torch.nn.Module,
        structure_key: tuple[int, ...],
        task_buf_order: tuple[str, ...] = (),
        task_buffers: dict[str, torch.Tensor] | None = None,
        compile_opts: dict[str, Any] | None = None,
        compiled_by_structure: dict | None = None,
    ) -> None:
        super().__init__()
        self.original_model = original_model
        self.compiled_forward_lower: torch.nn.Module | None = None
        self._task_buf_order = task_buf_order
        self._structure_key = structure_key
        self._compile_opts = compile_opts
        # Stored only for the first-forward compile call; freed afterwards.
        self._task_buffers = task_buffers
        # Shared dict across all _CompiledModel instances in the same Trainer.
        # A cache hit lets a second task with the same structure reuse the
        # already-traced graph without re-running make_fx.
        self._compiled_by_structure: dict = (
            compiled_by_structure if compiled_by_structure is not None else {}
        )
        # Resolved on the first forward: whether to compile the GRAPH lower
        # (graph-eligible mixed_types descriptors) or the dense forward_lower.
        self._graph_eligible: bool | None = None

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown lookups to original_model so that callers such as
        # share_params (which calls .get_descriptor(), .atomic_model, etc.) and
        # _compile_model (which calls .get_rcut(), .get_sel()) keep working
        # transparently after compilation replaces the plain model with this
        # wrapper.  nn.Module.__getattr__ is tried first so registered
        # submodules / parameters / buffers are never shadowed.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_model, name)

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        from deepmd.dpmodel.utils.nlist import (
            build_neighbor_list,
            extend_coord_with_ghosts,
        )
        from deepmd.dpmodel.utils.region import (
            normalize_coord,
        )

        nframes, nloc = atype.shape[:2]
        rcut = self.original_model.get_rcut()

        # Graph-eligible models (dpa1 concat-tebd, incl. attention) default-flip to the carry-all
        # GRAPH forward in eager training; the compiled lower must be the GRAPH
        # lower too, otherwise the eager (graph) and compiled (dense) backward
        # gradients diverge at fp64 accumulation and the optimizer amplifies it.
        if self._graph_eligible is None:
            self._graph_eligible = _model_uses_graph_lower(self.original_model)
        if self._graph_eligible:
            return self._forward_graph(
                coord, atype, box, fparam, aparam, charge_spin, nframes, nloc, rcut
            )

        sel = self.original_model.get_sel()

        # coord extension + nlist (data-dependent, run in eager)
        coord_3d = coord.detach().reshape(nframes, nloc, 3)
        box_flat = box.detach().reshape(nframes, 9) if box is not None else None

        if box_flat is not None:
            coord_norm = normalize_coord(coord_3d, box_flat.reshape(nframes, 3, 3))
        else:
            coord_norm = coord_3d

        ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
            coord_norm, atype, box_flat, rcut
        )
        nlist = build_neighbor_list(
            ext_coord,
            ext_atype,
            nloc,
            rcut,
            sel,
            distinguish_types=False,
            # model-level pair exclusion is a nlist-BUILD transform (decision
            # #18/A4); the compiled dense lower consumes a pre-excluded nlist.
            pair_excl=getattr(self.original_model.atomic_model, "pair_excl", None),
        )
        ext_coord = ext_coord.reshape(nframes, -1, 3)

        # Mirror the uncompiled path's optional-input defaulting (see
        # ``SeZMModel._forward_common`` -> ``convert_fparam_aparam`` /
        # ``convert_charge_spin``): a model configured with fparam or
        # charge_spin (``dim > 0``) substitutes its default when the data
        # omits it.  The compiled ``forward_lower`` is frozen to the *traced*
        # branch -- a present optional input bakes ``aten._to_copy(x, ...)``
        # into the graph, while an absent one is dropped during make_fx pytree
        # flattening -- so these inputs must be normalized to tensors here,
        # before both tracing and every compiled call.  Otherwise a graph
        # traced with the input present crashes when a later call (e.g. a
        # share_params task whose dataset omits it and relies on the default)
        # invokes it with None.  ``aparam`` has no default (it is required
        # whenever ``dim_aparam > 0``), so it needs no normalization; a genuine
        # absence is reported by ``forward_lower`` itself, as in eager mode.
        # ``get_default_*`` may return either a tensor or a raw ``list[float]``
        # (the sezm descriptor stores ``default_chg_spin`` as a list, and only
        # ``sezm_atomic_model`` wraps it via ``new_tensor``; the dp_atomic_model
        # family returns the descriptor list as-is), so coerce with
        # ``torch.as_tensor`` and ``reshape`` to ``(1, dim)`` before broadcasting.
        _model = self.original_model
        _dim_fparam = (
            _model.get_dim_fparam() if hasattr(_model, "get_dim_fparam") else 0
        )
        if fparam is None and _dim_fparam > 0:
            _default_fparam = _model.get_default_fparam()
            if _default_fparam is not None:
                fparam = (
                    torch.as_tensor(
                        _default_fparam, dtype=ext_coord.dtype, device=ext_coord.device
                    )
                    .reshape(1, _dim_fparam)
                    .expand(nframes, -1)
                )
        _dim_cs = (
            _model.get_dim_chg_spin() if hasattr(_model, "get_dim_chg_spin") else 0
        )
        if charge_spin is None and _dim_cs > 0:
            _default_cs = _model.get_default_chg_spin()
            if _default_cs is not None:
                charge_spin = (
                    torch.as_tensor(
                        _default_cs, dtype=ext_coord.dtype, device=ext_coord.device
                    )
                    .reshape(1, _dim_cs)
                    .expand(nframes, -1)
                )

        # Lazy compile: trace on the first real forward call using this
        # batch's tensors (prime-padded inside _trace_and_compile).
        # Mirrors DPA4's on-cache-miss compile so no separate get_data()
        # is needed during __init__.
        if self.compiled_forward_lower is None:
            # Optional inputs (fparam / charge_spin) are normalized to their
            # defaults above, so their presence is now config-driven (a
            # function of the model's ``dim_*``) rather than data-driven.
            # Tasks sharing this structure key share the same descriptor /
            # fitting net and therefore the same dims, so a single compiled
            # graph is safe to reuse across them.
            if self._structure_key in self._compiled_by_structure:
                compiled_lower, buf_order = self._compiled_by_structure[
                    self._structure_key
                ]
                log.info("Reusing compiled graph (shared model structure, lazy).")
            else:
                log.info(
                    "Lazy compile: tracing model on first forward call "
                    "(structure_key=%s).",
                    self._structure_key,
                )
                compiled_lower, buf_order = _trace_and_compile(
                    self.original_model,
                    ext_coord,
                    ext_atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    charge_spin=charge_spin,
                    task_buffers=self._task_buffers,
                    compile_opts=self._compile_opts,
                )
                self._compiled_by_structure[self._structure_key] = (
                    compiled_lower,
                    buf_order,
                )
            self.compiled_forward_lower = compiled_lower
            self._task_buf_order = buf_order
            self._task_buffers = None  # free; no longer needed after compile

        ext_coord = ext_coord.detach().requires_grad_(True)

        if self._task_buf_order:
            try:
                _fitting = self.original_model.get_fitting_net()
                _am = getattr(self.original_model, "atomic_model", None)
                _vals: list[torch.Tensor] = []
                for _name in self._task_buf_order:
                    if _name.startswith(_AM_PREFIX):
                        _actual = _name[len(_AM_PREFIX) :]
                        _vals.append(_am._buffers[_actual])
                    else:
                        _vals.append(getattr(_fitting, _name))
                task_buf_vals: tuple = tuple(_vals)
            except AttributeError as exc:
                raise RuntimeError(
                    f"Compiled graph expects task buffers {self._task_buf_order!r} "
                    "but they could not be retrieved from the model. "
                    "This is a bug in the compile path."
                ) from exc
        else:
            task_buf_vals = ()
        result = self.compiled_forward_lower(
            ext_coord,
            ext_atype,
            nlist,
            mapping,
            fparam,
            aparam,
            charge_spin,
            *task_buf_vals,
        )

        # Translate forward_lower keys -> forward keys.
        # ``extended_force`` lives on all extended atoms (nf, nall, 3).
        # Ghost-atom forces must be scatter-summed back to local atoms
        # via ``mapping`` — the same operation ``communicate_extended_output``
        # performs in the uncompiled path.
        out: dict[str, torch.Tensor] = {}
        out["atom_energy"] = result["atom_energy"]
        out["energy"] = result["energy"]
        if "extended_force" in result:
            ext_force = result["extended_force"]  # (nf, nall, 3)
            idx = mapping.unsqueeze(-1).expand_as(ext_force)  # (nf, nall, 3)
            force = torch.zeros(
                nframes, nloc, 3, dtype=ext_force.dtype, device=ext_force.device
            )
            force.scatter_add_(1, idx, ext_force)
            out["force"] = force
        if "virial" in result:
            out["virial"] = result["virial"]
        if "extended_virial" in result:
            out["extended_virial"] = result["extended_virial"]
        if "atom_virial" in result:
            out["atom_virial"] = result["atom_virial"]
        if "mask" in result:
            out["mask"] = result["mask"]
        return out

    def _forward_graph(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
        nframes: int,
        nloc: int,
        rcut: float,
    ) -> dict[str, torch.Tensor]:
        """Carry-all GRAPH forward -> compiled ``forward_common_lower_graph``.

        Builds the carry-all NeighborGraph eagerly (the SAME builder the eager
        uncompiled default-flip uses, so the graph tensors are bit-identical),
        then calls the compiled graph lower.  The graph force is per-LOCAL-node
        ``(N, 3)`` with ``N == nframes * nloc`` for a single-rank carry-all graph,
        so no extended->local scatter is needed; only the flat ``(N, *)`` node
        keys are unravelled to ``(nf, nloc, *)`` at the I/O boundary.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        _model = self.original_model

        coord_3d = coord.detach().reshape(nframes, nloc, 3)
        box_flat = box.detach().reshape(nframes, 9) if box is not None else None

        # Mirror the optional-input defaulting of the dense path / eager
        # call_common: a model configured with fparam / charge_spin substitutes
        # its default when the data omits it, so the compiled (frozen) branch
        # always sees a tensor.
        _dim_fparam = (
            _model.get_dim_fparam() if hasattr(_model, "get_dim_fparam") else 0
        )
        if fparam is None and _dim_fparam > 0:
            _default_fparam = _model.get_default_fparam()
            if _default_fparam is not None:
                fparam = (
                    torch.as_tensor(
                        _default_fparam, dtype=coord_3d.dtype, device=coord_3d.device
                    )
                    .reshape(1, _dim_fparam)
                    .expand(nframes, -1)
                )
        _dim_cs = (
            _model.get_dim_chg_spin() if hasattr(_model, "get_dim_chg_spin") else 0
        )
        if charge_spin is None and _dim_cs > 0:
            _default_cs = _model.get_default_chg_spin()
            if _default_cs is not None:
                charge_spin = (
                    torch.as_tensor(
                        _default_cs, dtype=coord_3d.dtype, device=coord_3d.device
                    )
                    .reshape(1, _dim_cs)
                    .expand(nframes, -1)
                )

        # Carry-all graph (dynamic E, no edge_capacity) — identical to the eager
        # uncompiled ``_call_common_graph`` builder so the two paths match. Model-
        # level pair_exclude is a graph-BUILD transform (decision #18): fold it
        # into edge_mask here so the compiled lower consumes a pre-excluded graph
        # (the lower no longer re-applies it), matching the eager path exactly.
        pair_excl = getattr(_model.atomic_model, "pair_excl", None)
        ng = build_neighbor_graph(coord_3d, atype, box_flat, rcut, pair_excl=pair_excl)
        atype_flat = atype.reshape(nframes * nloc)

        # Lazy compile of the GRAPH lower (cached per structure key).
        if self.compiled_forward_lower is None:
            if self._structure_key in self._compiled_by_structure:
                compiled_lower, buf_order = self._compiled_by_structure[
                    self._structure_key
                ]
                log.info("Reusing compiled graph lower (shared structure, lazy).")
            else:
                log.info(
                    "Lazy compile (graph lower): tracing on first forward call "
                    "(structure_key=%s).",
                    self._structure_key,
                )
                compiled_lower, buf_order = _trace_and_compile_graph(
                    _model,
                    fparam,
                    aparam,
                    charge_spin,
                    task_buffers=self._task_buffers,
                    compile_opts=self._compile_opts,
                )
                self._compiled_by_structure[self._structure_key] = (
                    compiled_lower,
                    buf_order,
                )
            self.compiled_forward_lower = compiled_lower
            self._task_buf_order = buf_order
            self._task_buffers = None

        # Feed a detached, grad-enabled edge_vec leaf: the traced graph's internal
        # ``edge_vec.detach()`` is stripped by ``_strip_saved_tensor_detach`` (as
        # for the dense ext_coord leaf), so the force backward roots at this input.
        edge_vec = ng.edge_vec.detach().requires_grad_(True)

        if self._task_buf_order:
            try:
                _fitting = _model.get_fitting_net()
                _am = getattr(_model, "atomic_model", None)
                _vals: list[torch.Tensor] = []
                for _name in self._task_buf_order:
                    if _name.startswith(_AM_PREFIX):
                        _actual = _name[len(_AM_PREFIX) :]
                        _vals.append(_am._buffers[_actual])
                    else:
                        _vals.append(getattr(_fitting, _name))
                task_buf_vals: tuple = tuple(_vals)
            except AttributeError as exc:
                raise RuntimeError(
                    f"Compiled graph expects task buffers {self._task_buf_order!r} "
                    "but they could not be retrieved from the model. "
                    "This is a bug in the compile path."
                ) from exc
        else:
            task_buf_vals = ()

        result = self.compiled_forward_lower(
            atype_flat,
            ng.n_node,
            ng.edge_index,
            edge_vec,
            ng.edge_mask,
            fparam,
            aparam,
            charge_spin,
            *task_buf_vals,
        )

        # The compiled graph lower emits PUBLIC keys on the FLAT node axis
        # (``atom_energy`` / ``force`` are (N, *); ``energy`` / ``virial`` are
        # (nf, *)).  Unravel the node-level keys to rectangular (nf, nloc, *) so
        # callers receive the same shapes as the dense path.
        N = nframes * nloc
        # Node-level (per-atom, lead dim N) public keys emitted by the graph
        # lower; the remaining keys are frame-level (lead dim nf) and must NOT
        # be unravelled. Keying on the NAME rather than the ``N != nframes``
        # shape heuristic keeps the single-atom case (nloc == 1, where
        # N == nframes) correct -- node-level outputs still reshape to
        # (nf, 1, *) instead of staying (nf, *).
        node_level_keys = {"atom_energy", "force", "atom_virial", "mask"}
        out: dict[str, torch.Tensor] = {}
        for key, val in result.items():
            if (
                key in node_level_keys
                and val is not None
                and val.shape[:1] == torch.Size([N])
            ):
                out[key] = val.reshape(nframes, nloc, *val.shape[1:])
            else:
                out[key] = val
        return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer(AbstractTrainer):
    """Training driver for the pt_expt backend.

    Uses ``DeepmdDataSystem`` for data loading (numpy batches converted
    to torch tensors at the boundary).  Supports single-task and multi-task
    training.  Single-GPU only.

    Parameters
    ----------
    config : dict
        Full training configuration.
    training_data : DeepmdDataSystem or dict
        Training data.  Dict of ``{model_key: DeepmdDataSystem}`` for multi-task.
    stat_file_path : DPPath or dict or None
        Path for saving / loading statistics.
    validation_data : DeepmdDataSystem or dict or None
        Validation data.
    init_model : str or None
        Path to a checkpoint to initialise weights from.
    restart_model : str or None
        Path to a checkpoint to *restart* training from (restores step + optimiser).
    shared_links : dict or None
        Parameter sharing rules for multi-task training.
    """

    def __init__(
        self,
        config: dict[str, Any],
        training_data: DeepmdDataSystem | dict,
        stat_file_path: DPPath | dict | None = None,
        validation_data: DeepmdDataSystem | dict | None = None,
        init_model: str | None = None,
        restart_model: str | None = None,
        finetune_model: str | None = None,
        finetune_links: dict | None = None,
        shared_links: dict | None = None,
    ) -> None:
        if finetune_model is not None and (
            init_model is not None or restart_model is not None
        ):
            raise ValueError(
                "finetune_model cannot be combined with init_model or restart_model."
            )
        resume_model = init_model or restart_model or finetune_model
        resuming = resume_model is not None
        self.restart_training = restart_model is not None

        model_params = config["model"]
        training_params = config["training"]
        validating_params = config.get("validating", {}) or {}

        # Task normalization --------------------------------------------------
        self.multi_task = "model_dict" in model_params
        self.model_keys = (
            list(model_params["model_dict"]) if self.multi_task else [DEFAULT_TASK_KEY]
        )
        self.num_model = len(self.model_keys)
        self.model_params_by_task = (
            {
                model_key: model_params["model_dict"][model_key]
                for model_key in self.model_keys
            }
            if self.multi_task
            else {DEFAULT_TASK_KEY: model_params}
        )
        self.training_data_by_task = _as_task_map(
            training_data,
            multi_task=self.multi_task,
            model_keys=self.model_keys,
        )
        self.validation_data_by_task = _as_task_map(
            validation_data,
            multi_task=self.multi_task,
            model_keys=self.model_keys,
        )
        self.stat_file_path_by_task = _as_task_map(
            stat_file_path,
            multi_task=self.multi_task,
            model_keys=self.model_keys,
        )

        # Distributed training detection
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        # Iteration config
        self.num_steps = training_params["numb_steps"]
        self.disp_file = training_params.get("disp_file", "lcurve.out")
        self.disp_freq = training_params.get("disp_freq", 1000)
        self.save_ckpt = training_params.get("save_ckpt", "model.ckpt")
        self.save_freq = training_params.get("save_freq", 1000)
        self.max_ckpt_keep = int(training_params.get("max_ckpt_keep", 5))
        self.display_in_training = training_params.get("disp_training", True)
        self.timing_in_training = training_params.get("time_training", True)
        self.change_bias_after_training = bool(
            training_params.get("change_bias_after_training", False)
        )

        # Model ---------------------------------------------------------------
        self.models: dict[str, torch.nn.Module] = {}
        do_case_embd, case_embd_index = (
            _get_case_embd_config(model_params) if self.multi_task else (False, {})
        )
        for model_key in self.model_keys:
            self.models[model_key] = get_model(
                deepcopy(self.model_params_by_task[model_key])
            ).to(DEVICE)
            if do_case_embd and not resuming:
                self.models[model_key].set_case_embd(case_embd_index[model_key])
        self.model = self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]

        # Loss ----------------------------------------------------------------
        self.losses: dict[str, EnergyLoss] = {}
        for model_key in self.model_keys:
            loss_param = (
                config["loss_dict"][model_key]
                if self.multi_task
                else config.get("loss", {})
            )
            self.losses[model_key] = get_loss(
                deepcopy(loss_param),
                config["learning_rate"]["start_lr"],
                len(self.model_params_by_task[model_key]["type_map"]),
                self.models[model_key],
            )
        self.loss = self.losses if self.multi_task else self.losses[DEFAULT_TASK_KEY]

        # Data requirements ---------------------------------------------------
        self.valid_numb_batch_by_task: dict[str, int] = {}
        for model_key in self.model_keys:
            data_requirement = list(self.losses[model_key].label_requirement)
            data_requirement += get_additional_data_requirement(self.models[model_key])
            self.training_data_by_task[model_key].add_data_requirements(
                data_requirement
            )
            if self.validation_data_by_task[model_key] is not None:
                self.validation_data_by_task[model_key].add_data_requirements(
                    data_requirement
                )
            if self.multi_task:
                valid_params = (
                    training_params["data_dict"][model_key].get("validation_data", {})
                    or {}
                )
            else:
                valid_params = training_params.get("validation_data", {}) or {}
            self.valid_numb_batch_by_task[model_key] = max(
                int(valid_params.get("numb_btch", 1)),
                1,
            )
        self.training_data = (
            self.training_data_by_task
            if self.multi_task
            else self.training_data_by_task[DEFAULT_TASK_KEY]
        )
        self.validation_data = (
            self.validation_data_by_task
            if self.multi_task
            else self.validation_data_by_task[DEFAULT_TASK_KEY]
        )
        self.valid_numb_batch = (
            self.valid_numb_batch_by_task
            if self.multi_task
            else self.valid_numb_batch_by_task[DEFAULT_TASK_KEY]
        )

        # Statistics ----------------------------------------------------------
        self._finetune_update_stat = False
        self._sample_funcs: dict[str, Any] = {}
        for model_key in self.model_keys:
            _nbatch = self.model_params_by_task[model_key].get("data_stat_nbatch", 10)
            _data = self.training_data_by_task[model_key]
            _stat_path = self.stat_file_path_by_task[model_key]

            @functools.lru_cache
            def _make_sample(
                _d: DeepmdDataSystem = _data, _n: int = _nbatch
            ) -> list[dict[str, np.ndarray]]:
                return make_stat_input(_d, _n)

            self._sample_funcs[model_key] = _make_sample

            _finetune_has_new_type = (
                finetune_model is not None
                and finetune_links is not None
                and model_key in finetune_links
                and finetune_links[model_key].get_has_new_type()
            )
            if _finetune_has_new_type:
                self._finetune_update_stat = True
            if (not resuming or _finetune_has_new_type) and self.rank == 0:
                self.models[model_key].compute_or_load_stat(
                    sampled_func=_make_sample,
                    stat_file_path=_stat_path,
                )
        if self.is_distributed:
            for model_key in self.model_keys:
                self._broadcast_model_stat(self.models[model_key])

        # Model probability (multi-task) --------------------------------------
        if self.multi_task:
            from deepmd.dpmodel.utils.training_utils import (
                resolve_model_prob,
            )

            self.model_prob = resolve_model_prob(
                self.model_keys,
                training_params.get("model_prob"),
                self.training_data_by_task,
            )
        else:
            self.model_prob = None

        # Learning rate -------------------------------------------------------
        self.lr_schedule = make_learning_rate_schedule(
            config["learning_rate"], self.num_steps
        )

        # Gradient clipping
        self.gradient_max_norm = training_params.get("gradient_max_norm", 0.0)

        # Model wrapper -------------------------------------------------------
        self.wrapper = ModelWrapper(self.model, self.loss, model_params=model_params)
        self.start_step = 0

        # Shared params (multi-task) ------------------------------------------
        self._shared_links = shared_links
        if shared_links is not None:
            _data_stat_protect = np.array(
                [
                    model_params["model_dict"][ii].get("data_stat_protect", 1e-2)
                    for ii in model_params["model_dict"]
                ]
            )
            if not np.allclose(_data_stat_protect, _data_stat_protect[0]):
                raise ValueError(
                    "Model key 'data_stat_protect' must be the same in each branch when multitask!"
                )
            self.wrapper.share_params(
                shared_links,
                resume=(resuming and not self._finetune_update_stat) or self.rank != 0,
                model_key_prob_map=dict(
                    zip(self.model_keys, self.model_prob, strict=True)
                ),
                data_stat_protect=_data_stat_protect[0],
            )

        # DDP wrapping --------------------------------------------------------
        if self.is_distributed:
            # Multi-task uses only one fitting_net per step, so unused
            # parameters exist in the graph. Single-task doesn't need this.
            _find_unused = self.multi_task
            if DEVICE.type == "cuda":
                from deepmd.pt_expt.utils.env import (
                    LOCAL_RANK,
                )

                torch.cuda.set_device(LOCAL_RANK)
                self.wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.wrapper,
                    device_ids=[LOCAL_RANK],
                    find_unused_parameters=_find_unused,
                    output_device=LOCAL_RANK,
                )
            else:
                # CPU (gloo backend) — no device_ids
                self.wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.wrapper,
                    find_unused_parameters=_find_unused,
                )

        # Optimiser -----------------------------------------------------------
        opt_type = training_params.get("opt_type", "Adam")
        # LambdaLR multiplies each param group's initial learning rate by the
        # lambda value.  Warmup schedules legitimately return zero at step 0,
        # so use the nonzero schedule base as the denominator and let the
        # lambda initialize the optimizer to the requested warmup value.
        initial_lr = float(self.lr_schedule.start_lr)

        if opt_type == "Adam":
            self.optimizer = torch.optim.Adam(self.wrapper.parameters(), lr=initial_lr)
        elif opt_type == "AdamW":
            weight_decay = training_params.get("weight_decay", 0.001)
            self.optimizer = torch.optim.AdamW(
                self.wrapper.parameters(),
                lr=initial_lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        for param_group in self.optimizer.param_groups:
            param_group["initial_lr"] = initial_lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: self.lr_schedule.value(step) / initial_lr,
            last_epoch=self.start_step - 1,
        )

        # Resume --------------------------------------------------------------
        if resuming:
            log.info(f"Resuming from {resume_model}.")
            is_pte = resume_model.endswith((".pte", ".pt2"))

            if is_pte:
                # .pte frozen model: no optimizer state, no step counter
                optimizer_state_dict = None
                self.start_step = 0
            else:
                state_dict = torch.load(
                    resume_model, map_location=DEVICE, weights_only=True
                )
                if "model" in state_dict:
                    optimizer_state_dict = (
                        state_dict["optimizer"]
                        if self.restart_training and finetune_model is None
                        else None
                    )
                    state_dict = state_dict["model"]
                else:
                    optimizer_state_dict = None
                self.start_step = (
                    state_dict["_extra_state"]["train_infos"]["step"]
                    if self.restart_training
                    else 0
                )

            if finetune_model is not None and finetune_links is not None:
                # --- Finetune: selective weight loading -----------------------

                # Build pretrained model(s) and load weights
                if is_pte:
                    from deepmd.pt_expt.model import (
                        BaseModel,
                    )
                    from deepmd.pt_expt.utils.serialization import (
                        serialize_from_file,
                    )

                    data = serialize_from_file(finetune_model)
                    pretrained_model_params = data["model_def_script"]
                    pretrained_model = BaseModel.deserialize(data["model"]).to(DEVICE)
                else:
                    pretrained_model_params = state_dict["_extra_state"]["model_params"]

                # Build pretrained model (single-task or multi-task)
                if "model_dict" not in pretrained_model_params:
                    # Single-task pretrained → wrap as {"Default": model}
                    if is_pte:
                        pretrained_models = pretrained_model
                    else:
                        pretrained_models = get_model(
                            deepcopy(pretrained_model_params)
                        ).to(DEVICE)
                else:
                    pretrained_models = {}
                    for pk in pretrained_model_params["model_dict"]:
                        pretrained_models[pk] = get_model(
                            deepcopy(pretrained_model_params["model_dict"][pk])
                        ).to(DEVICE)
                pretrained_wrapper = ModelWrapper(pretrained_models)
                if not is_pte:
                    pretrained_wrapper.load_state_dict(state_dict)

                # Per-branch type map change
                for model_key in self.model_keys:
                    finetune_rule = finetune_links[model_key]
                    _model_key_from = finetune_rule.get_model_branch()
                    if (
                        finetune_rule.get_finetune_tmap()
                        != pretrained_wrapper.model[_model_key_from].get_type_map()
                    ):
                        model_with_new_type_stat = (
                            self._unwrapped.model[model_key]
                            if finetune_rule.get_has_new_type()
                            else None
                        )
                        pretrained_wrapper.model[_model_key_from].change_type_map(
                            finetune_rule.get_finetune_tmap(),
                            model_with_new_type_stat=model_with_new_type_stat,
                        )

                for model_key in self.model_keys:
                    finetune_rule = finetune_links[model_key]
                    _model_key_from = finetune_rule.get_model_branch()
                    input_model_params = (
                        model_params["model_dict"][model_key]
                        if self.multi_task
                        else model_params
                    )
                    branch_pretrained_model_params = (
                        pretrained_model_params["model_dict"][_model_key_from]
                        if "model_dict" in pretrained_model_params
                        else pretrained_model_params
                    )
                    if (
                        "descriptor" in input_model_params
                        and "descriptor" in branch_pretrained_model_params
                    ):
                        warn_configuration_mismatch_during_finetune(
                            input_model_params["descriptor"],
                            branch_pretrained_model_params["descriptor"],
                            _model_key_from,
                        )

                # Selective weight copy (per-branch key remapping)
                pretrained_state = pretrained_wrapper.state_dict()
                target_state = self._unwrapped.state_dict()
                new_state = {}
                for key in target_state:
                    if key == "_extra_state":
                        new_state[key] = target_state[key]
                        continue
                    # Find which model_key this key belongs to
                    matched = False
                    for model_key in self.model_keys:
                        if f".{model_key}." not in key:
                            continue
                        matched = True
                        finetune_rule = finetune_links[model_key]
                        _key_from = finetune_rule.get_model_branch()
                        pretrained_key = key.replace(f".{model_key}.", f".{_key_from}.")
                        use_random = (
                            finetune_rule.get_random_fitting()
                            and ".descriptor." not in key
                        )
                        if use_random:
                            new_state[key] = target_state[key]
                        elif pretrained_key in pretrained_state:
                            new_state[key] = pretrained_state[pretrained_key]
                        else:
                            new_state[key] = target_state[key]
                        break
                    if not matched:
                        new_state[key] = target_state[key]
                self._unwrapped.load_state_dict(new_state)

                # Per-branch bias adjustment (rank 0 only, then broadcast)
                for model_key in self.model_keys:
                    finetune_rule = finetune_links[model_key]
                    if finetune_rule.get_resuming():
                        log.info(f"Model branch {model_key} will resume training.")
                        continue
                    if self.multi_task:
                        log.info(f"Model branch {model_key} will be fine-tuned.")
                    bias_mode = (
                        "change-by-statistic"
                        if not finetune_rule.get_random_fitting()
                        else "set-by-statistic"
                    )
                    if self.rank == 0:
                        self.models[model_key] = model_change_out_bias(
                            self.models[model_key],
                            self._sample_funcs[model_key],
                            _bias_adjust_mode=bias_mode,
                        )
                    if self.is_distributed:
                        self._broadcast_model_stat(self.models[model_key])
                self.model = (
                    self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]
                )
            else:
                # --- Normal resume (init_model / restart) --------------------
                self._unwrapped.load_state_dict(state_dict)

            if shared_links is not None:
                # Re-apply sharing after loading checkpoint
                self._unwrapped.share_params(
                    shared_links,
                    resume=True,
                    model_key_prob_map=dict(
                        zip(self.model_keys, self.model_prob, strict=True)
                    ),
                )

            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
                for param_group in self.optimizer.param_groups:
                    param_group["initial_lr"] = initial_lr
                # rebuild scheduler from the resumed step.
                # last_epoch handles the step offset; the lambda must NOT
                # add self.start_step again (that would double-count).
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lambda step: self.lr_schedule.value(step) / initial_lr,
                    last_epoch=self.start_step - 1,
                )

        # torch.compile -------------------------------------------------------
        self.enable_compile = training_params.get("enable_compile", False)
        if self.enable_compile:
            compile_opts = training_params.get("compile_options", {})
            log.info("Compiling model with torch.compile (%s)", compile_opts)
            self._compile_model(compile_opts)

        self.training_tasks = self._make_training_tasks()
        super().__init__(
            TrainerConfig.from_training_params(
                training_params,
                num_steps=self.num_steps,
                start_step=self.start_step,
                restart_training=self.restart_training,
            ),
            rank_context=RankContext(rank=self.rank, world_size=self.world_size),
        )
        self.full_validator = self._create_full_validator(
            validating_params=validating_params,
            validation_data=self.validation_data if not self.multi_task else None,
        )

    def _create_full_validator(
        self,
        *,
        validating_params: dict[str, Any],
        validation_data: Any | None,
    ) -> FullValidator | None:
        """Create the runtime full validator when it is active."""
        if not self._is_validation_requested(validating_params, "full_validation"):
            return None
        self._raise_if_full_validation_unsupported(validation_data)
        if validation_data is None:
            raise RuntimeError(
                "validation_data must be available after full validation checks."
            )
        return FullValidator(
            validating_params=validating_params,
            validation_data=validation_data,
            model=self.models[DEFAULT_TASK_KEY],
            state_store=self._unwrapped.train_infos,
            num_steps=self.num_steps,
            rank=self.rank,
            zero_stage=0,
            restart_training=self.restart_training,
            checkpoint_dir=resolve_best_checkpoint_dir(
                validating_params, self.save_ckpt
            ),
        )

    def _is_validation_requested(
        self,
        validating_params: dict[str, Any],
        flag_name: str,
    ) -> bool:
        """Check whether a full validation flow can trigger during this run."""
        if not validating_params.get(flag_name, False):
            return False
        start_step = resolve_full_validation_start_step(
            validating_params.get("full_val_start", 0.5),
            self.num_steps,
        )
        return start_step is not None and start_step <= self.num_steps

    def _raise_if_full_validation_unsupported(
        self,
        validation_data: Any | None,
    ) -> None:
        """Validate runtime full validation constraints."""
        if self.multi_task:
            raise ValueError(
                "validating.full_validation only supports single-task energy "
                "training; multi-task training is not supported."
            )

        has_spin = getattr(self.models[DEFAULT_TASK_KEY], "has_spin", False)
        if callable(has_spin):
            has_spin = has_spin()
        if has_spin or isinstance(self.loss, EnergySpinLoss):
            raise ValueError(
                "validating.full_validation only supports single-task energy "
                "training; spin-energy training is not supported."
            )

        if not isinstance(self.loss, EnergyLoss):
            raise ValueError(
                "validating.full_validation only supports single-task energy training."
            )

        if validation_data is None:
            raise ValueError(
                "validating.full_validation requires `training.validation_data` "
                "to be configured."
            )

    # ------------------------------------------------------------------
    # torch.compile helpers
    # ------------------------------------------------------------------

    def _compile_model(self, compile_opts: dict[str, Any]) -> None:
        """Replace ``self.model`` with a compiled version.

        The model's ``forward`` uses ``torch.autograd.grad`` (for force
        computation) with ``create_graph=True``, which creates a "double
        backward" that ``torch.compile`` cannot handle.

        Solution: use ``make_fx`` in ``tracing_mode="symbolic"`` to trace
        ``forward_lower``, decomposing ``torch.autograd.grad`` into
        primitive ops.  The symbolic trace keeps the extended-atom
        dimension (``nall``) and batch dimension (``nframes``) as
        symbolic shapes, so no padding or recompile-on-growth logic is
        needed.  The coord extension + nlist build (data-dependent
        control flow) are kept outside the compiled region.
        """
        # Disable DDPOptimizer: our compile region wraps only the inner
        # compute function, not the whole DDP model.  DDPOptimizer assumes
        # it owns the full model graph and splits at bucket boundaries,
        # producing subgraphs whose outputs include symbolic integers.
        # AOT Autograd then crashes with ``'int' object has no attribute
        # 'meta'`` (pytorch/pytorch#134182).
        torch._dynamo.config.optimize_ddp = False

        # Under DDP, self.wrapper is a DistributedDataParallel wrapper;
        # access the underlying ModelWrapper via .module.
        wrapper_mod = (
            self.wrapper.module
            if isinstance(self.wrapper, torch.nn.parallel.DistributedDataParallel)
            else self.wrapper
        )

        from collections import (
            defaultdict,
        )

        from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP

        # Pre-pass: group tasks by structure key and auto-detect per-task buffers.
        # Grouping is needed so _detect_task_buffers can diff buffer identities
        # across all tasks that share the same compiled graph.
        _key_for: dict[str, tuple[int, ...]] = {}
        _groups: defaultdict[tuple[int, ...], list[str]] = defaultdict(list)
        for task_key in self.model_keys:
            sk = _get_model_structure_key(wrapper_mod.model[task_key])
            _key_for[task_key] = sk
            _groups[sk].append(task_key)

        # Reject partial descriptor sharing (shared_level > 0) with torch.compile.
        # The compiled graph bakes the first task's descriptor constants, so tasks
        # sharing a graph must have identical descriptor parameters.  partial sharing
        # (e.g. shared_level=1, type_embedding shared but main block task-local)
        # violates this invariant.  Check directly from the config rather than
        # via parameter-identity heuristics.
        if self._shared_links is not None:
            for info in self._shared_links.values():
                for link_item in info["links"]:
                    if (
                        "descriptor" in link_item["shared_type"]
                        and int(link_item["shared_level"]) > 0
                    ):
                        raise RuntimeError(
                            f"torch.compile is incompatible with partial descriptor "
                            f"sharing (task {link_item['model_key']!r}, "
                            f"shared_level={link_item['shared_level']}). "
                            f"Use shared_level=0 for all descriptors, "
                            f"or set 'enable_compile: false'."
                        )

        _task_bufs_for: dict[str, dict[str, torch.Tensor]] = {}
        for group_keys in _groups.values():
            group_models = [wrapper_mod.model[k] for k in group_keys]
            for task_key in group_keys:
                _task_bufs_for[task_key] = _detect_task_buffers(
                    wrapper_mod.model[task_key], group_models
                )

        # Shared cache: structure_key -> (compiled_lower, task_buf_order).
        # Tasks with the same structure key reuse the same compiled graph.
        # The dict is passed to every _CompiledModel instance so the lazy
        # compile on the first forward can populate and share it.
        _compiled_by_structure: dict[tuple[int, ...], tuple] = {}

        for task_key in self.model_keys:
            model = wrapper_mod.model[task_key]

            # Compiled DPA1/se_atten_v2 attention is numerically more
            # sensitive than other descriptors: the inductor-fused and
            # eager force/grad outputs can diverge above 1e-10 on
            # multi-threaded CPU hosts because parallel reduction order
            # is hardware-dependent.  Warn but do not reject — energies
            # remain well within training tolerance and the user may
            # accept the trade-off for compile speed.
            descriptor = model.get_descriptor()
            if isinstance(descriptor, DescrptDPA1DP):
                n_attn = descriptor.get_numb_attn_layer()
                if n_attn > 0:
                    log.warning(
                        "Compiling DPA1/se_atten_v2 with %d attention "
                        "layer(s) (task=%s): the compiled forces/grads "
                        "are slightly hardware-sensitive (multi-thread "
                        "reduction order), and may not match the eager "
                        "path bit-for-bit.  Use 'enable_compile: false' "
                        "or 'attn_layer: 0' for fully reproducible runs.",
                        n_attn,
                        task_key,
                    )

            structure_key = _key_for[task_key]
            task_bufs = _task_bufs_for[task_key]

            wrapper_mod.model[task_key] = _CompiledModel(
                model,
                structure_key=structure_key,
                task_buf_order=tuple(task_bufs.keys()) if task_bufs else (),
                task_buffers=task_bufs if task_bufs else None,
                compile_opts=compile_opts,
                compiled_by_structure=_compiled_by_structure,
            )
            log.info(
                "Lazy compile registered (task=%s); will trace on first forward call.",
                task_key,
            )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def get_data(
        self,
        is_train: bool = True,
        task_key: str = "Default",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch a batch and split into input / label dicts.

        Parameters
        ----------
        is_train : bool
            Whether to fetch from training or validation data.
        task_key : str
            Task key for multi-task training.

        Returns
        -------
        input_dict, label_dict
        """
        task_key = task_key if self.multi_task else DEFAULT_TASK_KEY
        data_sys = (
            self.training_data_by_task[task_key]
            if is_train
            else self.validation_data_by_task[task_key]
        )
        if data_sys is None:
            return {}, {}

        batch = normalize_batch(data_sys.get_batch())
        input_dict, label_dict = split_batch(batch)

        # Drop optional inputs whose find_* flag is False so the model sees None.
        for opt_key in ("fparam", "charge_spin"):
            find_key = f"find_{opt_key}"
            if (
                opt_key in input_dict
                and find_key in label_dict
                and not bool(label_dict[find_key])
            ):
                input_dict.pop(opt_key)

        # Convert numpy values to torch tensors.
        for dd in (input_dict, label_dict):
            for key, val in dd.items():
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    if np.issubdtype(val.dtype, np.integer):
                        dd[key] = torch.from_numpy(val).to(DEVICE)
                    else:
                        dd[key] = torch.from_numpy(val).to(
                            dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                        )
                elif isinstance(val, (float, np.bool_)):
                    dd[key] = torch.tensor(
                        float(val), dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                    )
        # requires_grad on coord for force computation via autograd
        if "coord" in input_dict and input_dict["coord"] is not None:
            input_dict["coord"] = input_dict["coord"].requires_grad_(True)

        return input_dict, label_dict

    # ------------------------------------------------------------------
    # DDP helpers
    # ------------------------------------------------------------------

    @property
    def _unwrapped(self) -> "ModelWrapper":
        """Return the raw ModelWrapper, unwrapping DDP if active."""
        if hasattr(self.wrapper, "module"):
            return self.wrapper.module
        return self.wrapper

    @staticmethod
    def _broadcast_model_stat(model: torch.nn.Module) -> None:
        """Broadcast model parameters and buffers from rank 0 to all ranks."""
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        for b in model.buffers():
            dist.broadcast(b, src=0)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        ckpt_path = Path(f"{self.save_ckpt}-{step}.pt")
        self._save_checkpoint_to_path(ckpt_path, step=step)
        latest = Path(f"{self.save_ckpt}.pt")
        _replace_latest_checkpoint_link(latest, ckpt_path)
        self._cleanup_old_checkpoints()
        log.info(f"Saved checkpoint to {ckpt_path}")

    def _save_full_validation_checkpoint(
        self,
        save_path: Path,
        lr: float = 0.0,
        step: int = 0,
    ) -> None:
        """Save a checkpoint selected by full validation."""
        del lr
        self._save_checkpoint_to_path(save_path, step=step)

    def _save_checkpoint_to_path(self, ckpt_path: Path, *, step: int) -> None:
        """Serialize the current trainer state to an explicit checkpoint path."""
        self._unwrapped.train_infos["step"] = step
        # When compiled, wrapper.model[key] is _CompiledModel whose state_dict
        # uses keys like "original_model.*".  Restart would load into a plain
        # ModelWrapper expecting "model.{key}.*" keys → hard crash.  Temporarily
        # swap each _CompiledModel back to its original_model so the saved keys
        # match what a fresh __init__ expects, then restore.
        wrapper = self._unwrapped
        compiled_backup: dict[str, _CompiledModel] = {}
        for task_key in list(wrapper.model.keys()):
            m = wrapper.model[task_key]
            if isinstance(m, _CompiledModel):
                compiled_backup[task_key] = m
                wrapper.model[task_key] = m.original_model
        try:
            state = {
                "model": wrapper.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
        finally:
            for task_key, compiled in compiled_backup.items():
                wrapper.model[task_key] = compiled
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_path)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old step checkpoint files beyond the retention limit."""
        if self.max_ckpt_keep <= 0:
            return
        ckpt_prefix_path = Path(self.save_ckpt)
        ckpt_parent = ckpt_prefix_path.parent
        ckpt_prefix = ckpt_prefix_path.name
        checkpoints: list[tuple[int, Path]] = []
        for path in ckpt_parent.glob(f"{ckpt_prefix}-*.pt"):
            if path.is_dir() or path.is_symlink():
                continue
            step_text = path.name.removeprefix(f"{ckpt_prefix}-").removesuffix(".pt")
            if step_text.isdigit():
                checkpoints.append((int(step_text), path))
        for _, path in sorted(checkpoints)[: -self.max_ckpt_keep]:
            path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _optimizer_step(self) -> None:
        """Run optimizer and scheduler step outside torch._dynamo.

        Dynamo intercepts tensor creation inside Adam._init_group,
        which can trigger CUDA init on CPU-only builds.
        """
        self.optimizer.step()
        self.scheduler.step()

    def _make_training_tasks(self) -> TrainingTaskCollection:
        """Build the backend-independent task collection."""
        return TrainingTaskCollection(
            [
                TrainingTask(
                    key=model_key,
                    training_data=self.training_data_by_task[model_key],
                    validation_data=self.validation_data_by_task[model_key],
                    valid_numb_batch=self.valid_numb_batch_by_task[model_key],
                )
                for model_key in self.model_keys
            ],
            probabilities=self.model_prob,
        )

    def run(self) -> None:
        """Run pt_expt training through the backend-independent trainer loop."""
        log.info("Start to train %d steps.", self.num_steps)
        wall_start = time.time()
        super().run(self.training_tasks)
        if self.change_bias_after_training and self.num_steps > self.start_step:
            self._change_bias_after_training()
            if self.rank_context.is_chief:
                self.save_checkpoint(self.num_steps)
        log.info("Training finished. Total wall time: %.2fs", time.time() - wall_start)

    def _change_bias_after_training(self) -> None:
        if self.rank == 0:
            change_model_out_bias_by_task(
                self.models,
                self._sample_funcs,
                self.model_keys,
                bias_adjust_mode="change-by-statistic",
            )
        if self.is_distributed:
            for model_key in self.model_keys:
                self._broadcast_model_stat(self.models[model_key])
        self.model = self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]

    def run_full_validation(
        self,
        *,
        step: int,
        display_step: int,
        learning_rate: float,
    ) -> None:
        """Run optional full validation for one step."""
        if self.full_validator is None:
            return None
        self.full_validator.run(
            step_id=display_step,
            display_step=display_step,
            lr=learning_rate,
            save_checkpoint=self._save_full_validation_checkpoint,
        )
        return None

    def select_task(self, tasks: TrainingTaskCollection) -> TrainingTask:
        """Select a task using DeePMD's seeded random helper."""
        if not tasks.is_multitask:
            return tasks[tasks.keys[0]]
        from deepmd.utils import random as dp_random

        model_index = dp_random.choice(
            np.arange(len(tasks), dtype=np.int_),
            p=tasks.probabilities,
        )
        return tasks[tasks.keys[int(model_index)]]

    def on_train_begin(self, tasks: TrainingTaskCollection) -> None:
        """Switch the wrapper to training mode."""
        self.wrapper.train()

    def collect_display_results(
        self,
        tasks: TrainingTaskCollection,
        *,
        active_task: TrainingTask,
        step: int,
        step_result: TrainStepResult,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Collect display metrics in eval mode, then resume training mode."""
        self.wrapper.eval()
        self._display_cur_lr_sched = step_result.payload["cur_lr_sched"]
        try:
            return super().collect_display_results(
                tasks,
                active_task=active_task,
                step=step,
                step_result=step_result,
            )
        finally:
            self._display_cur_lr_sched = None
            self.wrapper.train()

    def train_step(self, task: TrainingTask, step: int) -> TrainStepResult:
        """Run one pt_expt optimizer step."""
        task_key = task.key
        self.optimizer.zero_grad(set_to_none=True)
        input_dict, label_dict = self.get_data(is_train=True, task_key=task_key)

        cur_lr_sched = self.scheduler.get_last_lr()[0]
        _model_pred, loss, more_loss = self.wrapper(
            **input_dict,
            cur_lr=cur_lr_sched,
            label=label_dict,
            task_key=task_key,
        )
        loss.backward()

        if self.gradient_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.wrapper.parameters(), self.gradient_max_norm
            )

        self._optimizer_step()
        return TrainStepResult(
            task_key=task_key,
            step=step,
            payload={
                "loss": loss,
                "more_loss": more_loss,
                "cur_lr_sched": cur_lr_sched,
            },
        )

    def evaluate_training(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> dict[str, float]:
        """Evaluate training loss terms for one task."""
        if step_result is not None and step_result.task_key == task.key:
            return self._more_loss_to_float(step_result.payload["more_loss"])

        self.optimizer.zero_grad()
        input_dict, label_dict = self.get_data(is_train=True, task_key=task.key)
        _, _loss, more_loss = self._unwrapped(
            **input_dict,
            cur_lr=self._get_display_cur_lr_sched(),
            label=label_dict,
            task_key=task.key,
        )
        return self._more_loss_to_float(more_loss)

    def evaluate_validation(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> dict[str, float] | None:
        """Evaluate validation loss terms for one task."""
        if task.validation_data is None:
            return None

        valid_results: dict[str, float] = {}
        sum_natoms = 0
        for _ii in range(task.valid_numb_batch):
            val_input, val_label = self.get_data(is_train=False, task_key=task.key)
            if not val_input:
                break
            _, _vloss, vmore = self._unwrapped(
                **val_input,
                cur_lr=self._get_display_cur_lr_sched(),
                label=val_label,
                task_key=task.key,
            )
            natoms = int(val_input["atype"].shape[-1])
            sum_natoms += natoms
            for key, value in vmore.items():
                if "l2_" not in key:
                    valid_results[key] = (
                        valid_results.get(key, 0.0) + self._to_float(value) * natoms
                    )
        if sum_natoms > 0:
            valid_results = {
                key: value / sum_natoms for key, value in valid_results.items()
            }
        return valid_results

    def learning_rate(self, step: int) -> float:
        """Return the configured learning rate for a zero-based step."""
        return float(self.lr_schedule.value(step))

    @staticmethod
    def _to_float(value: Any) -> float:
        return value.detach().item() if torch.is_tensor(value) else float(value)

    def _get_display_cur_lr_sched(self) -> float:
        cur_lr_sched = getattr(self, "_display_cur_lr_sched", None)
        if cur_lr_sched is None:
            cur_lr_sched = self.scheduler.get_last_lr()[0]
        return cur_lr_sched

    @classmethod
    def _more_loss_to_float(cls, more_loss: dict[str, Any]) -> dict[str, float]:
        return {
            key: cls._to_float(value)
            for key, value in more_loss.items()
            if "l2_" not in key
        }


def model_change_out_bias(
    _model: Any,
    _sample_func: Any,
    _bias_adjust_mode: str = "change-by-statistic",
) -> Any:
    """Change the output bias of a model based on sampled data.

    Parameters
    ----------
    _model
        The model whose bias should be adjusted.
    _sample_func
        Callable that returns sampled data for bias computation.
    _bias_adjust_mode
        ``"change-by-statistic"`` or ``"set-by-statistic"``.

    Returns
    -------
    The model with updated bias.
    """
    from deepmd.dpmodel.model.dp_model import (
        DPModelCommon,
    )

    return change_model_out_bias(
        _model,
        _sample_func,
        bias_adjust_mode=_bias_adjust_mode,
        recompute_input_stats=isinstance(_model, DPModelCommon),
    )


def _get_case_embd_config(
    model_params: dict[str, Any],
) -> tuple[bool, dict[str, int]]:
    """Check whether case embedding is enabled and build the index map.

    Parameters
    ----------
    model_params : dict
        Model parameters containing ``model_dict``.

    Returns
    -------
    do_case_embd : bool
        Whether case embedding is enabled.
    case_embd_index : dict
        Mapping from model key to case index (sorted alphabetically).
    """
    assert "model_dict" in model_params, (
        "Only support setting case embedding for multi-task model!"
    )
    model_keys = list(model_params["model_dict"])
    sorted_model_keys = sorted(model_keys)
    numb_case_embd_list = [
        model_params["model_dict"][mk].get("fitting_net", {}).get("dim_case_embd", 0)
        for mk in sorted_model_keys
    ]
    if not all(item == numb_case_embd_list[0] for item in numb_case_embd_list):
        raise ValueError(
            "All models must have the same dimension of case embedding, "
            f"while the settings are: {numb_case_embd_list}"
        )
    if numb_case_embd_list[0] == 0:
        return False, {}
    case_embd_index = {mk: idx for idx, mk in enumerate(sorted_model_keys)}
    return True, case_embd_index
