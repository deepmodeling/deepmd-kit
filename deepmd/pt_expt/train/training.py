# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training loop for the pt_expt backend.

Uses ``DeepmdDataSystem`` (numpy-based batch provider) instead of the
pt backend's ``DpLoaderSet`` + ``DataLoader``.  NumPy batches are
converted to torch tensors at the boundary.
"""

import functools
import logging
import time
from collections.abc import (
    Callable,
    Mapping,
)
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
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.optim import (
    ZeroRedundancyOptimizer,
)

try:
    from torch.distributed.fsdp import (
        fully_shard,
    )
except ImportError:
    fully_shard = None  # type: ignore[assignment]

from deepmd.dpmodel.train import (
    DEFAULT_TASK_KEY,
    AbstractTrainer,
    RankContext,
    ShardingPolicy,
    TrainerConfig,
    TrainingTask,
    TrainingTaskCollection,
    TrainStepResult,
    build_checkpoint_stores,
    change_model_out_bias,
    change_model_out_bias_by_task,
    resolve_step_schedule,
)
from deepmd.dpmodel.utils.batch import (
    normalize_batch,
    split_batch,
)
from deepmd.dpmodel.utils.learning_rate import (
    make_learning_rate_schedule,
)
from deepmd.dpmodel.utils.training_utils import (
    compute_total_numb_batch,
)
from deepmd.loggers.training import (
    log_parameter_counts,
)
from deepmd.pt.optimizer import (
    HybridMuonOptimizer,
)
from deepmd.pt.utils.compile_compat import (
    apply_global_compile_patches,
    build_inductor_compile_options,
    check_compile_torch_version,
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
from deepmd.pt_expt.model.graph_lower import (
    model_uses_graph_lower,
)
from deepmd.pt_expt.train.ema import (
    EMA_CHECKPOINT_KEY,
    ModelEMA,
    get_ema_checkpoint_prefix,
)
from deepmd.pt_expt.train.gradient import (
    NonFiniteGradGuard,
    clip_grad_norm_,
)
from deepmd.pt_expt.train.utils import (
    count_parameters,
    infer_env_defaults,
    resolve_best_checkpoint_dir,
    scoped_env_defaults,
)
from deepmd.pt_expt.train.validation import (
    FullValidator,
    build_full_validators,
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
from deepmd.utils.stat_file import (
    StatFileSpec,
    open_stat_file,
    run_stat_on_chief,
    stat_file_specs_by_task,
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


def _warn_compiled_attention(model: torch.nn.Module, task_key: str) -> None:
    """Warn when compiling DPA1/se_atten_v2 attention (hardware-sensitive).

    Compiled DPA1/se_atten_v2 attention is numerically more sensitive than
    other descriptors: the inductor-fused and eager force/grad outputs can
    diverge above 1e-10 on multi-threaded CPU hosts because parallel
    reduction order is hardware-dependent. Warn but do not reject —
    energies remain well within training tolerance and the user may accept
    the trade-off for compile speed.

    Compositions (``LinearEnergyAtomicModel``) have no single descriptor to
    probe; ``enable_compile`` must degrade gracefully for them instead of
    crashing on the reach-in (issue #5906 Task 4 audit).

    Parameters
    ----------
    model
        The per-task model about to be compiled.
    task_key
        The task label used in the warning message.
    """
    from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP

    try:
        descriptor = model.get_descriptor()
    except AttributeError:
        return
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
                source_policy="default" if has_default_fparam else "tracked",
            )
        )
    if _model.get_dim_aparam() > 0:
        additional_data_requirement.append(
            DataRequirementItem(
                "aparam", _model.get_dim_aparam(), atomic=True, must=True
            )
        )
    if _model.has_spin():
        # ``model.spin.allow_missing_label`` relaxes the spin label from
        # mandatory to optional with a zero default, so a system without a
        # ``spin`` file is filled with zeros rather than rejected. Mirrors
        # ``deepmd.pt.train.training.get_additional_data_requirement``.
        # Every spin model wrapper carries a ``spin`` attribute.
        allow_missing_spin = _model.spin.allow_missing_label
        additional_data_requirement.append(
            DataRequirementItem(
                "spin",
                ndof=3,
                atomic=True,
                must=not allow_missing_spin,
                default=0.0,
                source_policy="default" if allow_missing_spin else "tracked",
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
                source_policy="default" if has_default_cs else "tracked",
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

    # This is the common boundary immediately before every pt_expt
    # ``torch.compile`` call. Applying the idempotent process-global patches
    # here leaves eager-only imports untouched while still preceding all
    # Dynamo and Inductor configuration reads.
    apply_global_compile_patches()

    # Keep pt_expt training on the same compiler contract as the PT SeZM path.
    inductor_options = build_inductor_compile_options(inference=False)
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
    from torch._decomp import (
        get_decompositions,
    )
    from torch.fx.experimental.proxy_tensor import (
        make_fx,
    )

    from deepmd.pt_expt.model.make_model import (
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

    # Shared with the .pt2 export trace (serialization.py) so the two graph
    # traces can never desync on the input schema.  Training uses the run-time
    # float precision and device; optional tensors match the actual call.
    from deepmd.pt_expt.utils.serialization import (
        build_synthetic_graph_inputs,
        check_graph_trace_torch_version,
        count_synthetic_graph_edges,
    )

    check_graph_trace_torch_version(model)

    # Static edge capacity: derived from the ACTUAL edge count of the
    # synthetic trace system (the carry-all builder is sel-free; a
    # sel-derived estimate overflows whenever the real degree exceeds sel),
    # then prime-padded to stay distinct from nf and N.  ``+ 2`` keeps at
    # least two masked padding rows so the padded-tail branch is traced.
    # Trace on the MODEL's device, not the global ``DEVICE``: make_fx keeps the
    # real model parameters (``_allow_non_fake_inputs``), so the synthetic trace
    # inputs must live where the model does. A CUDA training run keeps the model
    # on ``DEVICE`` (these match), but callers that trace a CPU-placed model
    # (e.g. the graph .pt2/export path, which moves the model to CPU to dodge a
    # CUDA autograd-stream limitation) would otherwise mix a CPU model with
    # CUDA inputs and fail only on a GPU host.
    _trace_device = next(model.parameters()).device
    e_real = count_synthetic_graph_edges(
        model,
        nframes=trace_nf,
        nloc=nloc_trace,
        dtype=GLOBAL_PT_FLOAT_PRECISION,
        device=_trace_device,
    )
    e_max = _next_safe_prime(e_real + 2, _forbidden | {trace_nf, trace_N})
    sample = build_synthetic_graph_inputs(
        model,
        e_max=e_max,
        nframes=trace_nf,
        nloc=nloc_trace,
        dtype=GLOBAL_PT_FLOAT_PRECISION,
        device=_trace_device,
        want_fparam=fparam is not None,
        want_aparam=aparam is not None,
        want_charge_spin=charge_spin is not None,
    )
    (
        s_atype,
        s_n_node,
        s_n_local,
        s_edge_index,
        s_edge_vec,
        s_edge_mask,
        s_destination_order,
        s_destination_row_ptr,
        s_source_order,
        s_source_row_ptr,
        s_fparam,
        s_aparam,
        s_charge_spin,
    ) = sample

    def fn(
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        destination_order: torch.Tensor,
        destination_row_ptr: torch.Tensor,
        source_order: torch.Tensor,
        source_row_ptr: torch.Tensor,
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
                n_local,
                edge_index,
                edge_vec,
                edge_mask,
                destination_order,
                destination_row_ptr,
                source_order,
                source_row_ptr,
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
        s_n_local,
        s_edge_index,
        s_edge_vec,
        s_edge_mask,
        s_destination_order,
        s_destination_row_ptr,
        s_source_order,
        s_source_row_ptr,
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
        task_key: str = DEFAULT_TASK_KEY,
    ) -> None:
        super().__init__()
        self.original_model = original_model
        self.compiled_forward_lower: torch.nn.Module | None = None
        self._task_buf_order = task_buf_order
        self._structure_key = structure_key
        self._task_key = task_key
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

    def _compiled_lower_for(
        self,
        path: str,
        trace: "Callable[[], tuple[torch.nn.Module, tuple[str, ...]]]",
    ) -> tuple[torch.nn.Module, tuple[str, ...]]:
        """Return the compiled graph of this model, tracing it at most once.

        Tasks that share a model structure share one compiled graph, so a task
        reaching this point second only reports the reuse.

        Parameters
        ----------
        path : str
            Name of the lowering being compiled, as it appears in the log.
        trace : Callable[[], tuple[torch.nn.Module, tuple[str, ...]]]
            Traces and compiles the graph, returning it together with the
            order of the per-task buffers it expects.

        Returns
        -------
        tuple[torch.nn.Module, tuple[str, ...]]
            The compiled graph and its buffer order.
        """
        attributes = f"task={self._task_key}, path={path}"
        cached = self._compiled_by_structure.get(self._structure_key)
        if cached is not None:
            log.info("Reusing the graph compiled for an earlier task (%s).", attributes)
            return cached
        log.info("Tracing and compiling the model (%s).", attributes)
        started = time.perf_counter()
        compiled = trace()
        log.info(
            "Finished compiling (%s) in %.1f s.",
            attributes,
            time.perf_counter() - started,
        )
        self._compiled_by_structure[self._structure_key] = compiled
        return compiled

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

    def forward_ragged(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compiled forward over a batch whose node axis is already flat.

        The compiled lower works on that axis in either case -- its trace keeps
        the frame count, the node count and the edge count as independent
        symbols -- so a ragged batch simply skips the padding round trip the
        rectangular :meth:`forward` performs around it.

        Parameters
        ----------
        coord : torch.Tensor
            Local coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
        atype : torch.Tensor
            Local atom types with shape ``(N,)``.
        n_node : torch.Tensor
            Atoms per frame with shape ``(nf,)``.
        box : torch.Tensor or None, optional
            Simulation cell, ``(nf, 3, 3)`` or ``(nf, 9)``.
        fparam : torch.Tensor or None, optional
            Frame parameters with shape ``(nf, ndf)``.
        aparam : torch.Tensor or None, optional
            Atomic parameters with shape ``(N, nda)``.
        do_atomic_virial : bool, default: False
            Whether to return per-atom virials.
        charge_spin : torch.Tensor or None, optional
            Frame-level charge and spin conditioning with shape ``(nf, 2)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Public model keys; per-atom entries keep the flat axis.

        Raises
        ------
        NotImplementedError
            If the model reads a rectangular node axis, which cannot represent
            frames of unequal atom count without padding.
        """
        del do_atomic_virial
        if self._graph_eligible is None:
            self._graph_eligible = model_uses_graph_lower(self.original_model)
        if not self._graph_eligible:
            raise NotImplementedError(
                "a flat node axis requires a model whose descriptor reads one; "
                "this model compiles the dense (nlist) lower, whose batches "
                "must be padded to a common atom count"
            )
        return self._forward_graph(
            coord,
            atype,
            box,
            fparam,
            aparam,
            charge_spin,
            int(n_node.shape[0]),
            0,
            self.original_model.get_rcut(),
            n_node=n_node,
        )

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
            self._graph_eligible = model_uses_graph_lower(self.original_model)
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
            # Keep the candidate list merged, matching eager training's
            # DefaultNeighborList contract. forward_common_lower always calls
            # model.format_nlist, which performs the type split for non-mixed
            # descriptors. The shared builder globally truncates to sum(sel)
            # before either split, so distinguish_types=True would only move
            # the same layout transform earlier without changing the final
            # formatted neighbor list consumed by the lower model.
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
            compiled_lower, buf_order = self._compiled_lower_for(
                "neighbor-list",
                lambda: _trace_and_compile(
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
                ),
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

        # Translate forward_lower keys -> forward keys.  OUTPUT-AGNOSTIC:
        # every key passes through unchanged (energy models emit
        # atom_energy/energy/virial/..., property/dos/... models their own
        # keys -- the hardcoded energy-key copy here used to KeyError on any
        # non-energy fitting), EXCEPT the extended-region keys
        # ``extended_force`` (nf, nall, 3) and ``extended_virial``
        # (nf, nall, 9): their ghost rows are scatter-summed back onto
        # local owners via ``mapping`` -- the same fold
        # ``communicate_extended_output`` performs in the uncompiled path
        # (which exposes them as ``force`` / ``atom_virial``).  Folding
        # both keeps the compiled and uncompiled outputs key-for-key
        # consistent, including for a future atom-virial training
        # objective.
        out: dict[str, torch.Tensor] = {}
        if "extended_force" in result:
            ext_force = result["extended_force"]  # (nf, nall, 3)
            idx = mapping.unsqueeze(-1).expand_as(ext_force)  # (nf, nall, 3)
            force = torch.zeros(
                nframes, nloc, 3, dtype=ext_force.dtype, device=ext_force.device
            )
            force.scatter_add_(1, idx, ext_force)
            out["force"] = force
        if "extended_virial" in result:
            ext_virial = result["extended_virial"]  # (nf, nall, 9)
            idx = mapping.unsqueeze(-1).expand_as(ext_virial)  # (nf, nall, 9)
            atom_virial = torch.zeros(
                nframes,
                nloc,
                ext_virial.shape[-1],
                dtype=ext_virial.dtype,
                device=ext_virial.device,
            )
            atom_virial.scatter_add_(1, idx, ext_virial)
            out["atom_virial"] = atom_virial
        for key, val in result.items():
            if key not in ("extended_force", "extended_virial"):
                out[key] = val
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
        n_node: torch.Tensor | None = None,
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
            compact_nodes,
            expand_node_values,
        )
        from deepmd.pt_expt.utils.graph_builder import (
            build_neighbor_graph_for_method,
            build_ragged_neighbor_graph,
        )

        _model = self.original_model

        # A ragged batch already holds the node axis the lower works on; a
        # rectangular one is unravelled to it, and its per-atom outputs are
        # folded back at the end.
        ragged = n_node is not None
        n_padded = nframes * nloc
        # The builders take the shape the layout hands over: a flat node axis,
        # or the rectangular one a padded batch carries.
        coord_3d = (
            coord.detach().reshape(-1, 3)
            if ragged
            else coord.detach().reshape(nframes, nloc, 3)
        )
        box_flat = box.detach().reshape(nframes, 9) if box is not None else None
        # graph-lower ABI: aparam is FLAT on the node axis, (N, nda) -- like
        # every per-node tensor of the graph schema (the trace sample from
        # build_synthetic_graph_inputs is flat too, so the compiled lower's
        # input spec expects it). A ragged batch already carries it that way;
        # a rectangular one may fold the component axis into its frame rows,
        # so its node count is what unravels it.
        if aparam is not None and not ragged:
            aparam = aparam.reshape(n_padded, -1)

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
        method = getattr(_model, "neighbor_graph_method", "dense")
        if ragged:
            ng = build_ragged_neighbor_graph(
                method, coord_3d, atype, n_node, box_flat, rcut, pair_excl
            )
            atype_flat, node_index = atype, None
        else:
            ng = build_neighbor_graph_for_method(
                method, coord_3d, atype, box_flat, rcut, pair_excl
            )
            # A rectangular batch of unequal atom counts is padded to a common
            # width with phantom atoms (atype < 0). The builders leave them out
            # of every edge, so dropping them from the node axis costs nothing
            # and spares the network from evaluating them. On a batch of
            # uniform atom count this is a renumbering by the identity.
            atype_flat = atype.reshape(n_padded)
            ng, node_index = compact_nodes(ng, atype_flat >= 0)
            atype_flat = atype_flat[node_index]
            if aparam is not None:
                aparam = aparam[node_index]

        # Lazy compile of the GRAPH lower (cached per structure key).
        if self.compiled_forward_lower is None:
            compiled_lower, buf_order = self._compiled_lower_for(
                "neighbor-graph",
                lambda: _trace_and_compile_graph(
                    _model,
                    fparam,
                    aparam,
                    charge_spin,
                    task_buffers=self._task_buffers,
                    compile_opts=self._compile_opts,
                ),
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
            ng.n_node,
            ng.edge_index,
            edge_vec,
            ng.edge_mask,
            ng.destination_order,
            ng.destination_row_ptr,
            ng.source_order,
            ng.source_row_ptr,
            fparam,
            aparam,
            charge_spin,
            *task_buf_vals,
        )

        # The compiled graph lower emits PUBLIC keys on the FLAT node axis
        # (``atom_energy`` / ``force`` are (N, *); ``energy`` / ``virial`` are
        # (nf, *)). A ragged caller reads that axis directly. A rectangular one
        # has its node-level keys scattered back onto the padded width and
        # unravelled to (nf, nloc, *), where a phantom slot reads zero -- what
        # a masked-out atom contributed there before.
        if ragged:
            result["n_node"] = ng.n_node
            return result
        N = node_index.shape[0]
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
                out[key] = expand_node_values(val, node_index, n_padded).reshape(
                    nframes, nloc, *val.shape[1:]
                )
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
    stat_file_spec : StatFileSpec or dict or None
        Unopened statistics-cache configuration.
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
        stat_file_spec: StatFileSpec | Mapping[str, StatFileSpec] | None = None,
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
        optimizer_params = config.get("optimizer", {})
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
        self.stat_file_specs = stat_file_specs_by_task(
            stat_file_spec,
            self.model_keys,
        )

        # Distributed training detection
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.sharding = ShardingPolicy.from_training_params(
            training_params, is_distributed=self.is_distributed
        )

        # Iteration config
        self.disp_file = training_params.get("disp_file", "lcurve.out")
        self.disp_freq = training_params.get("disp_freq", 1000)
        self.save_ckpt = training_params.get("save_ckpt", "model.ckpt")
        self.save_freq = training_params.get("save_freq", 1000)
        self.enable_ema = bool(training_params.get("enable_ema", False))
        self.ema_decay = float(training_params.get("ema_decay", 0.999))
        self.ema_save_ckpt = get_ema_checkpoint_prefix(self.save_ckpt)
        self.display_in_training = training_params.get("disp_training", True)
        self.timing_in_training = training_params.get("time_training", True)
        self.change_bias_after_training = bool(
            training_params.get("change_bias_after_training", False)
        )
        self.enable_compile = bool(training_params.get("enable_compile", False))
        self._raise_if_sharding_unsupported()

        # Model ---------------------------------------------------------------
        self.models: dict[str, torch.nn.Module] = {}
        do_case_embd, case_embd_index = (
            _get_case_embd_config(model_params) if self.multi_task else (False, {})
        )
        # Descriptors sample the eval-time policy variables exactly once, while
        # they are being constructed; keep the config-derived defaults scoped to
        # construction so they do not leak into the rest of the process.
        with scoped_env_defaults(infer_env_defaults(validating_params)):
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

        # The layout settles before the schedule below, because under
        # ``mix:N`` it is what decides how many frames a batch holds, and the
        # schedule counts those batches to turn epochs into steps.
        self._configure_batch_layout(training_data, validation_data)

        # Statistics ----------------------------------------------------------
        self._finetune_update_stat = False
        self._sample_funcs: dict[str, Any] = {}
        for model_key in self.model_keys:
            _nbatch = self.model_params_by_task[model_key].get("data_stat_nbatch", 10)
            _data = self.training_data_by_task[model_key]

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
            if not resuming or _finetune_has_new_type:

                def initialize_statistics(
                    _model_key: str = model_key,
                    _sample_func: Callable[[], list[dict[str, np.ndarray]]] = (
                        _make_sample
                    ),
                ) -> None:
                    with open_stat_file(
                        self.stat_file_specs[_model_key]
                    ) as stat_file_path:
                        self.models[_model_key].compute_or_load_stat(
                            sampled_func=_sample_func,
                            stat_file_path=stat_file_path,
                        )

                self._run_stat_on_chief(
                    initialize_statistics,
                    operation=f"statistics initialization for task {model_key!r}",
                )

        # Training schedule ---------------------------------------------------
        schedule = resolve_step_schedule(
            training_params,
            multi_task=self.multi_task,
            model_keys=self.model_keys,
            training_data=self.training_data_by_task,
            epoch_length=self._epoch_length,
            broadcast=self._broadcast_value_from_rank0,
            rank=self.rank,
        )
        self.num_steps = schedule.num_steps
        self.model_prob = schedule.model_prob

        # Checkpoint layout ----------------------------------------------------
        # num_steps is final here, so a retention ratio can be converted into an
        # absolute keep count once.
        self.ckpt_store, self.ema_ckpt_store = build_checkpoint_stores(
            training_params,
            num_steps=self.num_steps,
            ema_prefix=self.ema_save_ckpt,
            rank=self.rank,
        )

        # Learning rate -------------------------------------------------------
        self.lr_schedule = make_learning_rate_schedule(
            config["learning_rate"], self.num_steps
        )

        # Gradient clipping
        self.gradient_max_norm = training_params.get("gradient_max_norm", 0.0)
        self.nonfinite_grad_guard = NonFiniteGradGuard()

        # Model wrapper -------------------------------------------------------
        self.wrapper = ModelWrapper(self.model, self.loss, model_params=model_params)
        self.start_step = 0

        # Shared params (multi-task) ------------------------------------------
        self._shared_links = shared_links
        synchronize_model_state = not resuming or self._finetune_update_stat
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
            share_kwargs = {
                "model_key_prob_map": dict(
                    zip(self.model_keys, self.model_prob, strict=True)
                ),
                "data_stat_protect": _data_stat_protect[0],
            }
            if synchronize_model_state:
                self._run_stat_on_chief(
                    lambda: self.wrapper.share_params(
                        shared_links,
                        resume=False,
                        **share_kwargs,
                    ),
                    operation="shared statistics merge",
                )

        if synchronize_model_state and self.is_distributed:
            for model_key in self.model_keys:
                self._broadcast_model_stat(self.models[model_key])

        if shared_links is not None:
            self.wrapper.share_params(
                shared_links,
                resume=True,
                **share_kwargs,
            )

        # Resume --------------------------------------------------------------
        # Weights are restored while the wrapper still owns whole tensors,
        # because a checkpoint records the model as a whole and its tensors
        # cannot be copied into sharded parameters.
        ema_state_dict = None
        optimizer_state_dict = None
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
                    # Optimizer and EMA state describe the weights of the run
                    # they were saved by; a finetune starts a new run and keeps
                    # neither.
                    continues_run = self.restart_training and finetune_model is None
                    optimizer_state_dict = (
                        state_dict["optimizer"] if continues_run else None
                    )
                    ema_state_dict = (
                        state_dict.get(EMA_CHECKPOINT_KEY) if continues_run else None
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

                    def update_finetune_bias(
                        _model_key: str = model_key,
                        _bias_mode: str = bias_mode,
                    ) -> None:
                        self.models[_model_key] = model_change_out_bias(
                            self.models[_model_key],
                            self._sample_funcs[_model_key],
                            _bias_adjust_mode=_bias_mode,
                        )

                    self._run_stat_on_chief(
                        update_finetune_bias,
                        operation=f"fine-tuning statistics for task {model_key!r}",
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

        # Distribution --------------------------------------------------------
        # The weights are in place, so a sharding strategy may cut them up.
        if self.is_distributed:
            self._distribute_wrapper()
            self._log_sharding_strategy()

        # Optimiser -----------------------------------------------------------
        opt_type = optimizer_params.get("type", "Adam")
        if opt_type not in ("Adam", "AdamW", "HybridMuon"):
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        # LambdaLR multiplies each param group's initial learning rate by the
        # lambda value.  Warmup schedules legitimately return zero at step 0,
        # so use the nonzero schedule base as the denominator and let the
        # lambda initialize the optimizer to the requested warmup value.
        initial_lr = float(self.lr_schedule.start_lr)
        adam_betas = (
            float(optimizer_params["adam_beta1"]),
            float(optimizer_params["adam_beta2"]),
        )
        weight_decay = float(optimizer_params["weight_decay"])

        if opt_type in ("Adam", "AdamW"):
            self.optimizer = self._create_optimizer(
                torch.optim.Adam if opt_type == "Adam" else torch.optim.AdamW,
                lr=initial_lr,
                betas=adam_betas,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = self._create_optimizer(
                HybridMuonOptimizer,
                lr=initial_lr,
                momentum=float(optimizer_params["momentum"]),
                weight_decay=weight_decay,
                adam_betas=adam_betas,
                lr_adjust=float(optimizer_params["lr_adjust"]),
                lr_adjust_coeff=float(optimizer_params["lr_adjust_coeff"]),
                muon_mode=str(optimizer_params["muon_mode"]),
                enable_gram=bool(optimizer_params["enable_gram"]),
                flash_muon=bool(optimizer_params["flash_muon"]),
                magma_muon=bool(optimizer_params["magma_muon"]),
                # Sharded parameters are DTensors, and several torch._foreach_*
                # ops lack sharding propagation, so the per-tensor path applies.
                use_foreach=False if self.sharding.shards_parameters else None,
            )
            # The parameter names route each tensor to Muon or Adam. They are
            # supplied after construction because a redundancy-sharded
            # optimizer treats every constructor keyword as a param-group
            # default, which would serialize the whole model into each
            # checkpoint.
            self._local_optimizer.set_param_names(
                tuple(self.wrapper.named_parameters())
            )

        if optimizer_state_dict is not None:
            self._load_optimizer_state(optimizer_state_dict)
        for param_group in self.optimizer.param_groups:
            param_group["initial_lr"] = initial_lr

        # The resumed step offset is carried by last_epoch; the lambda must not
        # add it again, which would advance the schedule twice.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: self.lr_schedule.value(step) / initial_lr,
            last_epoch=self.start_step - 1,
        )

        # Exponential moving average -------------------------------------------
        # The shadow tracks the raw models, whose parameter tensors the compiled
        # graphs keep sharing, so it is unaffected by compilation below.
        self.model_ema = (
            ModelEMA(self.model, decay=self.ema_decay, state=ema_state_dict)
            if self.enable_ema
            else None
        )

        self._configure_neighbor_graph_method(
            training_params.get("neighbor_graph_method", "auto")
        )

        # torch.compile -------------------------------------------------------
        if self.enable_compile:
            check_compile_torch_version()
            compile_opts = training_params.get("compile_options", {})
            if compile_opts:
                log.info("torch.compile options: %s", compile_opts)
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
        self.full_validator, self.ema_full_validator = self._create_full_validators(
            validating_params=validating_params,
            validation_data=self.validation_data if not self.multi_task else None,
        )

        if self.rank == 0:
            log_parameter_counts(
                {key: count_parameters(self.models[key]) for key in self.model_keys},
                multi_task=self.multi_task,
            )

    def _create_full_validators(
        self,
        *,
        validating_params: dict[str, Any],
        validation_data: Any | None,
    ) -> tuple[FullValidator | None, FullValidator | None]:
        """Create the live-weight and EMA-weight full validators."""
        return build_full_validators(
            validating_params=validating_params,
            validation_data=validation_data,
            model=self.model,
            state_store=self._unwrapped.train_infos,
            num_steps=self.num_steps,
            rank=self.rank,
            restart_training=self.restart_training,
            checkpoint_dir=resolve_best_checkpoint_dir(
                validating_params, self.save_ckpt
            ),
            ensure_supported=lambda: self._raise_if_full_validation_unsupported(
                validation_data
            ),
            model_ema=self.model_ema,
            sharding=self.sharding,
        )

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

        if self.sharding.shards_parameters:
            raise ValueError(
                "validating.full_validation only supports single-task energy "
                "training with training.zero_stage < 2."
            )

        if self.models[DEFAULT_TASK_KEY].has_spin() or isinstance(
            self.loss, EnergySpinLoss
        ):
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

    def _configure_batch_layout(self, *data_maps: Any) -> None:
        """Ask each LMDB data system for the layout its own model can consume.

        Under ``mix:N``, a model whose descriptor reads a flat node axis takes
        the frames of a batch concatenated, which spares it the padding that
        frames of unequal atom count would otherwise need. Every other model
        reads an ``(nf, nloc, ...)`` axis and needs them padded to a common
        width. Non-mixing batch rules retain their established rectangular
        layout. Only the trainer sees both sides, and it settles the question
        here, once, before any batch is drawn.

        A graph lower is necessary but not sufficient: the model must expose an
        entry that takes the flat axis, and the configured loss must accept that
        representation. Composed models (linear, ZBL bridging) have no such
        model entry. Generalized-force and Hessian losses require rectangular
        label axes. Native-spin models also stay rectangular because their
        public output translation needs the spin-specific force and mask.
        Requiring both capabilities keeps the complete objective on a layout it
        can evaluate.

        Each task has its own data system and its own model, so the answer is
        each task's own: a multi-task run pairing a graph model with a dense
        one gives the first concatenated batches and the second padded ones.

        This must run before the run length is resolved. Under ``mix:N`` the
        layout decides where the sampler cuts a batch -- padding is what makes
        a batch's cost depend on its widest frame -- so a schedule that
        counted batches first would count a packing training never uses, and
        would serve its first epoch in that packing.

        Parameters
        ----------
        *data_maps : Any
            The training and validation data systems, either bare or as the
            per-task mappings a multi-task run builds.
        """
        for task_key in self.model_keys:
            model = self.models[task_key]
            loss = self.losses[task_key]
            ragged = (
                not model.has_spin()
                and model_uses_graph_lower(model)
                and hasattr(model, "forward_ragged")
                and loss.supports_ragged_batches
            )
            for data_map in data_maps:
                data = (
                    data_map.get(task_key) if isinstance(data_map, dict) else data_map
                )
                if hasattr(data, "use_ragged_batches"):
                    data.use_ragged_batches(ragged)

    def _configure_neighbor_graph_method(self, requested: str) -> None:
        """Resolve and install the training graph builder on eligible models."""
        graph_models = [
            self.models[model_key]
            for model_key in self.model_keys
            if model_uses_graph_lower(self.models[model_key])
        ]
        if not graph_models:
            if requested != "auto":
                raise ValueError(
                    "training.neighbor_graph_method applies only to "
                    "graph-eligible energy models"
                )
            return

        from deepmd.pt_expt.utils.graph_builder import (
            resolve_neighbor_graph_method,
        )

        resolved = resolve_neighbor_graph_method(requested, DEVICE)
        for model in graph_models:
            model.neighbor_graph_method = resolved

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

            _warn_compiled_attention(model, task_key)

            structure_key = _key_for[task_key]
            task_bufs = _task_bufs_for[task_key]

            wrapper_mod.model[task_key] = _CompiledModel(
                model,
                structure_key=structure_key,
                task_buf_order=tuple(task_bufs.keys()) if task_bufs else (),
                task_buffers=task_bufs if task_bufs else None,
                compile_opts=compile_opts,
                compiled_by_structure=_compiled_by_structure,
                task_key=task_key,
            )
            log.info(
                "Compilation enabled (task=%s); the graph is traced and compiled "
                "on the first training step.",
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

    def _epoch_length(self, model_key: str) -> int:
        """Return the steps this rank takes during one epoch of a task.

        Parameters
        ----------
        model_key : str
            Key of the task whose training data is measured.

        Returns
        -------
        int
            Number of steps covering one pass over the task's training data.

        Notes
        -----
        A data system reports ``nbatches[i]``, the batch count of system ``i``,
        and ``sys_probs[i]``, the probability of drawing from that system, from
        which ``compute_total_numb_batch`` derives the dataset-wide epoch
        length ``ceil(max_i(nbatches[i] / sys_probs[i]))``. LMDB data reports
        that global count while its sampler shards batches evenly across ranks;
        legacy data systems remain replicated. In both cases one rank takes
        ``ceil(total / world_size)`` steps per epoch.
        """
        data = self.training_data_by_task[model_key]
        total = compute_total_numb_batch(data.nbatches, data.sys_probs)
        return int(np.ceil(total / self.world_size))

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    def _raise_if_sharding_unsupported(self) -> None:
        """Reject the run configurations that state sharding cannot serve.

        Raises
        ------
        ValueError
            If the requested stage conflicts with another training option.
        """
        if not self.sharding.enabled:
            return
        if self.multi_task:
            raise ValueError(
                "training.zero_stage is currently only supported in single-task "
                "training."
            )
        if self.change_bias_after_training:
            raise ValueError(
                "training.zero_stage does not support change_bias_after_training."
            )
        if not self.sharding.shards_parameters:
            return
        if self.enable_ema:
            raise ValueError(
                "training.enable_ema currently only supports training.zero_stage < 2."
            )
        if self.enable_compile:
            raise ValueError(
                "training.enable_compile only supports training.zero_stage < 2: "
                "the compiled graph is traced from the parameters, which FSDP2 "
                "shards as DTensors."
            )

    def _distribute_wrapper(self) -> None:
        """Place the wrapper under the parallel strategy of this run.

        Stages below two replicate the model and keep plain tensors, so the
        wrapper is held by ``DistributedDataParallel``. From stage two on the
        parameters themselves are sharded by FSDP2, which mutates the wrapper
        in place and therefore leaves no module to unwrap.
        """
        local_rank = None
        if DEVICE.type == "cuda":
            from deepmd.pt_expt.utils.env import (
                LOCAL_RANK,
            )

            local_rank = LOCAL_RANK
            torch.cuda.set_device(local_rank)

        if not self.sharding.shards_parameters:
            # Multi-task uses only one fitting_net per step, so unused
            # parameters exist in the graph. Single-task doesn't need this.
            kwargs: dict[str, Any] = {"find_unused_parameters": self.multi_task}
            if local_rank is not None:
                kwargs |= {"device_ids": [local_rank], "output_device": local_rank}
            self.wrapper = torch.nn.parallel.DistributedDataParallel(
                self.wrapper, **kwargs
            )
            return

        if fully_shard is None:
            raise RuntimeError(
                "training.zero_stage>=2 requires FSDP2 "
                "(``torch.distributed.fsdp.fully_shard``), which is missing "
                f"from PyTorch {torch.__version__}. Set training.zero_stage "
                "to 0 or 1 to stay on the DDP / ZeRO-1 path."
            )
        # Unlike the DDP constructor, FSDP2 does not broadcast: the ranks have
        # to already agree on the weights before they are cut into shards.
        for tensor in (*self.wrapper.parameters(), *self.wrapper.buffers()):
            dist.broadcast(tensor.data, src=0)
        self.wrapper = fully_shard(
            self.wrapper,
            reshard_after_forward=self.sharding.reshards_after_forward,
        )

    def _log_sharding_strategy(self) -> None:
        """Report the distribution strategy once the wrapper is in place."""
        if self.sharding.enabled and self.rank == 0:
            log.info(self.sharding.describe())

    def _load_optimizer_state(self, optimizer_state_dict: dict[str, Any]) -> None:
        """Restore optimizer state recorded as one whole.

        Unlike the weights, the optimizer is necessarily built after the model
        is distributed, so under parameter sharding its state is already made
        of shards and the recorded state has to be cut up to match.

        Parameters
        ----------
        optimizer_state_dict : dict[str, Any]
            The optimizer state as recorded in a checkpoint.
        """
        if not self.sharding.shards_parameters:
            self.optimizer.load_state_dict(optimizer_state_dict)
            return
        set_optimizer_state_dict(
            self.wrapper,
            self.optimizer,
            optim_state_dict=optimizer_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

    def _create_optimizer(
        self,
        optimizer_class: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """Construct the optimizer, sharding its state when the stage asks.

        Parameters
        ----------
        optimizer_class : type[torch.optim.Optimizer]
            The optimizer to construct.
        **kwargs
            Keyword arguments forwarded to the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer, wrapped in a ``ZeroRedundancyOptimizer`` when the
            stage shards optimizer state but not the parameters; from stage
            two on FSDP2 already shards the state that the optimizer derives
            from its sharded parameters.
        """
        if self.sharding.stage == 1:
            return ZeroRedundancyOptimizer(
                self.wrapper.parameters(),
                optimizer_class=optimizer_class,
                **kwargs,
            )
        return optimizer_class(self.wrapper.parameters(), **kwargs)

    @property
    def _local_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer that performs this rank's update.

        A redundancy-sharded optimizer owns no update of its own: it delegates
        to a local optimizer over this rank's share of the parameters, and it
        is that one which holds the per-parameter state and the name routing.
        """
        if self.sharding.stage == 1:
            return self.optimizer.optim
        return self.optimizer

    @property
    def _unwrapped(self) -> "ModelWrapper":
        """Return the raw ModelWrapper, unwrapping DDP if active."""
        if hasattr(self.wrapper, "module"):
            return self.wrapper.module
        return self.wrapper

    def _run_stat_on_chief(
        self,
        action: Callable[[], None],
        *,
        operation: str,
    ) -> None:
        """Run a statistics action on rank 0 and synchronize its outcome."""
        synchronize_failure: Callable[[bool], bool] | None = None
        if self.is_distributed:

            def broadcast_failure(failed: bool) -> bool:
                holder = [failed if self.rank == 0 else False]
                dist.broadcast_object_list(holder, src=0, device=DEVICE)
                return bool(holder[0])

            synchronize_failure = broadcast_failure

        run_stat_on_chief(
            action,
            is_chief=self.rank == 0,
            synchronize_failure=synchronize_failure,
            operation=operation,
        )

    @staticmethod
    def _broadcast_model_stat(model: torch.nn.Module) -> None:
        """Broadcast model parameters and buffers from rank 0 to all ranks."""
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        for b in model.buffers():
            dist.broadcast(b, src=0)

    def _broadcast_value_from_rank0(self, value: Any) -> Any:
        """Return rank 0's copy of ``value`` on every rank.

        Epoch lengths round a quotient of sampling probabilities that is often
        an exact integer in real arithmetic, so a last-bit difference between
        ranks -- as reduction kernels dispatched for different CPU features
        produce -- flips the rounded result and hence ``num_steps``. Ranks that
        disagree on ``num_steps`` also disagree on the full-validation start
        step and deadlock on mismatched collective calls, so the whole world
        adopts rank 0's value.
        """
        if not self.is_distributed:
            return value
        holder = [value]
        dist.broadcast_object_list(holder, src=0, device=DEVICE)
        return holder[0]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @property
    def checkpoint_is_collective(self) -> bool:
        """Whether assembling a checkpoint needs every rank."""
        return self.sharding.enabled

    def save_checkpoint(self, step: int) -> None:
        # Abort before writing if any gradient norm since the previous
        # checkpoint was non-finite, so a diverged interval is not persisted.
        self.nonfinite_grad_guard.raise_if_nonfinite(self.wrapper.named_parameters)
        ckpt_path = self.ckpt_store.path_for(step)
        self._save_checkpoint_to_path(ckpt_path, step=step)
        if self.rank == 0:
            self.ckpt_store.publish(ckpt_path)
            self.ckpt_store.prune(ckpt_path)
            log.info(f"Saved model to {ckpt_path}")
        if self.model_ema is not None:
            ema_path = self.ema_ckpt_store.path_for(step)
            self._save_checkpoint_to_path(ema_path, step=step, use_ema_weights=True)
            if self.rank == 0:
                self.ema_ckpt_store.publish(ema_path)
                self.ema_ckpt_store.prune(ema_path)

    def _save_full_validation_checkpoint(
        self,
        save_path: Path,
        lr: float = 0.0,
        step: int = 0,
    ) -> None:
        """Save a checkpoint selected by full validation."""
        del lr
        self._save_checkpoint_to_path(save_path, step=step)

    def _save_full_validation_ema_checkpoint(
        self,
        save_path: Path,
        lr: float = 0.0,
        step: int = 0,
    ) -> None:
        """Save an EMA-weight checkpoint selected by EMA full validation.

        The validator restores the live weights before selecting a checkpoint,
        so the shadow has to be applied again while writing it.
        """
        del lr
        self._save_checkpoint_to_path(save_path, step=step, use_ema_weights=True)

    def _save_checkpoint_to_path(
        self,
        ckpt_path: Path,
        *,
        step: int,
        use_ema_weights: bool = False,
    ) -> None:
        """Serialize the current trainer state to an explicit checkpoint path.

        Parameters
        ----------
        ckpt_path : Path
            Destination of the checkpoint file.
        step : int
            Training step recorded in the checkpoint.
        use_ema_weights : bool, optional
            Whether to substitute the EMA-smoothed weights for the live ones.
            Such a checkpoint is a deployment snapshot: it carries neither the
            optimizer state nor the EMA state, both of which describe the live
            weights it does not contain.
        """
        if use_ema_weights:
            with self.model_ema.apply_shadow(self.model):
                self._write_checkpoint(
                    ckpt_path,
                    step=step,
                    include_optimizer=False,
                    include_ema_state=False,
                )
            return
        self._write_checkpoint(ckpt_path, step=step)

    def _write_checkpoint(
        self,
        ckpt_path: Path,
        *,
        step: int,
        include_optimizer: bool = True,
        include_ema_state: bool = True,
    ) -> None:
        """Serialize the wrapper, and optionally optimizer and EMA state."""
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
            model_state, optim_state = self._collect_checkpoint_states(
                wrapper, include_optimizer=include_optimizer
            )
        finally:
            for task_key, compiled in compiled_backup.items():
                wrapper.model[task_key] = compiled
        # Sharded state is assembled on the chief; the other ranks have played
        # their part in the collectives above and hold nothing to write.
        if self.rank != 0:
            return
        state: dict[str, Any] = {"model": model_state}
        if optim_state is not None:
            state["optimizer"] = optim_state
        if include_ema_state and self.model_ema is not None:
            state[EMA_CHECKPOINT_KEY] = self.model_ema.state_dict()
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_path)

    def _collect_checkpoint_states(
        self,
        wrapper: "ModelWrapper",
        *,
        include_optimizer: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Gather the model and optimizer state a checkpoint records.

        Parameters
        ----------
        wrapper : ModelWrapper
            The unwrapped model wrapper, already stripped of compiled models.
            Under parameter sharding it is the sharded wrapper itself, because
            FSDP2 shards in place and compilation is rejected alongside it.
        include_optimizer : bool
            Whether the optimizer state belongs in the checkpoint.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any] | None]
            The model state and the optimizer state. Under sharding both are
            complete only on the chief; the other ranks contribute their shards
            and receive placeholders.
        """
        if self.sharding.shards_parameters:
            # FSDP2 reassembles the shards, so every rank has to take part.
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            return (
                get_model_state_dict(wrapper, options=options),
                get_optimizer_state_dict(wrapper, self.optimizer, options=options)
                if include_optimizer
                else None,
            )
        model_state = wrapper.state_dict()
        if not include_optimizer:
            return model_state, None
        if self.sharding.shards_optimizer_state:
            self.optimizer.consolidate_state_dict(to=0)
            return model_state, self.optimizer.state_dict() if self.rank == 0 else {}
        return model_state, self.optimizer.state_dict()

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
        try:
            super().run(self.training_tasks)
            if self.change_bias_after_training and self.num_steps > self.start_step:
                self._change_bias_after_training()
                if self.rank_context.is_chief:
                    self.save_checkpoint(self.num_steps)
        finally:
            self._close_data_systems()
        if self.rank_context.is_chief:
            log.info(f"Trained model has been saved to: {self.save_ckpt}")

    def _close_data_systems(self) -> None:
        """Release asynchronous data pipelines owned by this trainer."""
        closed: set[int] = set()
        for data_by_task in (
            self.training_data_by_task,
            self.validation_data_by_task,
        ):
            for data_system in data_by_task.values():
                if data_system is None or id(data_system) in closed:
                    continue
                closed.add(id(data_system))
                close = getattr(data_system, "close", None)
                if close is not None:
                    close()

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
        """Run the active full validation flows for one step."""
        for validator, save_checkpoint in (
            (self.full_validator, self._save_full_validation_checkpoint),
            (self.ema_full_validator, self._save_full_validation_ema_checkpoint),
        ):
            if validator is None:
                continue
            validator.run(
                step_id=display_step,
                display_step=display_step,
                lr=learning_rate,
                save_checkpoint=save_checkpoint,
            )

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
            self.nonfinite_grad_guard.update(
                clip_grad_norm_(
                    self.wrapper.parameters(),
                    self.gradient_max_norm,
                    # A sharded gradient is a DTensor: the overflow-safe
                    # reduction would measure this rank's shard alone, so the
                    # distributed-native norm applies instead.
                    stable=not self.sharding.shards_parameters,
                )
            )

        self._optimizer_step()
        if self.model_ema is not None:
            self.model_ema.update(self.model)
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
        """Evaluate validation loss terms for one task.

        Sharded parameters are gathered by the forward itself, which makes it
        a collective operation, while the display runs on the chief alone.
        Validation is therefore skipped from stage two on; the metrics remain
        available through the independent full validation flow, which every
        rank enters together.
        """
        if task.validation_data is None or self.sharding.shards_parameters:
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
            # The metrics are per-atom quantities, so each batch weighs by the
            # real atoms it holds summed over its frames. Phantom atoms
            # (atype < 0), which pad a mixed-nloc batch, contribute to none.
            natoms = int((val_input["atype"] >= 0).sum())
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
