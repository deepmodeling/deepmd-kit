# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 / SeZM → AOTInductor ``.pt2`` freeze path for the pt backend.

SeZM relies on a nested ``autograd.grad(create_graph=True)`` inside
``fit_output_to_model_output``; TorchScript cannot represent that
graph, so DPA4 / SeZM checkpoints are routed through AOTInductor instead.
The output archive layout follows the ``pt_expt`` convention, including the
metadata consumed by ``DeepPotPTExpt.cc`` and ``DeepSpinPTExpt.cc``.

Tracing runs on CPU (``make_fx`` with ``_allow_non_fake_inputs=True``
is brittle on CUDA because the proxy-tensor dispatcher does not set
up CUDA streams for the captured parameters).  The compiled package
is moved to the target device via ``move_to_device_pass`` before
``aoti_compile_and_package``.

``.pt2`` I/O is always float64, matching the C++ contract in
``DeepPotPTExpt::compute`` where LAMMPS coordinates are unconditionally
cast to ``torch::kFloat64``.  SeZM's own ``_input_type_cast`` bridges
fp64 inputs to whatever internal compute dtype the checkpoint uses.
"""

from __future__ import (
    annotations,
)

import ctypes
import json
import logging
import os
import tempfile
import zipfile
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.kernels.utils import (
    triton_infer_level,
)
from deepmd.pt.model.descriptor.sezm_nn.so2 import (
    SO2Convolution,
    SO2Linear,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.compile_compat import (
    build_inductor_compile_options,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.edge_schema import (
    edge_schema_from_extended,
)
from deepmd.utils.model_branch_dict import (
    get_model_dict,
)

log = logging.getLogger(__name__)


def _model_has_spin(model: torch.nn.Module) -> bool:
    """Return whether ``model`` uses the spin lower interface."""
    has_spin = getattr(model, "has_spin", False)
    return bool(has_spin() if callable(has_spin) else has_spin)


def _get_model_ntypes(model: torch.nn.Module) -> int:
    """Return atom type count even when the exported type map is empty."""
    type_map = list(model.get_type_map())
    if type_map:
        return len(type_map)
    descriptor = model.get_descriptor()
    return int(descriptor.get_ntypes())


def _model_has_message_passing(model: torch.nn.Module) -> bool:
    """Return whether the regular .pt2 graph requires a real atom mapping."""
    for obj in (
        model,
        getattr(model, "atomic_model", None),
        model.get_descriptor() if hasattr(model, "get_descriptor") else None,
    ):
        if obj is None or not hasattr(obj, "has_message_passing"):
            continue
        try:
            return bool(obj.has_message_passing())
        except (AttributeError, NotImplementedError):
            continue
    return False


def _strip_shape_assertions(graph_module: torch.nn.Module) -> None:
    """Remove deferred shape assertions from SeZM export graphs.

    SeZM lower inputs intentionally keep extended-atom and local-atom axes
    independent: regular exports pass ghost coordinates through ``coord`` while
    ``atype`` remains local-only, and spin exports slice both ``nall`` and
    ``nloc`` after virtual atom expansion. ``torch.export`` may turn these valid
    dynamic cases into deferred ``Ne(nall, nloc)`` assertions.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        ):
            graph.erase_node(node)
    graph.eliminate_dead_code()
    graph_module.recompile()


def _extract_state_and_params(
    ckpt: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unwrap a ``torch.load`` result into ``(state_dict, model_params)``.

    Accepts both the training-wrapper layout (weights under a top-level
    ``"model"`` key) and a bare state dict.
    """
    inner = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(inner, dict):
        raise ValueError("Unsupported checkpoint: expected a dict-like state dict.")
    extra = inner.get("_extra_state") or {}
    params = extra.get("model_params")
    if not isinstance(params, dict):
        raise ValueError("Unsupported checkpoint: missing '_extra_state.model_params'.")
    return inner, params


def is_sezm_checkpoint(ckpt_path: str) -> bool:
    """Best-effort detection used by the CLI to route DPA4 / SeZM checkpoints.

    Returns ``False`` for unreadable files or non-SeZM checkpoints; no
    exception leaks out so the caller can treat this as a pure routing
    signal.
    """
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    try:
        _, params = _extract_state_and_params(raw)
    except ValueError:
        return False
    if "model_dict" in params:
        return any(
            str(branch_params.get("type", "")).lower() in ("sezm", "dpa4")
            for branch_params in params["model_dict"].values()
        )
    return str(params.get("type", "")).lower() in ("sezm", "dpa4")


def _select_model_head(
    state_dict: dict[str, Any],
    params: dict[str, Any],
    head: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract a single selected model branch from a checkpoint."""
    if "model_dict" not in params:
        if head is not None:
            raise NotImplementedError(
                "SeZM .pt2 freeze does not yet support head selection for single-task checkpoints; pass head=None."
            )
        return state_dict, params

    model_alias_dict, _ = get_model_dict(params["model_dict"])
    model_keys = list(params["model_dict"])
    if head is None and "Default" in model_alias_dict:
        head = "Default"
        log.info(
            "Using default head %s for multitask SeZM freeze.", model_alias_dict[head]
        )
    if head is None:
        raise ValueError(
            "Head must be set for multitask SeZM/DPA4 freeze. "
            f"Available heads are: {model_keys}."
        )
    if head not in model_alias_dict:
        head_lower = head.lower()
        for key in model_alias_dict:
            if key.lower() == head_lower:
                head = key
                break
    if head not in model_alias_dict:
        raise ValueError(
            f"No head or alias named {head!r} in model. Available heads are: {model_keys}."
        )

    branch = model_alias_dict[head]
    branch_params = deepcopy(params["model_dict"][branch])
    branch_state: dict[str, Any] = {
        "_extra_state": deepcopy(state_dict.get("_extra_state", {})),
    }
    branch_state["_extra_state"]["model_params"] = branch_params
    prefix = f"model.{branch}."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            branch_state[key.replace(prefix, "model.Default.")] = value
    return branch_state, branch_params


def _to_py_list(value: Any) -> Any:
    """Coerce torch / numpy scalars into JSON-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(f"Cannot JSON-serialize value of type {type(value)!r}")


def _collect_metadata(
    model: torch.nn.Module,
    output_keys: list[str],
    is_spin: bool | None = None,
    do_atomic_virial: bool = False,
    has_comm_artifact: bool = False,
) -> dict:
    """Assemble the flat metadata dict expected by :class:`DeepPotPTExpt`.

    Mirrors the reader contract at ``source/api_cc/src/DeepPotPTExpt.cc`` and
    the metadata-only load path in ``deepmd.pt_expt.infer.deep_eval.DeepEval``:
    every field consumed by C++ LAMMPS inference **and** every field
    consumed by ``DeepEval._init_from_metadata`` must be present here.

    ``output_keys`` is the insertion order that the loader zips with
    ``AOTIModelPackageLoader::run``'s flat output vector.
    """
    if is_spin is None:
        is_spin = _model_has_spin(model)
    fitting_output_defs: list[dict[str, Any]] = []
    for vdef in model.atomic_output_def().get_data().values():
        fitting_output_defs.append(
            {
                "name": vdef.name,
                "shape": list(vdef.shape),
                "reducible": vdef.reducible,
                "r_differentiable": vdef.r_differentiable,
                "c_differentiable": vdef.c_differentiable,
                "atomic": vdef.atomic,
                # OutputVariableCategory is an IntEnum; force plain int for
                # deterministic JSON serialisation across Python versions.
                "category": int(vdef.category),
                "r_hessian": vdef.r_hessian,
                "magnetic": bool(vdef.magnetic or (is_spin and vdef.name == "energy")),
                "intensive": vdef.intensive,
            }
        )
    exports_atomic_virial = True if not is_spin else bool(do_atomic_virial)
    metadata = {
        "type_map": list(model.get_type_map()),
        "ntypes": _get_model_ntypes(model),
        "rcut": float(model.get_rcut()),
        "sel": [int(s) for s in model.get_sel()],
        "lower_input_kind": model.export_lower_input_kind(),
        "dim_fparam": int(model.get_dim_fparam()),
        "dim_aparam": int(model.get_dim_aparam()),
        "dim_chg_spin": int(model.get_dim_chg_spin()),
        "mixed_types": bool(model.mixed_types()),
        "has_message_passing": _model_has_message_passing(model),
        "has_comm_artifact": bool(has_comm_artifact),
        "do_atomic_virial": exports_atomic_virial,
        "nnei": int(sum(model.get_sel())),
        "has_default_fparam": bool(model.has_default_fparam()),
        "default_fparam": _to_py_list(model.get_default_fparam()),
        "default_chg_spin": _to_py_list(model.get_default_chg_spin()),
        "output_keys": list(output_keys),
        "fitting_output_defs": fitting_output_defs,
        # sel_type feeds DeepEval.get_sel_type() in metadata-only mode.
        # SeZM energy models return [] (every type selected).
        "sel_type": [int(t) for t in model.get_sel_type()],
        "is_spin": bool(is_spin),
    }
    if is_spin:
        metadata["ntypes_spin"] = int(model.spin.get_ntypes_spin())
        metadata["use_spin"] = [bool(v) for v in model.spin.use_spin]
    return metadata


def _tune_triton_configs(model: torch.nn.Module, target_device: torch.device) -> None:
    """Tune the shape-keyed Triton launch tables for this checkpoint's shapes.

    At ``DP_TRITON_INFER >= 2`` the traced graph bakes launch configurations
    resolved from the tables in ``deepmd.kernels.triton.sezm.tile_configs``.  Shape keys
    absent from the built-in tables (an untuned GPU model, or an untuned
    width/degree) are swept here on the local GPU -- the exact hardware the
    ``.pt2`` will run on, since AOTInductor artifacts are not portable across
    GPU models -- and registered for the current process before tracing.
    Keys already covered cost nothing.

    The fused value-path entries are then rebound: the mixing-stack operator
    selection (fp32 versus fp16x3) is fixed at construction time, which
    predates the registrations made here.
    """
    if triton_infer_level() < 2:
        return
    if target_device.type != "cuda" or not torch.cuda.is_available():
        return
    from deepmd.kernels.triton.sezm.so2_value_path import (
        SO2_VALUE_PATH_TRITON_AVAILABLE,
        make_triton_value_path,
    )

    if not SO2_VALUE_PATH_TRITON_AVAILABLE:
        return
    from deepmd.kernels.triton.sezm.sweep_tile_configs import (
        collect_model_shape_keys,
        tune_missing_configs,
    )
    from deepmd.kernels.triton.sezm.tile_configs import (
        _builtin_tables,
    )

    # The built-in tables and the sweep both resolve against the current
    # device; pin it to the AOTI target so a freeze aimed at a secondary GPU
    # tunes and looks up the right hardware (mixed-model hosts).
    if target_device.index is not None:
        torch.cuda.set_device(target_device)
        _builtin_tables.cache_clear()

    shape_keys = collect_model_shape_keys(model)
    registered = tune_missing_configs(
        shape_keys, level=triton_infer_level(), device=target_device
    )
    if registered:
        log.info(
            "Registered freshly tuned Triton launch configurations: %s",
            {family: sorted(entries) for family, entries in registered.items()},
        )
    else:
        log.info(
            "Triton launch tables already cover this checkpoint's shapes on %s; "
            "no tuning needed.",
            torch.cuda.get_device_name(target_device),
        )
    # Rebind unconditionally: the fp32-versus-fp16x3 stack selection was made
    # at construction time, possibly against a different current device's
    # tables, and must reflect the target device and any fresh registrations.
    for module in model.modules():
        if isinstance(module, SO2Convolution) and module.triton_infer_level >= 2:
            module._triton_value_path = make_triton_value_path(module)


# The trace-time sendlist for the with-comm artifact embeds the address of a
# numpy array (``int**`` contract of ``border_op``). The array must outlive the
# trace + export call; the exported graph never reads it at runtime (the op is
# opaque), so a module-level keepalive is sufficient.
_TRACE_SENDLIST_KEEPALIVE: list[np.ndarray] = []


def _build_sample_extended(
    model: torch.nn.Module,
    nframes: int,
    nloc: int,
    device: torch.device,
    has_spin: bool,
) -> tuple[torch.Tensor | None, ...]:
    """Build the extended-region sample tensors shared by the lower builders.

    Returns ``(ext_coord, ext_atype, nlist, mapping, ext_spin, fparam, aparam,
    charge_spin)``; tensors are float64 / int64 (matching the ``.pt2`` I/O
    contract). ``ext_spin`` is ``None`` unless ``has_spin``.
    """
    rcut = float(model.get_rcut())
    sel = list(model.get_sel())
    ntypes = len(model.get_type_map())
    if ntypes == 0:
        ntypes = int(model.get_descriptor().get_ntypes())
    if ntypes <= 0:
        raise ValueError("SeZM .pt2 freeze requires at least one atom type.")
    dim_fparam = int(model.get_dim_fparam())
    dim_aparam = int(model.get_dim_aparam())
    dim_chg_spin = int(model.get_dim_chg_spin())
    mixed_types = bool(model.mixed_types())

    box_size = rcut * 3.0
    box = np.eye(3, dtype=np.float64) * box_size
    box_np = box.reshape(1, 9)

    rng = np.random.default_rng(42)
    coord_np = rng.random((nframes, nloc, 3), dtype=np.float64) * box_size * 0.5
    coord_np += box_size * 0.25  # centre roughly in the middle of the cell

    atype_np = np.zeros((nframes, nloc), dtype=np.int32)
    for i in range(nloc):
        atype_np[:, i] = i % ntypes
    spin_np = np.zeros((nframes, nloc, 3), dtype=np.float64)
    if has_spin:
        atom_idx = np.arange(nloc, dtype=np.float64).reshape(1, nloc)
        spin_np[:, :, 0] = 0.10 + 0.01 * atom_idx
        spin_np[:, :, 1] = 0.20 + 0.02 * atom_idx
        spin_np[:, :, 2] = 0.05

    coord_normalized = normalize_coord(
        coord_np.reshape(nframes, nloc, 3),
        np.tile(box.reshape(1, 3, 3), (nframes, 1, 1)),
    )
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype_np, np.tile(box_np, (nframes, 1)), rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=not mixed_types,
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)

    ext_coord = torch.tensor(extended_coord, dtype=torch.float64, device=device)
    ext_atype = torch.tensor(extended_atype, dtype=torch.int64, device=device)
    nlist_t = torch.tensor(nlist, dtype=torch.int64, device=device)
    mapping_t = torch.tensor(mapping, dtype=torch.int64, device=device)
    ext_spin = None
    if has_spin:
        extended_spin = np.take_along_axis(spin_np, mapping[..., None], axis=1)
        ext_spin = torch.tensor(extended_spin, dtype=torch.float64, device=device)
    fparam = (
        torch.zeros(nframes, dim_fparam, dtype=torch.float64, device=device)
        if dim_fparam > 0
        else None
    )
    aparam = (
        torch.zeros(nframes, nloc, dim_aparam, dtype=torch.float64, device=device)
        if dim_aparam > 0
        else None
    )
    charge_spin = (
        torch.zeros(nframes, dim_chg_spin, dtype=torch.float64, device=device)
        if dim_chg_spin > 0
        else None
    )
    return (
        ext_coord,
        ext_atype,
        nlist_t,
        mapping_t,
        ext_spin,
        fparam,
        aparam,
        charge_spin,
    )


def _make_sample_inputs(
    model: torch.nn.Module,
    nframes: int,
    nloc: int,
    device: torch.device,
    has_spin: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Build representative ``forward_common_lower`` inputs for tracing.

    Three lower ABIs are produced, selected by ``model.export_lower_input_kind()``
    and whether the model carries spin:

    - virtual spin (``nlist``): the DeepSpin extended-input signature, since the
      graph expands virtual atoms internally;
    - native spin (``edge_vec``): the energy edge schema plus the owned-atom
      spins (the first ``nloc`` extended rows, where ``mapping`` is identity);
    - energy (``edge_vec``): the plain single-domain edge schema.
    """
    (
        ext_coord,
        ext_atype,
        nlist_t,
        mapping_t,
        ext_spin,
        fparam,
        aparam,
        charge_spin,
    ) = _build_sample_extended(model, nframes, nloc, device, has_spin)
    if has_spin and model.export_lower_input_kind() == "nlist":
        return (
            ext_coord,
            ext_atype,
            ext_spin,
            nlist_t,
            mapping_t,
            fparam,
            aparam,
            charge_spin,
        )
    formatted_nlist: torch.Tensor = model.format_nlist(ext_coord, ext_atype, nlist_t)
    edge_schema = edge_schema_from_extended(
        ext_coord,
        ext_atype[:, :nloc],
        formatted_nlist,
        mapping_t,
    )
    if has_spin:
        return (
            edge_schema.coord,
            edge_schema.atype,
            edge_schema.edge_index,
            edge_schema.edge_vec,
            edge_schema.edge_scatter_index,
            edge_schema.edge_mask,
            ext_spin[:, :nloc],
            fparam,
            aparam,
            charge_spin,
        )
    return (
        edge_schema.coord,
        edge_schema.atype,
        edge_schema.edge_index,
        edge_schema.edge_vec,
        edge_schema.edge_scatter_index,
        edge_schema.edge_mask,
        fparam,
        aparam,
        charge_spin,
    )


def _make_edge_comm_tensors(
    mapping: torch.Tensor,
    nloc: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build a single self-send swap so the with-comm trace runs ``border_op``.

    A LAMMPS run supplies the real per-swap communication plan at inference time;
    the trace only needs valid in-range indices so the eager output-key probe can
    execute the opaque op. Ghost slot ``k`` copies its owner's local index
    ``mapping[nloc + k]``.
    """
    nall = int(mapping.shape[1])
    nghost = nall - nloc
    send_count = max(1, nghost)
    owner = mapping[0, nloc:nall].to(dtype=torch.int32).cpu().numpy()
    indices = np.ascontiguousarray(np.resize(owner, send_count).astype(np.int32))
    _TRACE_SENDLIST_KEEPALIVE.append(indices)
    addr = indices.ctypes.data_as(ctypes.c_void_p).value
    return (
        torch.tensor([addr], dtype=torch.int64, device=device),  # send_list (int**)
        torch.zeros(1, dtype=torch.int32, device=device),  # send_proc (self)
        torch.zeros(1, dtype=torch.int32, device=device),  # recv_proc (self)
        torch.tensor([send_count], dtype=torch.int32, device=device),  # send_num
        torch.tensor([send_count], dtype=torch.int32, device=device),  # recv_num
        torch.zeros(1, dtype=torch.int64, device=device),  # communicator
        torch.tensor(nloc, dtype=torch.int32, device=device),  # nlocal
        torch.tensor(nghost, dtype=torch.int32, device=device),  # nghost
    )


def _make_comm_sample_inputs(
    model: torch.nn.Module,
    nloc: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, ...]:
    """Build with-comm edge inputs for tracing the parallel ``.pt2`` artifact.

    The parallel path indexes the extended node set directly, so ``edge_index``
    coincides with ``edge_scatter_index`` (both extended) and ghost features are
    refreshed via ``border_op`` rather than gathered through a folded mapping.
    The frame axis is fixed at one, matching LAMMPS single-frame inference. The
    native spin scheme threads the EXTENDED per-node spin (ghost spins ride the
    same exchange), inserted after ``edge_mask`` to match its with-comm signature.
    """
    has_spin = _model_has_spin(model)
    (
        ext_coord,
        ext_atype,
        nlist_t,
        mapping_t,
        ext_spin,
        fparam,
        aparam,
        charge_spin,
    ) = _build_sample_extended(
        model, nframes=1, nloc=nloc, device=device, has_spin=has_spin
    )
    formatted_nlist: torch.Tensor = model.format_nlist(ext_coord, ext_atype, nlist_t)
    edge_schema = edge_schema_from_extended(
        ext_coord,
        ext_atype[:, :nloc],
        formatted_nlist,
        mapping_t,
    )
    edge_inputs = (
        edge_schema.coord,  # (1, nall, 3)
        edge_schema.atype,  # (1, nloc)
        ext_atype,  # (1, nall)
        edge_schema.edge_scatter_index,  # edge_index: extended (2, E)
        edge_schema.edge_vec,
        edge_schema.edge_scatter_index,  # edge_scatter_index: extended (2, E)
        edge_schema.edge_mask,
    )
    comm_tensors = _make_edge_comm_tensors(mapping_t, nloc, device)
    if has_spin:
        return (*edge_inputs, ext_spin, fparam, aparam, charge_spin, *comm_tensors)
    return (*edge_inputs, fparam, aparam, charge_spin, *comm_tensors)


def _resolve_nframes(
    model: torch.nn.Module,
    nloc: int,
    device: torch.device,
    start: int = 2,
    has_spin: bool = False,
) -> tuple[int, tuple[torch.Tensor | None, ...]]:
    """Pick an ``nframes`` that does not collide with any other dim size.

    ``torch.export``'s duck-sizing unifies symbolic dims whose concrete
    sample values match; if ``nframes`` happens to equal, say, the
    spatial ``3`` or the virial ``9``, the ExportedProgram rejects
    later calls whose ``nframes`` differs.  Bumping ``nframes`` until
    no collision is left keeps the export safe.
    """
    nframes = start
    sample = _make_sample_inputs(
        model,
        nframes=nframes,
        nloc=nloc,
        device=device,
        has_spin=has_spin,
    )
    other_dims: set[int] = set()
    for t in sample:
        if t is not None:
            other_dims.update(t.shape[1:])
    while nframes in other_dims:
        nframes += 1
    if nframes != start:
        sample = _make_sample_inputs(
            model,
            nframes=nframes,
            nloc=nloc,
            device=device,
            has_spin=has_spin,
        )
    return nframes, sample


def _build_dynamic_shapes(
    sample_inputs: tuple[torch.Tensor | None, ...],
) -> tuple:
    """Build positional dynamic-shape constraints for the traced lower input.

    The lower ABI is recovered from the sample structure: a floating-point
    tensor at index 2 is the extended spin of the deepspin-scheme nlist contract,
    while an integer ``edge_index`` there marks the edge contract. A native-spin
    edge sample carries the extra per-local-atom spin tensor, giving it ten
    positional entries against the energy contract's nine.
    """
    nframes_dim = torch.export.Dim("nframes", min=1)
    nloc_dim = torch.export.Dim("nloc", min=1)
    nedge_dim = torch.export.Dim("nedge", min=2)
    is_nlist_spin = (
        len(sample_inputs) >= 3
        and sample_inputs[2] is not None
        and sample_inputs[2].is_floating_point()
    )
    if is_nlist_spin:
        nall_dim = torch.export.Dim("nall", min=4)
        fparam = sample_inputs[5]
        aparam = sample_inputs[6]
        charge_spin = sample_inputs[7] if len(sample_inputs) == 8 else None
        shapes = (
            {0: nframes_dim, 1: nall_dim},  # extended_coord
            {0: nframes_dim, 1: nall_dim},  # extended_atype
            {0: nframes_dim, 1: nall_dim},  # extended_spin
            {0: nframes_dim, 1: nloc_dim},  # nlist
            {0: nframes_dim, 1: nall_dim},  # mapping
            {0: nframes_dim} if fparam is not None else None,
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,
        )
        if len(sample_inputs) == 8:
            shapes = (*shapes, {0: nframes_dim} if charge_spin is not None else None)
        return shapes

    nall_dim = torch.export.Dim("nall", min=1)
    edge_shapes = (
        {0: nframes_dim, 1: nall_dim},  # extended_coord: (nframes, nall, 3)
        {0: nframes_dim, 1: nloc_dim},  # atype
        {1: nedge_dim},  # edge_index
        {0: nedge_dim},  # edge_vec
        {1: nedge_dim},  # edge_scatter_index
        {0: nedge_dim},  # edge_mask
    )
    # Native-spin edge contract: extra per-local-atom spin leaf at index 6.
    is_native_spin = len(sample_inputs) == 10
    if is_native_spin:
        fparam, aparam, charge_spin = (
            sample_inputs[7],
            sample_inputs[8],
            sample_inputs[9],
        )
        return (
            *edge_shapes,
            {0: nframes_dim, 1: nloc_dim},  # spin: (nframes, nloc, 3)
            {0: nframes_dim} if fparam is not None else None,
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,
            {0: nframes_dim} if charge_spin is not None else None,
        )
    fparam = sample_inputs[6]
    aparam = sample_inputs[7]
    charge_spin = sample_inputs[8] if len(sample_inputs) == 9 else None
    shapes = (
        *edge_shapes,
        {0: nframes_dim} if fparam is not None else None,
        {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,
    )
    if len(sample_inputs) == 9:
        shapes = (*shapes, {0: nframes_dim} if charge_spin is not None else None)
    return shapes


def _build_with_comm_dynamic_shapes(
    sample_inputs: tuple[torch.Tensor | None, ...],
) -> tuple:
    """Build dynamic-shape constraints for the parallel with-comm lower input.

    The frame axis is fixed at one (LAMMPS single-frame inference), so only
    ``nall``, ``nloc`` and ``nedge`` vary. The eight communication tensors are
    static: ``nswap`` is fixed at LAMMPS init and the graph carries no variation
    across its value (``border_op`` is opaque to the exported program). The
    native spin contract inserts the extended (nall) spin after ``edge_mask``,
    giving 19 positional entries against the energy contract's 18.
    """
    nall_dim = torch.export.Dim("nall", min=1)
    nloc_dim = torch.export.Dim("nloc", min=1)
    nedge_dim = torch.export.Dim("nedge", min=2)
    edge_base = (
        {1: nall_dim},  # coord: (1, nall, 3)
        {1: nloc_dim},  # atype: (1, nloc)
        {1: nall_dim},  # extended_atype: (1, nall)
        {1: nedge_dim},  # edge_index: (2, nedge)
        {0: nedge_dim},  # edge_vec: (nedge, 3)
        {1: nedge_dim},  # edge_scatter_index: (2, nedge)
        {0: nedge_dim},  # edge_mask: (nedge,)
    )
    is_native_spin = len(sample_inputs) == 19
    if is_native_spin:
        fparam, aparam, charge_spin = (
            sample_inputs[8],
            sample_inputs[9],
            sample_inputs[10],
        )
        base = (
            *edge_base,
            {1: nall_dim},  # spin: (1, nall, 3)
            None if fparam is None else {},  # fparam: (1, ndf) static
            None if aparam is None else {1: nloc_dim},  # aparam: (1, nloc, nda)
            None if charge_spin is None else {},  # charge_spin: (1, nchg) static
        )
        return (*base, *((None,) * 8))
    fparam = sample_inputs[7]
    aparam = sample_inputs[8]
    charge_spin = sample_inputs[9]
    base = (
        *edge_base,
        None if fparam is None else {},  # fparam: (1, ndf) static
        None if aparam is None else {1: nloc_dim},  # aparam: (1, nloc, nda)
        None if charge_spin is None else {},  # charge_spin: (1, nchg) static
    )
    return (*base, *((None,) * 8))


def _export_with_comm_artifact(
    model: torch.nn.Module,
    *,
    target_device: torch.device,
    compile_options: dict[str, Any],
) -> bytes:
    """Trace, export and compile the parallel with-comm ``.pt2`` artifact.

    The artifact mirrors the regular edge graph but exchanges ghost node
    features across ranks via ``border_op``. Returns the compiled package bytes
    for nesting under ``model/extra/forward_lower_with_comm.pt2``; tracing runs
    on CPU and the package is moved to ``target_device`` before compilation.
    """
    from torch._inductor import (
        aoti_compile_and_package,
    )
    from torch._inductor import config as inductor_config

    sample_inputs = _make_comm_sample_inputs(model, nloc=7, device=torch.device("cpu"))
    traced = model.forward_common_lower_exportable_with_comm(*sample_inputs)
    exported = torch.export.export(
        traced,
        sample_inputs,
        dynamic_shapes=_build_with_comm_dynamic_shapes(sample_inputs),
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )
    _strip_shape_assertions(exported.graph_module)
    if target_device.type != "cpu":
        from torch.export.passes import (
            move_to_device_pass,
        )

        exported = move_to_device_pass(exported, target_device)
    with tempfile.TemporaryDirectory() as td:
        wc_path = os.path.join(td, "forward_lower_with_comm.pt2")
        with inductor_config.patch({**compile_options, "triton.max_tiles": 1}):
            aoti_compile_and_package(exported, package_path=wc_path)
        with open(wc_path, "rb") as fh:
            return fh.read()


def freeze_sezm_to_pt2(
    ckpt_path: str,
    out_path: str,
    *,
    device: torch.device | None = None,
    head: str | None = None,
    atomic_virial: bool = True,
) -> None:
    """Freeze a SeZM checkpoint into an AOTInductor ``.pt2`` archive.

    Parameters
    ----------
    ckpt_path
        Path to the SeZM training checkpoint (``.pt``).
    out_path
        Destination file.  A ``.pt2`` suffix is expected.
    device
        Target device for the compiled shared library.  Defaults to
        :data:`DEVICE`.  Tracing itself always runs on CPU.
    head
        Model head to export from a multi-task checkpoint. If omitted, the
        ``Default`` head is used when present; otherwise multi-task checkpoints
        must pass an explicit head. Single-task checkpoints must pass ``None``.
    atomic_virial
        Whether the exported model exposes per-atom virial.  Enabled by
        default: the edge-force scatter assembles the per-atom virial as a
        free by-product of the single backward, so exporting it carries no
        compute cost.
    """
    log.info(
        "Set DP_TRITON_INFER to the desired level (0-3) before freezing; "
        "the selected Triton inference kernels are baked into the .pt2 archive."
    )

    from torch._inductor import (
        aoti_compile_and_package,
    )
    from torch._inductor import config as inductor_config

    target_device = device if device is not None else DEVICE

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict, params = _extract_state_and_params(raw)
    state_dict, params = _select_model_head(state_dict, params, head)

    model_type = str(params.get("type", "")).lower()
    if model_type not in ("sezm", "dpa4"):
        raise ValueError(
            f"freeze_sezm_to_pt2 expects a SeZM/DPA4 checkpoint, got type={params.get('type')!r}."
        )

    model = get_model(params)
    is_spin = _model_has_spin(model)
    ModelWrapper(model).load_state_dict(state_dict)
    model.eval()
    model.to("cpu")

    # The SO(2) linear mixer selects its block-diagonal vs dense matmul from a
    # Python device branch that make_fx resolves at trace time. Since tracing
    # always runs on CPU, pin the choice to the AOTI target device: non-CPU
    # targets bake the block-diagonal contraction (which skips the structural
    # off-|m| zeros); CPU targets keep the dense einsum that dodges the Inductor
    # AVX2 codegen bug.
    force_block_diag = target_device.type != "cpu"
    for module in model.modules():
        if isinstance(module, SO2Linear):
            module._force_block_diag_matmul = force_block_diag

    # Sweep any Triton launch-table keys this checkpoint needs that are not
    # covered for the local GPU, so the traced graph bakes tuned launches.
    _tune_triton_configs(model, target_device)

    _, sample_inputs_cpu = _resolve_nframes(
        model,
        nloc=7,
        device=torch.device("cpu"),
        has_spin=is_spin,
    )

    # Each model's exportable signature matches its sample tuple positionally
    # (energy / native-spin edge ABI, or virtual-spin nlist ABI), so a single
    # splat covers all three contracts.
    log.info("Tracing the lower graph on CPU (make_fx)...")
    traced = model.forward_common_lower_exportable(*sample_inputs_cpu)

    # Output key order is taken from a concrete run; Python dict order
    # is stable and matches what DeepPotPTExpt::extract_outputs zips
    # against AOTIModelPackageLoader::run's output vector.
    with torch.no_grad():
        sample_out = traced(*sample_inputs_cpu)
    output_keys = list(sample_out.keys())

    log.info("Exporting the traced graph (torch.export)...")
    exported = torch.export.export(
        traced,
        sample_inputs_cpu,
        dynamic_shapes=_build_dynamic_shapes(sample_inputs_cpu),
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )
    _strip_shape_assertions(exported.graph_module)

    # move_to_device_pass handles FakeTensor device propagation cleanly;
    # a naive .to(device) on the exported program does not.
    if target_device.type != "cpu":
        from torch.export.passes import (
            move_to_device_pass,
        )

        exported = move_to_device_pass(exported, target_device)

    out_path_str = str(out_path)
    compile_options = build_inductor_compile_options(inference=True)
    # Keep AOTInductor aligned with the eval compile path.  ``triton.max_tiles=1``
    # keeps data-dependent edge axes on Triton's x grid, whose bound is large
    # enough for production-scale neighbor lists.
    log.info(
        "Compiling the AOTInductor package for %s (the slowest freeze stage; "
        "typically several minutes)...",
        target_device,
    )
    with inductor_config.patch({**compile_options, "triton.max_tiles": 1}):
        aoti_compile_and_package(exported, package_path=out_path_str)

    # Second artifact: the LAMMPS multi-rank with-comm graph. It threads the
    # eight border_op communication tensors so cross-rank ghost features are
    # exchanged between interaction blocks. Gated on the edge_vec lower contract
    # (energy and native spin), so virtual spin (nlist interface) is excluded;
    # bridging models report supports_edge_parallel()=False (Source Freeze
    # Propagation is not rank-decomposable). Both fall back to single-rank.
    with_comm = (
        model.export_lower_input_kind() == "edge_vec" and model.supports_edge_parallel()
    )
    with_comm_bytes: bytes | None = None
    if with_comm:
        log.info(
            "Compiling the parallel with-comm artifact (second AOTInductor "
            "compilation)..."
        )
        with_comm_bytes = _export_with_comm_artifact(
            model,
            target_device=target_device,
            compile_options=compile_options,
        )

    metadata = _collect_metadata(
        model,
        output_keys=output_keys,
        is_spin=is_spin,
        do_atomic_virial=atomic_virial,
        has_comm_artifact=with_comm,
    )
    with zipfile.ZipFile(out_path_str, "a") as zf:
        zf.writestr("model/extra/metadata.json", json.dumps(metadata))
        if with_comm_bytes is not None:
            zf.writestr("model/extra/forward_lower_with_comm.pt2", with_comm_bytes)
        # The raw training params are preserved so `dp change-bias` and
        # other downstream tooling can recover the exact training config.
        # ``default=str`` is a safety net for exotic nested values.
        zf.writestr(
            "model/extra/model_def_script.json",
            json.dumps(params, default=str),
        )

    log.info(
        "Saved SeZM .pt2 to %s (device=%s, output_keys=%s)",
        out_path_str,
        target_device,
        output_keys,
    )
    log.info(
        "Thank you for using the DPA4/SeZM model! If it benefits your "
        "research, please cite the DPA4 paper "
        "(https://arxiv.org/abs/2606.02419):"
    )
    log.info(
        "\n"
        "@article{li2026dpa4,\n"
        "  title = {{DPA4}: Pushing the Accuracy-Cost Frontier of Interatomic "
        "Potentials with {EMFA} {SO(2)} Convolution},\n"
        "  author = {Li, Tiancheng and Li, Wentao and Peng, Anyang and "
        "Xue, Jianming and Zhang, Linfeng and Zhang, Duo and Wang, Han},\n"
        "  journal = {arXiv preprint arXiv:2606.02419},\n"
        "  year = {2026},\n"
        "  eprint = {2606.02419},\n"
        "  archivePrefix = {arXiv},\n"
        "  primaryClass = {physics.chem-ph},\n"
        "  doi = {10.48550/arXiv.2606.02419},\n"
        "  url = {https://arxiv.org/abs/2606.02419}\n"
        "}"
    )


__all__ = [
    "freeze_sezm_to_pt2",
    "is_sezm_checkpoint",
]
