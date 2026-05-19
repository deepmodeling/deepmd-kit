# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 / SeZM → AOTInductor ``.pt2`` freeze path for the pt backend.

SeZM relies on a nested ``autograd.grad(create_graph=True)`` inside
``fit_output_to_model_output``; TorchScript cannot represent that
graph, so DPA4 / SeZM checkpoints are routed through AOTInductor instead.
The output archive layout matches the ``pt_expt`` convention and is
consumed directly by ``DeepPotPTExpt.cc`` without any C++ change.

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

import json
import logging
import zipfile
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
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.env import (
    DEVICE,
)

log = logging.getLogger(__name__)


def _model_has_spin(model: torch.nn.Module) -> bool:
    """Return whether ``model`` uses the spin lower interface."""
    has_spin = getattr(model, "has_spin", False)
    return bool(has_spin() if callable(has_spin) else has_spin)


def _strip_shape_assertions(graph_module: torch.nn.Module) -> None:
    """Remove deferred shape assertions from spin export graphs.

    The spin lower path slices tensors using both ``nall`` and ``nloc`` after
    virtual atom expansion. ``torch.export`` may turn valid dynamic cases into
    deferred ``Ne(nall, nloc)`` assertions, even though the graph works for both
    NoPBC and ghost-atom inputs. The generic pt_expt spin exporter applies the
    same cleanup.
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
    return str(params.get("type", "")).lower() in ("sezm", "dpa4")


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
    metadata = {
        "type_map": list(model.get_type_map()),
        "rcut": float(model.get_rcut()),
        "sel": [int(s) for s in model.get_sel()],
        "dim_fparam": int(model.get_dim_fparam()),
        "dim_aparam": int(model.get_dim_aparam()),
        "mixed_types": bool(model.mixed_types()),
        "has_default_fparam": bool(model.has_default_fparam()),
        "default_fparam": _to_py_list(model.get_default_fparam()),
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


def _make_sample_inputs(
    model: torch.nn.Module,
    nframes: int,
    nloc: int,
    device: torch.device,
    has_spin: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Build representative ``forward_common_lower`` inputs for tracing.

    Tensors are float64 / int64 (matching the ``.pt2`` I/O contract).
    """
    rcut = float(model.get_rcut())
    sel = list(model.get_sel())
    ntypes = len(model.get_type_map())
    dim_fparam = int(model.get_dim_fparam())
    dim_aparam = int(model.get_dim_aparam())
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
    if has_spin:
        return ext_coord, ext_atype, ext_spin, nlist_t, mapping_t, fparam, aparam
    return ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam


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
    """Positional ``dynamic_shapes`` for the traced
    ``(ext_coord, ext_atype, nlist, mapping, fparam, aparam)`` signature.
    """
    nframes_dim = torch.export.Dim("nframes", min=1)
    has_spin = len(sample_inputs) == 7
    # Spin export currently generates a valid lower-bound guard from its
    # virtual-atom split/concat pattern. Matching the bound keeps export strict,
    # while `_strip_shape_assertions` removes the spurious deferred guards later.
    nall_dim = torch.export.Dim("nall", min=4 if has_spin else 1)
    nloc_dim = torch.export.Dim("nloc", min=1)
    fparam = sample_inputs[5] if has_spin else sample_inputs[4]
    aparam = sample_inputs[6] if has_spin else sample_inputs[5]
    if has_spin:
        return (
            {0: nframes_dim, 1: nall_dim},  # extended_coord
            {0: nframes_dim, 1: nall_dim},  # extended_atype
            {0: nframes_dim, 1: nall_dim},  # extended_spin
            {0: nframes_dim, 1: nloc_dim},  # nlist
            {0: nframes_dim, 1: nall_dim},  # mapping
            {0: nframes_dim} if fparam is not None else None,
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,
        )
    return (
        {0: nframes_dim, 1: nall_dim},  # extended_coord: (nframes, nall, 3)
        {0: nframes_dim, 1: nall_dim},  # extended_atype: (nframes, nall)
        {0: nframes_dim, 1: nloc_dim},  # nlist: (nframes, nloc, nnei)
        {0: nframes_dim, 1: nall_dim},  # mapping: (nframes, nall)
        {0: nframes_dim} if fparam is not None else None,
        {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,
    )


def freeze_sezm_to_pt2(
    ckpt_path: str,
    out_path: str,
    *,
    device: torch.device | None = None,
    head: str | None = None,
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
        Reserved for future multi-task support; must be ``None``.
    """
    from torch._inductor import (
        aoti_compile_and_package,
    )

    target_device = device if device is not None else DEVICE

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict, params = _extract_state_and_params(raw)

    if str(params.get("type", "")).lower() != "sezm":
        raise ValueError(
            f"freeze_sezm_to_pt2 expects a SeZM checkpoint, got type={params.get('type')!r}."
        )
    if "model_dict" in params:
        raise NotImplementedError(
            "SeZM .pt2 freeze does not yet support multi-task checkpoints."
        )
    if head is not None:
        raise NotImplementedError(
            "SeZM .pt2 freeze does not yet support head selection; pass head=None."
        )

    model = get_model(params)
    is_spin = _model_has_spin(model)
    ModelWrapper(model).load_state_dict(state_dict)
    model.eval()
    model.to("cpu")

    _, sample_inputs_cpu = _resolve_nframes(
        model,
        nloc=7,
        device=torch.device("cpu"),
        has_spin=is_spin,
    )

    # do_atomic_virial=True pulls every key that DeepPotPTExpt may read
    # (energy, energy_redu, energy_derv_r, energy_derv_c, energy_derv_c_redu)
    # into the traced graph.
    traced = model.forward_common_lower_exportable(
        *sample_inputs_cpu,
        do_atomic_virial=True,
    )

    # Output key order is taken from a concrete run; Python dict order
    # is stable and matches what DeepPotPTExpt::extract_outputs zips
    # against AOTIModelPackageLoader::run's output vector.
    with torch.no_grad():
        sample_out = traced(*sample_inputs_cpu)
    output_keys = list(sample_out.keys())

    exported = torch.export.export(
        traced,
        sample_inputs_cpu,
        dynamic_shapes=_build_dynamic_shapes(sample_inputs_cpu),
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )
    if is_spin:
        _strip_shape_assertions(exported.graph_module)

    # move_to_device_pass handles FakeTensor device propagation cleanly;
    # a naive .to(device) on the exported program does not.
    if target_device.type != "cpu":
        from torch.export.passes import (
            move_to_device_pass,
        )

        exported = move_to_device_pass(exported, target_device)

    out_path_str = str(out_path)
    aoti_compile_and_package(exported, package_path=out_path_str)

    metadata = _collect_metadata(model, output_keys=output_keys, is_spin=is_spin)
    with zipfile.ZipFile(out_path_str, "a") as zf:
        zf.writestr("model/extra/metadata.json", json.dumps(metadata))
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


__all__ = [
    "freeze_sezm_to_pt2",
    "is_sezm_checkpoint",
]
