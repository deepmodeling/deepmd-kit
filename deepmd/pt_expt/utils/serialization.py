# SPDX-License-Identifier: LGPL-3.0-or-later
import contextlib
import ctypes
import json
import logging
import os
from collections.abc import (
    Iterator,
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

log = logging.getLogger(__name__)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.dpmodel.utils.serialization import (
    traverse_model_dict,
)

# ---------------------------------------------------------------------------
# AOTInductor ``.pt2`` archive layout.
#
# PyTorch 2.11 tightened the single-model ``.pt2`` convention so that every
# entry in the ZIP archive must live under the top-level ``model/`` directory.
# Any stray root-level file makes
# ``torch.export.pt2_archive._package.load_pt2`` raise ``RuntimeError`` at
# load time; the upper-level ``torch._inductor.package.package.load_package``
# then emits a misleading ``Loading outdated pt2 file. Please regenerate
# your package.`` warning and falls back to the legacy C++ loader.
#
# deepmd-kit therefore stores its metadata JSON blobs under ``model/extra/``
# so that the strict ``load_pt2`` loader accepts the archive without
# complaint.  The C++ reader (``commonPTExpt.h::read_zip_entry``) resolves
# this layout transparently because it matches ``entry_name`` as a
# ``/``-delimited suffix.
# ---------------------------------------------------------------------------
PT2_EXTRA_PREFIX = "model/extra/"


def _strip_shape_assertions(graph_module: torch.nn.Module) -> None:
    """Neutralise deferred shape-guard assertion nodes in an exported graph.

    ``torch.export`` (with ``prefer_deferred_runtime_asserts_over_guards=True``)
    inserts ``aten._assert_scalar`` nodes for symbolic-shape relationships
    discovered during tracing.  The assertion messages use opaque symbolic names
    (e.g. ``Ne(s22, s96)``), so filtering by message content is not reliable; we
    replace each assertion's condition with ``True`` rather than erasing the node
    (erasing can disturb the FX graph and yield NaN gradients on some torch
    versions).

    Called from two export paths in ``_trace_and_export``:

    * **spin (dense) models** — atom-doubling slice patterns depend on
      ``(nall - nloc)``, producing spurious guards like ``Ne(nall, nloc)``; the
      model is correct even when ``nall == nloc`` (NoPBC, no ghosts).
    * **graph models** — the dynamic edge axis (``Dim("nedge")``) produces
      shape-specialization guards on the edge count ``E``.

    In both contexts every input is constructed well-formed by the
    builder (spin: valid atom doubling; graph: ``build_neighbor_graph`` /
    ``buildGraphTensors`` always emit ``E >= min_edges == 2`` with in-range,
    masked edges). Malformed runtime tensors are outside this exported ABI and
    are not guaranteed to trigger these shape assertions.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        ):
            node.args = (True, node.args[1])
    graph_module.recompile()


def _numpy_to_json_serializable(model_obj: dict) -> dict:
    """Convert numpy arrays in a model dict to JSON-serializable lists."""
    return traverse_model_dict(
        model_obj,
        lambda x: (
            {
                "@class": "np.ndarray",
                "@is_variable": True,
                "dtype": x.dtype.name,
                "value": x.tolist(),
            }
            if isinstance(x, np.ndarray)
            else x
        ),
    )


def _json_to_numpy(model_obj: dict) -> dict:
    """Convert JSON-serialized numpy arrays back to np.ndarray."""
    return traverse_model_dict(
        model_obj,
        lambda x: (
            np.asarray(x["value"], dtype=np.dtype(x["dtype"]))
            if isinstance(x, dict) and x.get("@class") == "np.ndarray"
            else x
        ),
    )


def _metadata_value_to_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _needs_with_comm_artifact(model: torch.nn.Module) -> bool:
    """Return ``True`` if the model needs a "with-comm" AOTI artifact compiled.

    The with-comm artifact carries the per-layer ``deepmd_export::border_op``
    calls that exchange node-embedding tensors across MPI ranks. Multi-rank
    LAMMPS dispatches to it when the descriptor's message passing extends
    across rank boundaries (i.e. layers consume neighbour features that
    live on a different rank). Non-GNN descriptors and GNN descriptors with
    ``use_loc_mapping=True`` keep all per-layer messaging local to each
    rank's owned atoms; they need only the regular artifact.

    Delegates to ``descriptor.has_message_passing_across_ranks()``, which
    descriptor classes implement explicitly. Returns ``False`` defensively
    when the model has no single descriptor (linear/zbl/frozen) or when
    the method is somehow missing or raises.
    """
    desc = getattr(getattr(model, "atomic_model", None), "descriptor", None)
    if desc is None or not hasattr(desc, "has_message_passing_across_ranks"):
        return False
    try:
        return bool(desc.has_message_passing_across_ranks())
    except (AttributeError, NotImplementedError):
        return False


def check_graph_trace_torch_version(model: torch.nn.Module) -> None:
    """Fail fast when the graph trace needs unbacked-SymInt support torch lacks.

    The compact ``center_edge_pairs`` realization used by graph attention
    (``attn_layer > 0``) relies on unbacked-SymInt tracing
    (``torch._check_is_size`` hints on ``nonzero`` / tensor-``repeat`` outputs,
    see ``deepmd/dpmodel/utils/neighbor_graph/pairs.py``), which is only solid
    from torch >= 2.6. On older torch the trace dies deep inside
    ``make_fx``/AOTI with an obscure ``GuardOnDataDependentSymNode`` (or an
    ``AttributeError`` on ``_check_is_size``), so both graph trace sites (the
    ``.pt2`` export below and the training compile in
    ``training._trace_and_compile_graph``) call this guard first. Factorizable
    models (``attn_layer == 0``) trace with backed symbols only and are not
    restricted.

    Parameters
    ----------
    model
        The graph-eligible model about to be traced. The attention depth is
        read from ``model.atomic_model.descriptor.get_numb_attn_layer()``;
        models without a single descriptor (linear/zbl/frozen) pass the
        check (they take the dense route anyway).

    Raises
    ------
    RuntimeError
        If the descriptor has ``attn_layer > 0`` and the running torch is
        older than 2.6.
    """
    desc = getattr(getattr(model, "atomic_model", None), "descriptor", None)
    get_n_attn = getattr(desc, "get_numb_attn_layer", None)
    n_attn = get_n_attn() if get_n_attn is not None else 0
    if n_attn <= 0:
        return
    version = torch.__version__.split("+")[0]
    major_minor = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
    if len(major_minor) == 2 and major_minor < (2, 6):
        raise RuntimeError(
            f"graph-form tracing of attention layers (attn_layer={n_attn}) "
            f"requires torch >= 2.6 (unbacked-SymInt support for the compact "
            f"center_edge_pairs realization); found torch {torch.__version__}. "
            "Upgrade torch, set 'attn_layer: 0', or use the dense (nlist) path."
        )


# Module-level cache for the trace-time sendlist buffer. The pointer
# value embedded in ``send_list_tensor`` references this numpy array's
# data; the array must outlive the trace + export call.  Caching here
# (rather than per-call) is fine because the contents are never read by
# the exported graph at runtime — only by the eager call inside
# ``make_fx`` when extracting output keys, and by ``torch.export`` when
# materializing example inputs.
_TRACE_SENDLIST_KEEPALIVE: list[np.ndarray] = []


def _make_comm_sample_inputs(
    nloc: int,
    nghost: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build trivial-but-valid comm tensors for tracing the with-comm variant.

    Tracing with ``nswap == 0`` specializes the dimension, so the sample uses
    ``nswap == 1``
    with a single self-send swap whose sendlist points to ``nghost``
    local atoms (the actual indices don't matter for the trace — only
    the validity of the pointer matters; ``border_op`` is opaque to
    ``torch.export`` via the ``deepmd_export::border_op`` wrapper).

    Returns ``(send_list, send_proc, recv_proc, send_num, recv_num,
    communicator, nlocal_ts, nghost_ts)`` — 8 tensors, matching the
    canonical positional order of
    ``forward_common_lower_exportable_with_comm``.
    """
    nswap = 1
    send_count = max(1, nghost)
    # The trace-time sendlist must be a real ``int**``: a tensor of
    # int64 values, each value the address of a contiguous int32 array.
    indices = np.zeros(send_count, dtype=np.int32)
    _TRACE_SENDLIST_KEEPALIVE.append(indices)
    addr = indices.ctypes.data_as(ctypes.c_void_p).value
    send_list = torch.tensor([addr], dtype=torch.int64, device=device)
    send_proc = torch.zeros(nswap, dtype=torch.int32, device=device)
    recv_proc = torch.zeros(nswap, dtype=torch.int32, device=device)
    send_num = torch.tensor([send_count], dtype=torch.int32, device=device)
    recv_num = torch.tensor([send_count], dtype=torch.int32, device=device)
    communicator = torch.zeros(1, dtype=torch.int64, device=device)
    nlocal_ts = torch.tensor(nloc, dtype=torch.int32, device=device)
    nghost_ts = torch.tensor(nghost, dtype=torch.int32, device=device)
    return (
        send_list,
        send_proc,
        recv_proc,
        send_num,
        recv_num,
        communicator,
        nlocal_ts,
        nghost_ts,
    )


def _make_sample_inputs(
    model: torch.nn.Module,
    nframes: int = 1,
    nloc: int = 7,
    has_spin: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Create sample inputs for tracing forward_lower.

    Parameters
    ----------
    model : torch.nn.Module
        The pt_expt model (must have get_rcut, get_sel, get_type_map, etc.).
    nframes : int
        Number of frames.
    nloc : int
        Number of local atoms.
    has_spin : bool
        If True, create an extended spin tensor and return 7 tensors.

    Returns
    -------
    tuple
        (ext_coord, ext_atype, nlist, mapping, fparam, aparam, charge_spin) or
        (ext_coord, ext_atype, ext_spin, nlist, mapping, fparam, aparam,
        charge_spin) when has_spin.
    """
    rcut = model.get_rcut()
    sel = model.get_sel()
    ntypes = len(model.get_type_map())
    dim_fparam = model.get_dim_fparam()
    dim_aparam = model.get_dim_aparam()
    mixed_types = model.mixed_types()

    # Create a simple box large enough to avoid PBC issues
    box_size = rcut * 3.0
    box = np.eye(3, dtype=np.float64) * box_size
    box_np = box.reshape(1, 9)

    # Random coords inside the box
    rng = np.random.default_rng(42)
    coord_np = rng.random((nframes, nloc, 3), dtype=np.float64) * box_size * 0.5
    coord_np += box_size * 0.25  # center in box

    # Assign atom types: distribute across types
    atype_np = np.zeros((nframes, nloc), dtype=np.int32)
    for i in range(nloc):
        atype_np[:, i] = i % ntypes

    # Normalize and extend
    coord_normalized = normalize_coord(
        coord_np.reshape(nframes, nloc, 3),
        np.tile(box.reshape(1, 3, 3), (nframes, 1, 1)),
    )
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized,
        atype_np,
        np.tile(box_np, (nframes, 1)),
        rcut,
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

    # Convert to torch tensors
    import deepmd.pt_expt.utils.env as _env

    ext_coord = torch.tensor(extended_coord, dtype=torch.float64, device=_env.DEVICE)
    ext_atype = torch.tensor(extended_atype, dtype=torch.int64, device=_env.DEVICE)
    nlist_t = torch.tensor(nlist, dtype=torch.int64, device=_env.DEVICE)
    mapping_t = torch.tensor(mapping, dtype=torch.int64, device=_env.DEVICE)

    if dim_fparam > 0:
        fparam = torch.zeros(
            nframes, dim_fparam, dtype=torch.float64, device=_env.DEVICE
        )
    else:
        fparam = None

    if dim_aparam > 0:
        aparam = torch.zeros(
            nframes, nloc, dim_aparam, dtype=torch.float64, device=_env.DEVICE
        )
    else:
        aparam = None

    dim_chg_spin = model.get_dim_chg_spin() if hasattr(model, "get_dim_chg_spin") else 0
    if dim_chg_spin > 0:
        charge_spin = torch.zeros(
            nframes, dim_chg_spin, dtype=torch.float64, device=_env.DEVICE
        )
    else:
        charge_spin = None

    if has_spin:
        nall = extended_coord.shape[1]
        ext_spin = torch.zeros(
            nframes, nall, 3, dtype=torch.float64, device=_env.DEVICE
        )
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

    return ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam, charge_spin


def build_synthetic_graph_inputs(
    model: torch.nn.Module,
    e_max: int,
    nframes: int = 2,
    nloc: int = 7,
    *,
    dtype: torch.dtype,
    edge_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    want_fparam: bool = True,
    want_aparam: bool = True,
    want_charge_spin: bool = True,
) -> tuple[torch.Tensor | None, ...]:
    """Build a synthetic carry-all ``NeighborGraph`` for graph-lower tracing.

    Single source of the trace-time graph inputs, shared by ``.pt2`` export
    (:func:`_trace_and_export`) and compiled training
    (:func:`deepmd.pt_expt.train.training._trace_and_compile_graph`), so the two
    traces can never desync on the graph input schema.  Builds a small random
    system, runs the carry-all
    :func:`~deepmd.dpmodel.utils.neighbor_graph.build_neighbor_graph` with a
    padded ``GraphLayout(edge_capacity=e_max)`` trace sample, then canonicalizes
    it to the destination-major deployment ABI. The exported edge axis remains
    dynamic; the concrete capacity only supplies representative tensors to
    ``make_fx``. Inputs follow the positional order expected by
    ``forward_(common_)lower_graph``:
    ``(atype, n_node, n_local, edge_index, edge_vec, edge_mask, destination_order,
    destination_row_ptr, source_row_ptr, source_order, fparam, aparam,
    charge_spin)``.

    The system (``rng(42)``, ``box = rcut*3``, centered coords, ``atype[:, i] =
    i % ntypes``) is identical for both callers; the only two former differences
    are now parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The pt_expt energy model (must expose ``get_rcut``/``get_type_map``/...).
    e_max : int
        Concrete edge-axis size used by the trace sample.
    nframes : int
        Number of frames in the sample system.
    nloc : int
        Number of local atoms per frame (``N == nframes * nloc``).
    dtype : torch.dtype
        Float precision of ``coord``/``fparam`` and other conditioning inputs.
    edge_dtype : torch.dtype, optional
        Precision of ``edge_vec``. Defaults to ``dtype``. A compressed DPA1
        graph artifact uses float32 because its descriptor and analytical
        backward both compute in float32; generic graph artifacts preserve the
        model input precision.
    device : torch.device, optional
        Target device.  Defaults to ``deepmd.pt_expt.utils.env.DEVICE``; the
        export path passes ``cpu`` explicitly (make_fx traces on CPU).
    want_fparam, want_aparam, want_charge_spin : bool
        Whether to emit the optional conditioning tensor when its ``dim > 0``.
        Export passes the defaults (``True`` = include if present); training
        passes ``x is not None`` so the traced branch matches the run-time call.
    """
    import deepmd.pt_expt.utils.env as _env
    from deepmd.dpmodel.utils.neighbor_graph import (
        GraphLayout,
        build_neighbor_graph,
    )

    if device is None:
        device = _env.DEVICE
    if edge_dtype is None:
        edge_dtype = dtype

    rcut = model.get_rcut()
    ntypes = len(model.get_type_map())
    dim_fparam = model.get_dim_fparam()
    dim_aparam = model.get_dim_aparam()
    dim_chg_spin = model.get_dim_chg_spin() if hasattr(model, "get_dim_chg_spin") else 0

    # Box large enough to avoid PBC degeneracy; centered coords.
    box_size = rcut * 3.0
    box_np = (np.eye(3, dtype=np.float64) * box_size).reshape(1, 9)
    rng = np.random.default_rng(42)
    coord_np = rng.random((nframes, nloc, 3)) * box_size * 0.5 + box_size * 0.25
    atype_np = np.zeros((nframes, nloc), dtype=np.int64)
    for i in range(nloc):
        atype_np[:, i] = i % ntypes

    coord_t = torch.tensor(coord_np, dtype=dtype, device=device)
    atype_t = torch.tensor(atype_np, dtype=torch.int64, device=device)
    box_t = torch.tensor(np.tile(box_np, (nframes, 1)), dtype=dtype, device=device)
    graph = build_neighbor_graph(
        coord_t,
        atype_t,
        box_t,
        rcut,
        layout=GraphLayout(edge_capacity=e_max),
        canonicalize=True,
    )

    fparam = (
        torch.zeros(nframes, dim_fparam, dtype=dtype, device=device)
        if (want_fparam and dim_fparam > 0)
        else None
    )
    aparam = (
        torch.zeros(nframes, nloc, dim_aparam, dtype=dtype, device=device)
        if (want_aparam and dim_aparam > 0)
        else None
    )
    charge_spin = (
        torch.zeros(nframes, dim_chg_spin, dtype=dtype, device=device)
        if (want_charge_spin and dim_chg_spin > 0)
        else None
    )
    # Keep total and owned counts value-distinct during tracing so export does
    # not specialize the multi-rank ownership relation to ``n_local == n_node``.
    n_local = torch.clamp(graph.n_node - 1, min=1)

    return (
        atype_t.reshape(-1),
        graph.n_node,
        n_local,
        graph.edge_index,
        graph.edge_vec.to(edge_dtype),
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        fparam,
        aparam,
        charge_spin,
    )


def build_synthetic_canonical_graph_inputs(
    model: torch.nn.Module,
    e_max: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build the compact canonical trace inputs for compressed DPA1."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    sample = build_synthetic_graph_inputs(
        model,
        e_max,
        dtype=torch.float32,
        edge_dtype=torch.float32,
        device=device,
        want_fparam=False,
        want_aparam=False,
        want_charge_spin=False,
    )
    (
        atype,
        n_node,
        n_local,
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_row_ptr,
        source_order,
        _fparam,
        _aparam,
        _charge_spin,
    ) = sample
    graph = NeighborGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
        n_local=n_local,
        destination_order=destination_order,
        destination_row_ptr=destination_row_ptr,
        source_row_ptr=source_row_ptr,
        source_order=source_order,
        destination_sorted=True,
    )
    compact = canonical_graph_from_neighbor_graph(graph)
    return (
        atype,
        compact.n_node,
        compact.n_local,
        compact.source,
        compact.edge_vec,
        compact.destination_row_ptr,
        compact.source_row_ptr,
        compact.source_order,
    )


def _build_canonical_graph_dynamic_shapes(
    *sample_inputs: torch.Tensor,
) -> tuple:
    """Build dynamic shapes for the eight-tensor compact deployment ABI."""
    del sample_inputs
    nframes_dim = torch.export.Dim("nframes", min=1)
    node_dim = torch.export.Dim("n_node_total", min=1)
    edge_storage_dim = torch.export.Dim("nedge_storage", min=2)
    return (
        {0: node_dim},
        {0: nframes_dim},
        {0: nframes_dim},
        {0: edge_storage_dim},
        {0: edge_storage_dim},
        {0: node_dim + 1},
        {0: node_dim + 1},
        {0: edge_storage_dim},
    )


def _build_graph_dynamic_shapes(
    *sample_inputs: torch.Tensor | None,
) -> tuple:
    """Build dynamic-shape specifications for the graph-form forward_lower export.

    ``nframes`` (the ``n_node`` axis), ``N`` (the flat node axis), and the edge
    axis ``E`` are all dynamic dimensions. ``E`` is marked
    ``Dim("nedge", min=2)`` so
    the AOTI artifact accepts any system size with no capacity ceiling. The
    ``min=2`` lower bound mirrors the dense path's ``Dim("nnei", min=...)`` and
    matches the carry-all builder's
    ``min_edges=2`` guard (every dynamic graph carries >=2 edges).

    Parameters
    ----------
    *sample_inputs : torch.Tensor | None
        ``(atype, n_node, n_local, edge_index, edge_vec, edge_mask,
        destination_order, destination_row_ptr, source_row_ptr, source_order,
        fparam, aparam, charge_spin)`` — 13 entries matching
        ``forward_lower_graph_exportable``.
    """
    fparam = sample_inputs[10]
    aparam = sample_inputs[11]
    charge_spin = sample_inputs[12]
    nframes_dim = torch.export.Dim("nframes", min=1)
    n_node_total_dim = torch.export.Dim("n_node_total", min=1)
    nedge_dim = torch.export.Dim("nedge", min=2)
    nloc_dim = torch.export.Dim("nloc", min=1)
    return (
        {0: n_node_total_dim},  # atype: (N,)
        {0: nframes_dim},  # n_node: (nf,)
        {0: nframes_dim},  # n_local: (nf,)
        {1: nedge_dim},  # edge_index: (2, E) — E dynamic
        {0: nedge_dim},  # edge_vec: (E, 3) — E dynamic
        {0: nedge_dim},  # edge_mask: (E,) — E dynamic
        {0: nedge_dim},  # destination_order: (E,)
        {0: n_node_total_dim + 1},  # destination_row_ptr: (N + 1,)
        {0: n_node_total_dim + 1},  # source_row_ptr: (N + 1,)
        {0: nedge_dim},  # source_order: (E,)
        {0: nframes_dim} if fparam is not None else None,  # fparam: (nf, ndf)
        # aparam: (nf, nloc, nda) — both the frame AND atom axes are dynamic,
        # matching the dense ``_build_dynamic_shapes`` (otherwise a dim_aparam>0
        # graph export specializes nloc to the sample size and breaks at runtime).
        {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,  # aparam
        {0: nframes_dim} if charge_spin is not None else None,  # charge_spin
    )


def _build_dynamic_shapes(
    *sample_inputs: torch.Tensor | None,
    has_spin: bool = False,
    with_comm_dict: bool = False,
    model_nnei: int = 1,
) -> tuple:
    """Build dynamic shape specifications for torch.export.

    Marks nframes, nloc, nall and nnei as dynamic dimensions so the exported
    program handles arbitrary frame, atom and neighbor counts.

    When ``with_comm_dict`` is True, 8 additional comm tensors are
    appended to the returned tuple — matching the positional order of
    ``forward_common_lower_exportable_with_comm``.  ``nswap`` is the
    only dynamic dim among them; the rest are scalar or fixed-size.

    Parameters
    ----------
    *sample_inputs : torch.Tensor | None
        Sample inputs: 6 tensors (non-spin) or 7 (spin), optionally
        followed by 8 comm tensors when ``with_comm_dict``.
    has_spin : bool
        Whether the inputs include an extended_spin tensor.
    with_comm_dict : bool
        Whether the inputs include the 8 comm tensors.
    model_nnei : int
        The model's sum(sel).  Used as the min for the dynamic nnei dim.
    Returns a tuple (not dict) to match positional args of the make_fx
    traced module, whose arg names may have suffixes like ``_1``.
    """
    # When tracing the with-comm variant, nframes is static at 1.
    # Rationale: pt_expt's Repflow/Repformer parallel-mode override
    # mirrors pt's repflows.py:593 ``node_ebd.squeeze(0)`` /
    # ``…unsqueeze(0)`` pattern, which only works for nb=1. LAMMPS
    # always drives inference with one frame so this matches reality.
    # Marking nframes static (not dynamic) means it does not
    # participate in duck-sizing — so the nframes==2 collision-avoidance
    # chosen for the regular variant is *not* needed here, and the
    # static value (1) is safe regardless of other tensors' sizes.
    nframes_dim: torch.export.Dim | int = (
        1 if with_comm_dict else torch.export.Dim("nframes", min=1)
    )
    # Spin models double atom count internally (real + virtual). Some
    # GNN ops in the spin path generate a min=4 constraint on the
    # *pre-doubling* nall axis (matches "Suggested fixes" from
    # torch.export's CONSTRAINT_VIOLATION error). Bump the min for spin
    # so the export does not error on the inferred guard.
    nall_min = 4 if has_spin else 1
    nall_dim = torch.export.Dim("nall", min=nall_min)
    nloc_dim = torch.export.Dim("nloc", min=1)
    nnei_dim = torch.export.Dim("nnei", min=max(1, model_nnei))

    if has_spin:
        # (ext_coord, ext_atype, ext_spin, nlist, mapping, fparam, aparam, charge_spin)
        fparam = sample_inputs[5]
        aparam = sample_inputs[6]
        charge_spin = sample_inputs[7]
        base = (
            {0: nframes_dim, 1: nall_dim},  # extended_coord: (nframes, nall, 3)
            {0: nframes_dim, 1: nall_dim},  # extended_atype: (nframes, nall)
            {0: nframes_dim, 1: nall_dim},  # extended_spin: (nframes, nall, 3)
            {
                0: nframes_dim,
                1: nloc_dim,
                2: nnei_dim,
            },  # nlist: (nframes, nloc, nnei) — nnei is dynamic
            {0: nframes_dim, 1: nall_dim},  # mapping: (nframes, nall)
            {0: nframes_dim} if fparam is not None else None,  # fparam
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,  # aparam
            {0: nframes_dim} if charge_spin is not None else None,  # charge_spin
        )
    else:
        # (ext_coord, ext_atype, nlist, mapping, fparam, aparam, charge_spin)
        fparam = sample_inputs[4]
        aparam = sample_inputs[5]
        charge_spin = sample_inputs[6]
        base = (
            {0: nframes_dim, 1: nall_dim},  # extended_coord: (nframes, nall, 3)
            {0: nframes_dim, 1: nall_dim},  # extended_atype: (nframes, nall)
            {
                0: nframes_dim,
                1: nloc_dim,
                2: nnei_dim,
            },  # nlist: (nframes, nloc, nnei) — nnei is dynamic
            {0: nframes_dim, 1: nall_dim},  # mapping: (nframes, nall)
            {0: nframes_dim} if fparam is not None else None,  # fparam
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,  # aparam
            {0: nframes_dim} if charge_spin is not None else None,  # charge_spin
        )

    if not with_comm_dict:
        return base

    # All 8 comm tensors have static shapes:
    #   send_list, send_proc, recv_proc, send_num, recv_num: (nswap,)
    #   communicator: (1,)
    #   nlocal, nghost: scalar
    # nswap is fixed once at LAMMPS init (it depends on the processor
    # grid which doesn't change at runtime), so it's safe to bake it
    # in as static at the trace value.  Marking nswap dynamic instead
    # raises a Constraints-violated error because the trace specialises
    # it to the sample value (1) downstream of border_op anyway —
    # there is no graph variation across nswap values.
    return (*base, None, None, None, None, None, None, None, None)


def _graph_edge_dtype(model: torch.nn.Module, lower_kind: str) -> str:
    """Return the graph edge-vector dtype encoded by the deployment artifact.

    Geometrically compressed DPA1 with float32 descriptor statistics evaluates
    both descriptor directions in float32 and therefore accepts float32
    geometry directly. Other graph descriptors retain the model-agnostic
    float64 geometry ABI.
    """
    atomic_model = getattr(model, "atomic_model", None)
    descriptor = getattr(atomic_model, "descriptor", None)
    descriptor_block = getattr(descriptor, "se_atten", None)
    statistics = getattr(descriptor_block, "mean", None)
    if (
        lower_kind in ("graph", "dpa1_canonical")
        and bool(getattr(descriptor, "geo_compress", False))
        and isinstance(statistics, torch.Tensor)
        and statistics.dtype == torch.float32
    ):
        return "float32"
    return "float64"


def _supports_graph_export(model: torch.nn.Module) -> bool:
    """Whether the model has an exportable graph-lower implementation.

    A compressed descriptor must use its opaque graph operator during export;
    tracing through the reference tabulation kernel is unsupported.
    """
    atomic_model = getattr(model, "atomic_model", None)
    descriptor = getattr(atomic_model, "descriptor", None)
    if not bool(getattr(descriptor, "geo_compress", False)):
        return True
    eligible = getattr(descriptor, "_fused_eligible", None)
    return callable(eligible) and bool(eligible("cuda"))


def _collect_metadata(
    model: torch.nn.Module,
    is_spin: bool = False,
    lower_kind: str = "nlist",
) -> dict:
    """Collect metadata from the model for C++ inference.

    This metadata is stored as ``metadata.json`` in both .pt2 and .pte archives.
    Training config is stored separately in ``model_def_script.json``.  C++ reads
    flat JSON fields because compiling model API methods as AOTInductor
    entry points is impractical (~12 s per trivial function) and string
    outputs (``get_type_map``) cannot be expressed as tensor I/O.

    The ``fitting_output_defs`` list is also included so that
    ``ModelOutputDef`` can be reconstructed without loading the full model.
    """
    if is_spin:
        fitting_output_def = model.model_output_def().def_outp
    else:
        fitting_output_def = model.atomic_output_def()
    fitting_output_defs = []
    for vdef in fitting_output_def.get_data().values():
        # Keep metadata aligned with physical fitting outputs only.
        if is_spin and vdef.name == "mask":
            continue
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
                "magnetic": vdef.magnetic,
                "intensive": vdef.intensive,
            }
        )
    meta = {
        "type_map": model.get_type_map(),
        "rcut": model.get_rcut(),
        "sel": model.get_sel(),
        "nnei": sum(model.get_sel()),
        "dim_fparam": model.get_dim_fparam(),
        "dim_aparam": model.get_dim_aparam(),
        "dim_chg_spin": (
            model.get_dim_chg_spin() if hasattr(model, "get_dim_chg_spin") else 0
        ),
        "mixed_types": model.mixed_types(),
        "has_default_fparam": model.has_default_fparam(),
        "default_fparam": model.get_default_fparam(),
        "has_chg_spin_ebd": (
            model.has_chg_spin_ebd() if hasattr(model, "has_chg_spin_ebd") else False
        ),
        "has_default_chg_spin": (
            model.has_default_chg_spin()
            if hasattr(model, "has_default_chg_spin")
            else False
        ),
        "default_chg_spin": (
            _metadata_value_to_json(model.get_default_chg_spin())
            if hasattr(model, "get_default_chg_spin")
            else None
        ),
        "fitting_output_defs": fitting_output_defs,
        # sel_type enables `DeepEval.get_sel_type()` without a dpmodel
        # round-trip; required for dipole/polar/wfc models in metadata-only
        # inference (energy models return []).
        "sel_type": [int(t) for t in model.get_sel_type()],
        "is_spin": is_spin,
    }
    if is_spin:
        meta["ntypes_spin"] = model.spin.get_ntypes_spin()
        meta["use_spin"] = [bool(v) for v in model.spin.use_spin]
    # Whether multi-rank LAMMPS needs a second "with-comm" AOTI artifact
    # (per-layer ghost-feature MPI exchange via deepmd_export::border_op).
    # The C++ DeepPotPTExpt / DeepSpinPTExpt loaders branch on this flag.
    meta["has_comm_artifact"] = _needs_with_comm_artifact(model)

    # Whether the model's regular .pt2 graph consumes the ``mapping``
    # tensor to gather per-layer ghost-atom features from local atoms.
    # Mirrors the descriptor's ``has_message_passing()`` API: True for
    # any message-passing descriptor (DPA2, DPA3, hybrids over those);
    # False for non-message-passing descriptors (se_e2_a, DPA1, etc.).
    # The C++ side gates its fail-fast on this — an absent mapping is
    # fatal only for models that would silently corrupt ghost features
    # otherwise.
    #
    # Lookup order: model -> atomic_model -> descriptor.  Going through
    # ``atomic_model.has_message_passing()`` is important for composite
    # atomic models (e.g. ``LinearAtomicModel`` in DP-ZBL) which don't
    # expose a single ``.descriptor`` but do aggregate the flag across
    # their sub-models.  ``descriptor.has_message_passing()`` is the
    # fallback for any future wrapper that lacks the higher-level
    # methods.
    def _probe_has_message_passing(obj: object) -> bool | None:
        if obj is None or not hasattr(obj, "has_message_passing"):
            return None
        try:
            return bool(obj.has_message_passing())
        except (AttributeError, NotImplementedError):
            return None

    result: bool | None = None
    for obj in (
        model,
        getattr(model, "atomic_model", None),
        getattr(getattr(model, "atomic_model", None), "descriptor", None),
    ):
        result = _probe_has_message_passing(obj)
        if result is not None:
            break
    meta["has_message_passing"] = result if result is not None else False

    # Which input schema the compiled AOTI forward consumes:
    #   "nlist" → dense quartet (extended_coord, extended_atype, nlist, mapping)
    #   "graph" → NeighborGraph (atype, n_node, edge_index, edge_vec, edge_mask)
    # The C++ loader branches on this to build the matching inputs.
    meta["lower_input_kind"] = lower_kind
    meta["graph_edge_dtype"] = _graph_edge_dtype(model, lower_kind)
    return meta


def serialize_from_file(model_file: str) -> dict:
    """Serialize a .pte or .pt2 model file to a dictionary.

    Reads the model dict stored in the model archive.

    Parameters
    ----------
    model_file : str
        The model file to be serialized (.pte or .pt2).

    Returns
    -------
    dict
        The serialized model data.  If the archive contains
        ``model_def_script.json`` (training config), it is included
        under the ``"model_def_script"`` key.
    """
    if model_file.endswith(".pt2"):
        return _serialize_from_file_pt2(model_file)
    else:
        return _serialize_from_file_pte(model_file)


def _serialize_from_file_pte(model_file: str) -> dict:
    """Serialize a .pte model file to a dictionary."""
    extra_files = {"model.json": "", "model_def_script.json": ""}
    torch.export.load(model_file, extra_files=extra_files)
    model_dict = json.loads(extra_files["model.json"])
    model_dict = _json_to_numpy(model_dict)
    if extra_files["model_def_script.json"]:
        model_dict["model_def_script"] = json.loads(
            extra_files["model_def_script.json"]
        )
    return model_dict


def _serialize_from_file_pt2(model_file: str) -> dict:
    """Serialize a .pt2 model file to a dictionary.

    Reads the model dict stored in the ``model/extra/`` directory of the
    ``.pt2`` ZIP archive.
    """
    import zipfile

    model_json_entry = PT2_EXTRA_PREFIX + "model.json"
    model_def_script_entry = PT2_EXTRA_PREFIX + "model_def_script.json"
    with zipfile.ZipFile(model_file, "r") as zf:
        names = zf.namelist()
        if model_json_entry not in names:
            raise ValueError(
                f"Invalid .pt2 file '{model_file}': missing '{model_json_entry}'"
            )
        model_json = zf.read(model_json_entry).decode("utf-8")
        model_def_script_json = ""
        if model_def_script_entry in names:
            model_def_script_json = zf.read(model_def_script_entry).decode("utf-8")
    model_dict = json.loads(model_json)
    model_dict = _json_to_numpy(model_dict)
    if model_def_script_json:
        model_dict["model_def_script"] = json.loads(model_def_script_json)
    return model_dict


@contextlib.contextmanager
def _cuda_infer_at_least_2() -> Iterator[None]:
    """Pin ``DP_CUDA_INFER`` to at least 2 for the duration of a trace.

    Level 2 emits the inference pipeline as explicit descriptor, fitting,
    descriptor-backward, and CSR force/virial custom operators. These operators
    remain opaque through ``torch.export``. The level-1 autograd lower can
    decompose the analytic backward to aten, while level 0 selects the
    untraceable reference tabulation. Level 2 degrades internally when an
    operator is unavailable or ineligible, so it is a safe floor for graph
    export.
    """
    from deepmd.kernels.utils import (
        cuda_infer_level,
    )

    saved = os.environ.get("DP_CUDA_INFER")
    if cuda_infer_level() < 2:
        os.environ["DP_CUDA_INFER"] = "2"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("DP_CUDA_INFER", None)
        else:
            os.environ["DP_CUDA_INFER"] = saved


def _resolve_lower_kind(model_file: str, data: dict, lower_kind: str) -> str:
    """Resolve ``lower_kind="auto"`` to a concrete lower-forward schema.

    ``"auto"`` selects the graph lower for a graph-lower model whose graph
    implementation is exportable to ``.pt2`` and the dense nlist lower for
    everything else. An explicit ``"nlist"`` / ``"graph"`` is returned
    unchanged.
    """
    if lower_kind != "auto":
        return lower_kind
    if not model_file.endswith(".pt2") or data["model"].get("type") == "spin_ener":
        return "nlist"
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.train.training import (
        _model_uses_graph_lower,
    )

    model = BaseModel.deserialize(data["model"])
    if _model_uses_graph_lower(model) and _supports_graph_export(model):
        from deepmd.kernels.cuda.dpa1.canonical import (
            canonical_model_eligible,
        )

        return "dpa1_canonical" if canonical_model_eligible(model) else "graph"
    return "nlist"


def deserialize_to_file(
    model_file: str,
    data: dict,
    model_json_override: dict | None = None,
    do_atomic_virial: bool = False,
    lower_kind: str = "nlist",
) -> None:
    """Deserialize a dictionary to a .pte or .pt2 model file.

    Builds a pt_expt model from the dict, traces it via make_fx,
    exports with dynamic shapes, and saves.

    Parameters
    ----------
    model_file : str
        The model file to be saved (.pte or .pt2).
    data : dict
        The dictionary to be deserialized (same format as dpmodel's
        serialize output, with "model" and optionally "model_def_script" keys).
        If ``data["model_def_script"]`` is present, it is embedded in the
        output so that ``--use-pretrain-script`` can extract descriptor/fitting
        params at finetune time.
    model_json_override : dict or None
        If provided, this dict is stored in model.json instead of ``data``.
        Used by ``dp compress`` to store the compressed model dict while
        tracing the uncompressed model (make_fx cannot trace custom ops).
    do_atomic_virial : bool
        If True, export with per-atom virial correction (3 extra backward
        passes, ~2.5x slower).  Default False for best performance. Forced True
        for a graph lower, whose LAMMPS Kokkos consumer always reads it.
    lower_kind : str
        Which lower-forward schema the compiled AOTI graph consumes:
        ``"nlist"`` (default) traces the dense quartet
        (``extended_coord``/``extended_atype``/``nlist``/``mapping``);
        ``"graph"`` traces the NeighborGraph schema
        (``atype``/``n_node``/``edge_index``/``edge_vec``/``edge_mask`` and
        the destination/source CSR views) with a DYNAMIC edge axis ``E``
        (``Dim("nedge", min=2)``), so the artifact accepts any system size.
        ``"auto"`` (used by ``convert-backend``)
        resolves to ``"graph"`` for an exportable graph-lower ``.pt2`` and
        ``"nlist"`` otherwise (see :func:`_resolve_lower_kind`). A graph lower always
        preserves the fused inference operators (``DP_CUDA_INFER >= 2``) and
        the per-atom virial.
        The selected schema is recorded as ``lower_input_kind`` in
        ``metadata.json``.
    """
    lower_kind = _resolve_lower_kind(model_file, data, lower_kind)
    # A graph lower deploys the fused inference pipeline. The trace runs at
    # DP_CUDA_INFER >= 2 so the analytic backward and CSR scatter remain custom
    # operators, while the per-atom virial is mandatory for the LAMMPS Kokkos
    # consumer.
    if lower_kind in ("graph", "dpa1_canonical"):
        do_atomic_virial = True
        ctx: contextlib.AbstractContextManager = _cuda_infer_at_least_2()
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if model_file.endswith(".pt2"):
            _deserialize_to_file_pt2(
                model_file,
                data,
                model_json_override,
                do_atomic_virial,
                lower_kind,
            )
        else:
            _deserialize_to_file_pte(
                model_file,
                data,
                model_json_override,
                do_atomic_virial,
                lower_kind,
            )


def _trace_and_export(
    data: dict,
    model_json_override: dict | None = None,
    with_comm_dict: bool = False,
    do_atomic_virial: bool = False,
    lower_kind: str = "nlist",
) -> tuple:
    """Common logic: build model, trace, export.

    Parameters
    ----------
    data
        Serialized model dict (with "model" and optionally
        "model_def_script" keys).
    model_json_override
        Optional alternate dict to embed as model.json (used by
        ``dp compress`` to store the compressed model dict while
        tracing the uncompressed one).
    with_comm_dict
        If True, trace ``forward_common_lower_exportable_with_comm``
        instead of the regular variant. The resulting exported program
        accepts 8 additional positional comm tensors (``send_list``,
        ``send_proc``, ``recv_proc``, ``send_num``, ``recv_num``,
        ``communicator``, ``nlocal``, ``nghost``) used by the pt_expt
        Repflow/Repformer override to drive MPI ghost-atom exchange.
        Only valid for models that need cross-rank ghost-feature exchange
        (see ``_needs_with_comm_artifact``).
    do_atomic_virial
        If True, the traced graph computes per-atom virial (extra
        autograd.grad backward passes); off by default to keep .pt2
        inference fast. Mirrors PR #5407 in upstream master.
    lower_kind
        ``"nlist"`` (default) traces the dense quartet forward; ``"graph"``
        traces ``forward_lower_graph_exportable`` over the NeighborGraph schema
        with a dynamic edge axis. Recorded as ``lower_input_kind`` in metadata.

    Returns
    -------
    tuple
        ``(exported, metadata, data_for_json, output_keys)``.
    """
    from copy import (
        deepcopy,
    )

    import deepmd.pt_expt.utils.env as _env
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )

    target_device = _env.DEVICE

    # Detect spin model
    is_spin = data["model"].get("type") == "spin_ener"

    # 1. Deserialize model on CPU for make_fx tracing.
    # make_fx with _allow_non_fake_inputs=True keeps real model parameters;
    # on CUDA the autograd engine requires CUDA streams for those real
    # tensors during torch.autograd.grad, but proxy-tensor dispatch doesn't
    # set streams up → assertion failure.  Tracing on CPU avoids this.
    if is_spin:
        from deepmd.pt_expt.model.spin_model import (
            SpinModel,
        )

        model = SpinModel.deserialize(data["model"])
    else:
        model = BaseModel.deserialize(data["model"])
    model.to("cpu")
    model.eval()
    if lower_kind == "graph" and not _supports_graph_export(model):
        raise NotImplementedError(
            "graph-form export of a compressed descriptor requires its "
            "float32 fused graph operator; use lower_kind='nlist' for this model"
        )

    # Autotune checkpoint-specific custom-kernel launch tables on the target
    # GPU before tracing. The model itself remains on CPU for tracing.
    from deepmd.kernels.autotune import (
        run_autotune,
    )

    run_autotune(model, target_device)

    # 2. Collect metadata
    metadata = _collect_metadata(
        model,
        is_spin=is_spin,
        lower_kind=lower_kind,
    )

    # Graph-form exports use a dynamic edge axis and an energy-model contract.
    if lower_kind in ("graph", "dpa1_canonical"):
        import math

        check_graph_trace_torch_version(model)
        if is_spin:
            raise NotImplementedError(
                "graph-form .pt2 export is not supported for spin models"
            )
        if with_comm_dict:
            raise NotImplementedError(
                "graph-form .pt2 export does not support the with-comm artifact "
                "required for multi-rank message passing"
            )
        canonical = lower_kind == "dpa1_canonical"
        required_method = (
            "forward_lower_canonical_graph_exportable"
            if canonical
            else "forward_lower_graph_exportable"
        )
        if not hasattr(model, required_method):
            raise NotImplementedError(
                f"model {type(model).__name__} has no {required_method}"
            )
        if canonical:
            from deepmd.kernels.cuda.dpa1.canonical import (
                canonical_model_eligible,
            )

            if not canonical_model_eligible(model):
                raise NotImplementedError(
                    "compact canonical export requires an eligible compressed "
                    "DPA1 energy model"
                )

        nloc_sample = 7
        nnei = sum(model.get_sel())
        e_sample = math.ceil(1.25 * nloc_sample * nnei)
        if canonical:
            sample_inputs = build_synthetic_canonical_graph_inputs(
                model,
                e_sample,
                device=torch.device("cpu"),
            )
            traced = model.forward_lower_canonical_graph_exportable(
                *sample_inputs,
                do_atomic_virial=do_atomic_virial,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )
            dynamic_shapes = _build_canonical_graph_dynamic_shapes(*sample_inputs)
        else:
            edge_dtype = (
                torch.float32
                if metadata["graph_edge_dtype"] == "float32"
                else torch.float64
            )
            sample_inputs = build_synthetic_graph_inputs(
                model,
                e_max=e_sample,
                nframes=2,
                nloc=nloc_sample,
                dtype=torch.float64,
                edge_dtype=edge_dtype,
                device=torch.device("cpu"),
            )
            traced = model.forward_lower_graph_exportable(
                *sample_inputs[:10],
                fparam=sample_inputs[10],
                aparam=sample_inputs[11],
                do_atomic_virial=do_atomic_virial,
                charge_spin=sample_inputs[12],
                destination_sorted=True,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )
            dynamic_shapes = _build_graph_dynamic_shapes(*sample_inputs)
        sample_out = traced(*sample_inputs)
        output_keys = list(sample_out.keys())
        exported = torch.export.export(
            traced,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

        # Neutralise shape-guard assertion nodes on the dynamic edge axis.
        # ``prefer_deferred_runtime_asserts_over_guards=True`` converts the
        # symbolic-shape guards discovered while tracing into deferred
        # ``aten._assert_scalar`` nodes. Replacing each condition with ``True``
        # preserves graph structure while allowing the AOTI artifact to
        # generalise across edge counts.
        _strip_shape_assertions(exported.graph_module)

        if target_device.type != "cpu":
            from torch.export.passes import (
                move_to_device_pass,
            )

            exported = move_to_device_pass(exported, target_device)

        metadata["do_atomic_virial"] = do_atomic_virial
        # The AOTI forward accepts any edge
        # count, so there is no ``edge_capacity`` to persist. The C++ / Python
        # conversion hub builds the carry-all graph at its exact (tight) edge
        # count and feeds it straight through.

        json_source = model_json_override if model_json_override is not None else data
        data_for_json = deepcopy(json_source)
        data_for_json = _numpy_to_json_serializable(data_for_json)

        return exported, metadata, data_for_json, output_keys

    # 3. Create sample inputs on CPU for tracing
    # torch.export's duck-sizing unifies dimensions with the same sample value,
    # so nframes must differ from every other dimension in the sample tensors.
    # We first build with nframes=2, collect all non-batch dimension sizes,
    # then rebuild if there is a collision.
    _orig_device = _env.DEVICE
    _env.DEVICE = torch.device("cpu")
    try:
        if with_comm_dict:
            # The pt_expt parallel-mode override (in pt's repflows.py
            # line 593 too) uses ``squeeze(0)`` / ``unsqueeze(0)`` on
            # ``node_ebd`` and so requires ``nframes == 1``.  LAMMPS
            # always drives inference with one frame, so this is the
            # only realistic shape — and we mark dim 0 static in
            # ``_build_dynamic_shapes`` to match.
            nframes = 1
            sample_inputs = _make_sample_inputs(
                model,
                nframes=nframes,
                has_spin=is_spin,
            )
        else:
            nframes = 2
            sample_inputs = _make_sample_inputs(
                model,
                nframes=nframes,
                has_spin=is_spin,
            )
            # Collect all dimension sizes except dim-0 (nframes) from every tensor
            other_dims: set[int] = set()
            for t in sample_inputs:
                if t is not None:
                    other_dims.update(t.shape[1:])
            while nframes in other_dims:
                nframes += 1
            if nframes != 2:
                sample_inputs = _make_sample_inputs(
                    model, nframes=nframes, has_spin=is_spin
                )
    finally:
        _env.DEVICE = _orig_device

    if is_spin:
        (
            ext_coord,
            ext_atype,
            ext_spin,
            nlist_t,
            mapping_t,
            fparam,
            aparam,
            charge_spin,
        ) = sample_inputs
    else:
        (
            ext_coord,
            ext_atype,
            nlist_t,
            mapping_t,
            fparam,
            aparam,
            charge_spin,
        ) = sample_inputs

    # 3b. Build comm-tensor sample inputs when tracing the with-comm
    # variant (only valid for GNN models). The actual values don't
    # matter for tracing — only that they're valid tensors of the right
    # shape and dtype.  See ``_make_comm_sample_inputs``.
    if with_comm_dict:
        # Load libdeepmd_op_pt.so and register border_op fake/autograd
        # metadata now — deferred from import time so normal utils imports
        # don't force-load the op library and break fake-op ordering.
        from deepmd.pt_expt.utils.comm import (
            ensure_comm_registered,
        )

        ensure_comm_registered()
        if not _needs_with_comm_artifact(model):
            raise ValueError(
                "with_comm_dict=True requested but the model's descriptor "
                "does not need cross-rank message passing "
                "(has_message_passing_across_ranks() is False) — "
                "there's nothing to compile."
            )
        nloc_sample = nlist_t.shape[1]
        nall_sample = ext_atype.shape[1]
        nghost_sample = nall_sample - nloc_sample
        comm_inputs = _make_comm_sample_inputs(
            nloc=nloc_sample,
            nghost=nghost_sample,
            device=torch.device("cpu"),
        )
        sample_inputs = sample_inputs + comm_inputs

    # 4. Trace via make_fx on CPU.
    # This decomposes torch.autograd.grad into aten ops so the resulting
    # GraphModule no longer contains autograd calls.
    log.info("Tracing the lower graph on CPU (make_fx)...")
    if is_spin:
        if with_comm_dict:
            traced = model.forward_common_lower_exportable_with_comm(
                ext_coord,
                ext_atype,
                ext_spin,
                nlist_t,
                mapping_t,
                fparam,
                aparam,
                charge_spin,
                *comm_inputs,
                do_atomic_virial=do_atomic_virial,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )
        else:
            traced = model.forward_common_lower_exportable(
                ext_coord,
                ext_atype,
                ext_spin,
                nlist_t,
                mapping_t,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                charge_spin=charge_spin,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )
        # 5. Extract output keys from the CPU-traced module.
        sample_out = traced(*sample_inputs)
    else:
        if with_comm_dict:
            traced = model.forward_common_lower_exportable_with_comm(
                ext_coord,
                ext_atype,
                nlist_t,
                mapping_t,
                fparam,
                aparam,
                charge_spin,
                *comm_inputs,
                do_atomic_virial=do_atomic_virial,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )
        else:
            traced = model.forward_common_lower_exportable(
                ext_coord,
                ext_atype,
                nlist_t,
                mapping_t,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                charge_spin=charge_spin,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )
        # 5. Extract output keys from the CPU-traced module.
        sample_out = traced(*sample_inputs)

    output_keys = list(sample_out.keys())

    # 6. Export on CPU.
    # make_fx on CPU bakes device='cpu' into tensor-creation ops in the
    # graph.  Exporting on CPU keeps devices consistent; we move the
    # ExportedProgram to the target device afterwards via the official
    # move_to_device_pass (avoids FakeTensor device-propagation errors).
    log.info("Exporting the traced graph (torch.export)...")
    dynamic_shapes = _build_dynamic_shapes(
        *sample_inputs,
        has_spin=is_spin,
        with_comm_dict=with_comm_dict,
        model_nnei=sum(model.get_sel()),
    )
    exported = torch.export.export(
        traced,
        sample_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

    if is_spin:
        # The spin model's atom-doubling slice patterns depend on
        # (nall - nloc), producing guards like Ne(nall, nloc).  These are
        # spurious — the model is correct when nall == nloc (NoPBC).
        # Non-spin models don't emit shape guards because the short-circuit
        # order in `_format_nlist` (dpmodel) keeps the dynamic `nnei` axis
        # free of symbolic comparisons when `extra_nlist_sort=True`
        # (see `forward_common_lower_exportable` in pt_expt/model/make_model.py).
        _strip_shape_assertions(exported.graph_module)

    # 7. Move the exported program to the target device if needed.
    if target_device.type != "cpu":
        from torch.export.passes import (
            move_to_device_pass,
        )

        exported = move_to_device_pass(exported, target_device)

    # 8. Record export-time config in metadata
    metadata["do_atomic_virial"] = do_atomic_virial

    # 9. Prepare JSON-serializable model dict
    json_source = model_json_override if model_json_override is not None else data
    data_for_json = deepcopy(json_source)
    data_for_json = _numpy_to_json_serializable(data_for_json)

    return exported, metadata, data_for_json, output_keys


def _deserialize_to_file_pte(
    model_file: str,
    data: dict,
    model_json_override: dict | None = None,
    do_atomic_virial: bool = False,
    lower_kind: str = "nlist",
) -> None:
    """Deserialize a dictionary to a .pte model file."""
    exported, metadata, data_for_json, output_keys = _trace_and_export(
        data,
        model_json_override,
        do_atomic_virial=do_atomic_virial,
        lower_kind=lower_kind,
    )

    model_def_script = data.get("model_def_script") or {}
    metadata["output_keys"] = output_keys
    extra_files = {
        "metadata.json": json.dumps(metadata),
        "model_def_script.json": json.dumps(model_def_script),
        "model.json": json.dumps(data_for_json, separators=(",", ":")),
    }

    torch.export.save(exported, model_file, extra_files=extra_files)


def _deserialize_to_file_pt2(
    model_file: str,
    data: dict,
    model_json_override: dict | None = None,
    do_atomic_virial: bool = False,
    lower_kind: str = "nlist",
) -> None:
    """Deserialize a dictionary to a .pt2 model file (AOTInductor).

    Uses torch._inductor.aoti_compile_and_package to compile the exported
    program into a .pt2 package (ZIP archive with compiled shared libraries),
    then embeds metadata into the archive.

    For models whose descriptor reports
    ``has_message_passing_across_ranks() == True`` (DPA2, DPA3 with
    ``use_loc_mapping=False``, or hybrids wrapping such children),
    compiles a SECOND ``with-comm`` artifact and packs it alongside the
    regular one. The ``with-comm`` variant accepts comm-dict tensors as
    additional positional inputs and drives MPI ghost-atom exchange via
    ``deepmd_export::border_op``. The C++ ``DeepPotPTExpt`` loader picks
    the artifact based on the LAMMPS rank count at runtime.

    Layout inside the .pt2 ZIP (PyTorch 2.11 strict layout):
        regular   →  artifact at ``model/`` (AOTInductor's own layout)
        with-comm →  ``model/extra/forward_lower_with_comm.pt2`` (nested ZIP)
        metadata  →  ``model/extra/metadata.json`` with
                     ``has_comm_artifact`` flag. The C++ reader matches
                     by ``/``-delimited suffix so the legacy root-level
                     ``extra/`` layout still loads.

    Old .pt2 files (pre-this-change) lack ``has_comm_artifact`` so the
    C++ loader must default to ``False`` when the field is missing.
    """
    import os
    import tempfile
    import zipfile

    from torch._inductor import (
        aoti_compile_and_package,
    )

    # First artifact: regular (no comm). Always produced.
    exported, metadata, data_for_json, output_keys = _trace_and_export(
        data,
        model_json_override,
        do_atomic_virial=do_atomic_virial,
        lower_kind=lower_kind,
    )
    metadata["output_keys"] = output_keys

    # On CUDA, aggressive kernel fusion (default realize_opcount_threshold=30)
    # causes NaN in the backward pass (force/virial) of attention-based
    # descriptors (DPA1, DPA2). Setting threshold=0 prevents fusion and
    # avoids the NaN. Only applied on CUDA; CPU compilation is unaffected.
    #
    # ``assert_indirect_indexing`` (default True) makes inductor emit an
    # ``AOTI_TORCH_CHECK`` bounds assertion for every indirect (data-dependent)
    # index. In the CPU-vectorised codegen for DPA4/SeZM's per-node
    # gather/scatter (the descriptor broadcasts a per-node value across its
    # edges), inductor mis-hoists that assertion ABOVE the declaration of the
    # index temporary, emitting C++ that references an undeclared ``tmpN`` and
    # fails to compile ("use of undeclared identifier"). The asserted indices
    # are loop counters that are in-bounds by construction, so the check is
    # redundant; disabling it removes the broken assertion while leaving
    # vectorisation (and therefore inference throughput) untouched.
    #
    # NOTE: ``torch._inductor.config`` is a process-wide singleton. The
    # save/restore pattern here is NOT thread-safe — concurrent AOTInductor
    # compilations from multiple threads would race on this global. Callers
    # must serialise ``.pt2`` exports if running under a thread pool.
    # Processes are fine (each has its own inductor config).
    import deepmd.pt_expt.utils.env as _env
    from deepmd.pt.utils.compile_compat import (
        build_inductor_compile_options,
        patch_inductor_force_int64_indexing,
    )

    is_cuda = _env.DEVICE.type == "cuda"
    # Force int64 tensor indexing so the flattened index of a large
    # data-dependent tensor never wraps past 2**31 into an illegal address.
    patch_inductor_force_int64_indexing()
    # The AOTInductor freeze must use the same Inductor lockdown as the
    # pt-backend compile -- most importantly ``triton.max_tiles = 1``, which
    # keeps the data-dependent edge / node axis on the x launch dimension
    # (limit 2**31-1) rather than a 2-D y/z tile (limit 65535). Without it a
    # compressed graph .pt2 (its level-1 lower carries an aten glue over the
    # ``(n_node, NG * axis)`` descriptor) launches an out-of-range grid and
    # fails at runtime with a CUDA "invalid argument" once that tensor exceeds
    # 2**22 elements. The two deepmd-specific relaxations layer on top.
    aoti_configs = build_inductor_compile_options(inference=True)
    aoti_configs["assert_indirect_indexing"] = False
    if is_cuda:
        aoti_configs["realize_opcount_threshold"] = 0
    log.info(
        "Compiling the AOTInductor package for %s (the slowest freeze stage; "
        "typically several minutes)...",
        _env.DEVICE,
    )
    aoti_compile_and_package(
        exported, package_path=model_file, inductor_configs=aoti_configs
    )

    # Second artifact: with-comm. Only for descriptors whose message
    # passing extends across rank boundaries. The flag was computed
    # from the model in ``_collect_metadata`` and is already in
    # ``metadata`` here.
    has_comm_artifact = bool(metadata.get("has_comm_artifact"))
    with_comm_bytes: bytes | None = None
    with_comm_output_keys: list[str] | None = None
    if has_comm_artifact:
        exported_wc, _meta_wc, _data_wc, with_comm_output_keys = _trace_and_export(
            data,
            model_json_override,
            with_comm_dict=True,
            do_atomic_virial=do_atomic_virial,
        )
        with tempfile.TemporaryDirectory() as td:
            wc_path = os.path.join(td, "forward_lower_with_comm.pt2")
            aoti_compile_and_package(
                exported_wc, package_path=wc_path, inductor_configs=aoti_configs
            )
            with open(wc_path, "rb") as f:
                with_comm_bytes = f.read()
        # The output keys are identical between the two artifacts (same
        # forward_lower output dict); record only one set in metadata.
        # If they ever diverge we'll surface a hard error here.
        if with_comm_output_keys != output_keys:
            raise RuntimeError(
                "with-comm artifact output keys diverge from regular: "
                f"regular={output_keys} vs with_comm={with_comm_output_keys}"
            )

    # Embed metadata + supplementary files into the .pt2 ZIP archive.
    # Entries are placed under ``model/extra/`` so the strict PyTorch
    # 2.11 ``load_pt2`` loader accepts the archive without emitting the
    # "outdated pt2 file" fallback warning.  See the module-level
    # comment on ``PT2_EXTRA_PREFIX`` for the rationale.  The C++ reader
    # (``commonPTExpt.h::read_zip_entry``) accepts both the legacy
    # root-level ``extra/`` layout and the new ``model/extra/`` layout
    # via suffix matching, so the with-comm artifact moves with the
    # rest.
    model_def_script = data.get("model_def_script") or {}
    with zipfile.ZipFile(model_file, "a") as zf:
        zf.writestr(PT2_EXTRA_PREFIX + "metadata.json", json.dumps(metadata))
        zf.writestr(
            PT2_EXTRA_PREFIX + "model_def_script.json",
            json.dumps(model_def_script),
        )
        zf.writestr(
            PT2_EXTRA_PREFIX + "model.json",
            json.dumps(data_for_json, separators=(",", ":")),
        )
        if with_comm_bytes is not None:
            zf.writestr(
                PT2_EXTRA_PREFIX + "forward_lower_with_comm.pt2", with_comm_bytes
            )
