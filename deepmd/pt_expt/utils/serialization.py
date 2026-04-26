# SPDX-License-Identifier: LGPL-3.0-or-later
import ctypes
import json

import numpy as np
import torch

from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.dpmodel.utils.serialization import (
    traverse_model_dict,
)


def _strip_shape_assertions(graph_module: torch.nn.Module) -> None:
    """Remove shape-guard assertion nodes from a spin model's exported graph.

    ``torch.export`` inserts ``aten._assert_scalar`` nodes for symbolic shape
    relationships discovered during tracing.  For the spin model, the atom-
    doubling logic creates slice patterns that depend on ``(nall - nloc)``,
    producing guards like ``Ne(nall, nloc)``.  These guards are spurious: the
    model computes correct results even when ``nall == nloc`` (NoPBC, no ghost
    atoms).

    This function is **only called for spin models** (guarded by ``if is_spin``
    in ``_trace_and_export``).  The assertion messages use opaque symbolic
    variable names (e.g. ``Ne(s22, s96)``) rather than human-readable names,
    so filtering by message content is not reliable.  Since
    ``prefer_deferred_runtime_asserts_over_guards=True`` converts all shape
    guards into these deferred assertions, and the only shape relationships in
    the spin model involve nall/nloc, removing all of them is safe in this
    context.
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


def _has_message_passing(model: torch.nn.Module) -> bool:
    """Detect whether a model's descriptor uses GNN-style message passing.

    GNN descriptors (DPA2 with repformers, DPA3 with repflows) require
    a per-layer ghost-atom MPI exchange when running multi-rank LAMMPS,
    which means a separate ``with-comm`` AOTInductor artifact must be
    compiled.  Non-GNN descriptors (se_e2_a, se_r, se_t, se_t_tebd,
    DPA1, hybrid-of-non-GNN) need only the regular artifact.

    Additional gate: ``use_loc_mapping=True`` GNN models (the default
    for DPA3) keep nlist in local-only indexing, so per-layer ghost
    exchange is meaningless — these get only the regular artifact.
    Multi-rank LAMMPS for GNN requires use_loc_mapping=False.

    Returns False if the descriptor's ``has_message_passing()`` query
    cannot be answered (e.g. linear/zbl/frozen models without a single
    descriptor) — those are assumed local.
    """
    try:
        descriptor = model.atomic_model.descriptor
    except AttributeError:
        return False
    if not hasattr(descriptor, "has_message_passing"):
        return False
    try:
        if not descriptor.has_message_passing():
            return False
    except (AttributeError, NotImplementedError):
        return False
    # Walk into the GNN block (repflows / repformers) to inspect
    # ``use_loc_mapping``. The attribute lives on the block, not on the
    # top-level descriptor wrapper.
    for attr in ("repflows", "repformers"):
        block = getattr(descriptor, attr, None)
        if block is None:
            continue
        if getattr(block, "use_loc_mapping", False):
            return False
    return True


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

    Phase 0 finding: tracing with ``nswap == 0`` causes the dim to
    specialize, so we must use ``nswap >= 1``.  We use ``nswap == 1``
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
        (ext_coord, ext_atype, nlist, mapping, fparam, aparam) or
        (ext_coord, ext_atype, ext_spin, nlist, mapping, fparam, aparam) when has_spin.
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

    if has_spin:
        nall = extended_coord.shape[1]
        ext_spin = torch.zeros(
            nframes, nall, 3, dtype=torch.float64, device=_env.DEVICE
        )
        return ext_coord, ext_atype, ext_spin, nlist_t, mapping_t, fparam, aparam

    return ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam


def _build_dynamic_shapes(
    *sample_inputs: torch.Tensor | None,
    has_spin: bool = False,
    with_comm_dict: bool = False,
) -> tuple:
    """Build dynamic shape specifications for torch.export.

    Marks nframes, nloc and nall as dynamic dimensions so the exported
    program handles arbitrary frame and atom counts.

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
    nall_dim = torch.export.Dim("nall", min=1)
    nloc_dim = torch.export.Dim("nloc", min=1)

    if has_spin:
        # (ext_coord, ext_atype, ext_spin, nlist, mapping, fparam, aparam)
        fparam = sample_inputs[5]
        aparam = sample_inputs[6]
        base = (
            {0: nframes_dim, 1: nall_dim},  # extended_coord: (nframes, nall, 3)
            {0: nframes_dim, 1: nall_dim},  # extended_atype: (nframes, nall)
            {0: nframes_dim, 1: nall_dim},  # extended_spin: (nframes, nall, 3)
            {0: nframes_dim, 1: nloc_dim},  # nlist: (nframes, nloc, nnei)
            {0: nframes_dim, 1: nall_dim},  # mapping: (nframes, nall)
            {0: nframes_dim} if fparam is not None else None,  # fparam
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,  # aparam
        )
    else:
        # (ext_coord, ext_atype, nlist, mapping, fparam, aparam)
        fparam = sample_inputs[4]
        aparam = sample_inputs[5]
        base = (
            {0: nframes_dim, 1: nall_dim},  # extended_coord: (nframes, nall, 3)
            {0: nframes_dim, 1: nall_dim},  # extended_atype: (nframes, nall)
            {0: nframes_dim, 1: nloc_dim},  # nlist: (nframes, nloc, nnei)
            {0: nframes_dim, 1: nall_dim},  # mapping: (nframes, nall)
            {0: nframes_dim} if fparam is not None else None,  # fparam
            {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,  # aparam
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
    return base + (None, None, None, None, None, None, None, None)


def _collect_metadata(model: torch.nn.Module, is_spin: bool = False) -> dict:
    """Collect metadata from the model for C++ inference.

    This metadata is stored as ``metadata.json`` in both .pt2 and .pte archives.
    Training config is stored separately in ``model_def_script.json``.  C++ reads
    flat JSON fields because compiling model API methods as AOTInductor
    entry points is impractical (~12 s per trivial function) and string
    outputs (``get_type_map``) cannot be expressed as tensor I/O.

    The ``fitting_output_defs`` list is also included so that
    ``ModelOutputDef`` can be reconstructed without loading the full model.
    """
    fitting_output_def = model.atomic_output_def()
    fitting_output_defs = []
    for vdef in fitting_output_def.get_data().values():
        fitting_output_defs.append(
            {
                "name": vdef.name,
                "shape": list(vdef.shape),
                "reducible": vdef.reducible,
                "r_differentiable": vdef.r_differentiable,
                "c_differentiable": vdef.c_differentiable,
                "atomic": vdef.atomic,
                "category": vdef.category,
                "r_hessian": vdef.r_hessian,
                "magnetic": vdef.magnetic,
                "intensive": vdef.intensive,
            }
        )
    meta = {
        "type_map": model.get_type_map(),
        "rcut": model.get_rcut(),
        "sel": model.get_sel(),
        "dim_fparam": model.get_dim_fparam(),
        "dim_aparam": model.get_dim_aparam(),
        "mixed_types": model.mixed_types(),
        "has_default_fparam": model.has_default_fparam(),
        "default_fparam": model.get_default_fparam(),
        "fitting_output_defs": fitting_output_defs,
        "is_spin": is_spin,
    }
    if is_spin:
        meta["ntypes_spin"] = model.spin.get_ntypes_spin()
        meta["use_spin"] = [bool(v) for v in model.spin.use_spin]
    # Record whether the model uses GNN-style message passing.  When
    # True, .pt2 deserialization compiles a second ``with-comm`` artifact
    # so multi-rank LAMMPS can drive ghost-atom MPI exchange through
    # the model.  C++ DeepPotPTExpt branches on this flag at load time.
    meta["has_message_passing"] = _has_message_passing(model)
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

    Reads the model dict stored in the extra/ directory of the .pt2 ZIP archive.
    """
    import zipfile

    with zipfile.ZipFile(model_file, "r") as zf:
        if "extra/model.json" not in zf.namelist():
            raise ValueError(
                f"Invalid .pt2 file '{model_file}': missing 'extra/model.json'"
            )
        model_json = zf.read("extra/model.json").decode("utf-8")
        model_def_script_json = ""
        if "extra/model_def_script.json" in zf.namelist():
            model_def_script_json = zf.read("extra/model_def_script.json").decode(
                "utf-8"
            )
    model_dict = json.loads(model_json)
    model_dict = _json_to_numpy(model_dict)
    if model_def_script_json:
        model_dict["model_def_script"] = json.loads(model_def_script_json)
    return model_dict


def deserialize_to_file(
    model_file: str,
    data: dict,
    model_json_override: dict | None = None,
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
    """
    if model_file.endswith(".pt2"):
        _deserialize_to_file_pt2(model_file, data, model_json_override)
    else:
        _deserialize_to_file_pte(model_file, data, model_json_override)


def _trace_and_export(
    data: dict,
    model_json_override: dict | None = None,
    with_comm_dict: bool = False,
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
        Only valid for GNN models (see ``_has_message_passing``).
    Returns (exported, metadata, data_for_json, output_keys).
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

    # 2. Collect metadata
    metadata = _collect_metadata(model, is_spin=is_spin)

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
        ext_coord, ext_atype, ext_spin, nlist_t, mapping_t, fparam, aparam = (
            sample_inputs
        )
    else:
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = sample_inputs

    # 3b. Build comm-tensor sample inputs when tracing the with-comm
    # variant (only valid for GNN models). The actual values don't
    # matter for tracing — only that they're valid tensors of the right
    # shape and dtype.  See ``_make_comm_sample_inputs``.
    if with_comm_dict:
        if not metadata.get("has_message_passing"):
            raise ValueError(
                "with_comm_dict=True requested but model has no GNN "
                "message-passing descriptor — there's nothing to compile."
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
                *comm_inputs,
                do_atomic_virial=True,
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
                do_atomic_virial=True,
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
                *comm_inputs,
                do_atomic_virial=True,
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
                do_atomic_virial=True,
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
    dynamic_shapes = _build_dynamic_shapes(
        *sample_inputs,
        has_spin=is_spin,
        with_comm_dict=with_comm_dict,
    )
    exported = torch.export.export(
        traced,
        sample_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

    if is_spin:
        # torch.export re-introduces shape-guard assertions even when
        # the make_fx graph has none.  The spin model's atom-doubling
        # logic creates slice patterns that depend on (nall - nloc),
        # producing guards like Ne(nall, nloc).  These guards are
        # spurious: the model is correct when nall == nloc (NoPBC).
        # Strip them from the exported graph so the model can be
        # used with any valid nall >= nloc.
        _strip_shape_assertions(exported.graph_module)

    # 7. Move the exported program to the target device if needed.
    if target_device.type != "cpu":
        from torch.export.passes import (
            move_to_device_pass,
        )

        exported = move_to_device_pass(exported, target_device)

    # 8. Prepare JSON-serializable model dict
    json_source = model_json_override if model_json_override is not None else data
    data_for_json = deepcopy(json_source)
    data_for_json = _numpy_to_json_serializable(data_for_json)

    return exported, metadata, data_for_json, output_keys


def _deserialize_to_file_pte(
    model_file: str,
    data: dict,
    model_json_override: dict | None = None,
) -> None:
    """Deserialize a dictionary to a .pte model file."""
    exported, metadata, data_for_json, output_keys = _trace_and_export(
        data, model_json_override
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
) -> None:
    """Deserialize a dictionary to a .pt2 model file (AOTInductor).

    Uses torch._inductor.aoti_compile_and_package to compile the exported
    program into a .pt2 package (ZIP archive with compiled shared libraries),
    then embeds metadata into the archive.

    For GNN models (descriptor.has_message_passing() is True), compiles
    a SECOND ``with-comm`` artifact and packs it alongside the regular
    one.  The ``with-comm`` variant accepts comm-dict tensors as
    additional positional inputs and drives MPI ghost-atom exchange via
    ``deepmd_export::border_op``.  The C++ ``DeepPotPTExpt`` loader picks
    the artifact based on the LAMMPS rank count at runtime.

    Layout inside the .pt2 ZIP:
        regular   →  artifact at the top of the archive (existing layout)
        with-comm →  ``extra/forward_lower_with_comm.pt2`` (nested ZIP)
        metadata  →  ``extra/metadata.json`` with ``has_message_passing``
                     and ``has_comm_artifact`` flags.

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
        data, model_json_override
    )
    aoti_compile_and_package(exported, package_path=model_file)
    metadata["output_keys"] = output_keys

    # Second artifact: with-comm. Only for GNN models.
    has_comm_artifact = bool(metadata.get("has_message_passing"))
    metadata["has_comm_artifact"] = has_comm_artifact
    with_comm_bytes: bytes | None = None
    with_comm_output_keys: list[str] | None = None
    if has_comm_artifact:
        exported_wc, _meta_wc, _data_wc, with_comm_output_keys = _trace_and_export(
            data,
            model_json_override,
            with_comm_dict=True,
        )
        with tempfile.TemporaryDirectory() as td:
            wc_path = os.path.join(td, "forward_lower_with_comm.pt2")
            aoti_compile_and_package(exported_wc, package_path=wc_path)
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

    # Embed metadata + supplementary files into the .pt2 ZIP archive
    model_def_script = data.get("model_def_script") or {}
    with zipfile.ZipFile(model_file, "a", zipfile.ZIP_STORED) as zf:
        zf.writestr("extra/metadata.json", json.dumps(metadata))
        zf.writestr("extra/model_def_script.json", json.dumps(model_def_script))
        zf.writestr(
            "extra/model.json",
            json.dumps(data_for_json, separators=(",", ":")),
        )
        if with_comm_bytes is not None:
            zf.writestr("extra/forward_lower_with_comm.pt2", with_comm_bytes)
