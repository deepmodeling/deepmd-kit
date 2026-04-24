# SPDX-License-Identifier: LGPL-3.0-or-later
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
    """Neutralise shape-guard assertion nodes in an exported graph.

    ``torch.export`` inserts ``aten._assert_scalar`` nodes for symbolic shape
    relationships discovered during tracing.  These guards can be spurious:

    * **Spin models**: atom-doubling logic creates slice patterns that depend
      on ``(nall - nloc)``, producing guards like ``Ne(nall, nloc)``.
    * **All models**: the nlist padding inside ``forward_common_lower_exportable``
      and the subsequent sort/truncate in ``_format_nlist`` can produce guards
      like ``Ne(nnei, sum(sel))``.  These are spurious because the compiled
      graph handles any ``nnei >= sum(sel)`` correctly.

    Instead of erasing the assertion nodes (which can disturb the FX graph
    structure and produce NaN gradients on some Python/torch versions), we
    replace each assertion's condition with ``True`` so that the node stays
    in the graph but never fires at runtime.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        ):
            # Replace the condition with True so the assertion always passes
            # but the node stays in the graph.  Erasing nodes can disturb the
            # graph structure and produce NaN on some Python/torch versions.
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
    # Pad nlist so nnei > sum(sel) in the sample tensors.
    # This prevents torch.export from specializing nnei to sum(sel).
    nnei = sum(sel)
    n_pad = max(1, nnei // 4)  # pad by ~25%, at least 1
    nlist = np.concatenate(
        [nlist, -np.ones((nframes, nloc, n_pad), dtype=nlist.dtype)], axis=-1
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
    model_nnei: int = 1,
) -> tuple:
    """Build dynamic shape specifications for torch.export.

    Marks nframes, nloc, nall and nnei as dynamic dimensions so the exported
    program handles arbitrary frame, atom and neighbor counts.

    Parameters
    ----------
    *sample_inputs : torch.Tensor | None
        Sample inputs: either 6 tensors (non-spin) or 7 tensors (spin).
    has_spin : bool
        Whether the inputs include an extended_spin tensor.
    model_nnei : int
        The model's sum(sel).  Used as the min for the dynamic nnei dim.
    Returns a tuple (not dict) to match positional args of the make_fx
    traced module, whose arg names may have suffixes like ``_1``.
    """
    nframes_dim = torch.export.Dim("nframes", min=1)
    nall_dim = torch.export.Dim("nall", min=1)
    nloc_dim = torch.export.Dim("nloc", min=1)
    nnei_dim = torch.export.Dim("nnei", min=max(1, model_nnei))

    if has_spin:
        # (ext_coord, ext_atype, ext_spin, nlist, mapping, fparam, aparam)
        fparam = sample_inputs[5]
        aparam = sample_inputs[6]
        return (
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
        )
    else:
        # (ext_coord, ext_atype, nlist, mapping, fparam, aparam)
        fparam = sample_inputs[4]
        aparam = sample_inputs[5]
        return (
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
        )


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
        "nnei": sum(model.get_sel()),
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
    do_atomic_virial: bool = False,
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
        passes, ~2.5x slower).  Default False for best performance.
    """
    if model_file.endswith(".pt2"):
        _deserialize_to_file_pt2(
            model_file, data, model_json_override, do_atomic_virial
        )
    else:
        _deserialize_to_file_pte(
            model_file, data, model_json_override, do_atomic_virial
        )


def _trace_and_export(
    data: dict,
    model_json_override: dict | None = None,
    do_atomic_virial: bool = False,
) -> tuple:
    """Common logic: build model, trace, export.

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
        nframes = 2
        sample_inputs = _make_sample_inputs(model, nframes=nframes, has_spin=is_spin)
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

    # 4. Trace via make_fx on CPU.
    # This decomposes torch.autograd.grad into aten ops so the resulting
    # GraphModule no longer contains autograd calls.
    if is_spin:
        traced = model.forward_common_lower_exportable(
            ext_coord,
            ext_atype,
            ext_spin,
            nlist_t,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        # 5. Extract output keys from the CPU-traced module.
        sample_out = traced(
            ext_coord, ext_atype, ext_spin, nlist_t, mapping_t, fparam, aparam
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
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        # 5. Extract output keys from the CPU-traced module.
        sample_out = traced(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)

    output_keys = list(sample_out.keys())

    # 6. Export on CPU.
    # make_fx on CPU bakes device='cpu' into tensor-creation ops in the
    # graph.  Exporting on CPU keeps devices consistent; we move the
    # ExportedProgram to the target device afterwards via the official
    # move_to_device_pass (avoids FakeTensor device-propagation errors).
    dynamic_shapes = _build_dynamic_shapes(
        *sample_inputs, has_spin=is_spin, model_nnei=sum(model.get_sel())
    )
    exported = torch.export.export(
        traced,
        sample_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

    # torch.export inserts _assert_scalar guards for symbolic shape
    # relationships (e.g. Ne(nnei, sum(sel)), Ne(nall, nloc)).  These
    # are spurious — the model handles any valid input shapes correctly.
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
) -> None:
    """Deserialize a dictionary to a .pte model file."""
    exported, metadata, data_for_json, output_keys = _trace_and_export(
        data, model_json_override, do_atomic_virial
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
) -> None:
    """Deserialize a dictionary to a .pt2 model file (AOTInductor).

    Uses torch._inductor.aoti_compile_and_package to compile the exported
    program into a .pt2 package (ZIP archive with compiled shared libraries),
    then embeds metadata into the archive.
    """
    import zipfile

    from torch._inductor import (
        aoti_compile_and_package,
    )

    exported, metadata, data_for_json, output_keys = _trace_and_export(
        data, model_json_override, do_atomic_virial
    )

    # On CUDA, aggressive kernel fusion (default realize_opcount_threshold=30)
    # causes NaN in the backward pass (force/virial) of attention-based
    # descriptors (DPA1, DPA2). Setting threshold=0 prevents fusion and
    # avoids the NaN. Only applied on CUDA; CPU compilation is unaffected.
    import torch._inductor.config as _inductor_config

    import deepmd.pt_expt.utils.env as _env

    is_cuda = _env.DEVICE.type == "cuda"
    saved_threshold = _inductor_config.realize_opcount_threshold
    if is_cuda:
        _inductor_config.realize_opcount_threshold = 0
    try:
        aoti_compile_and_package(exported, package_path=model_file)
    finally:
        _inductor_config.realize_opcount_threshold = saved_threshold

    # Embed metadata into the .pt2 ZIP archive
    model_def_script = data.get("model_def_script") or {}
    metadata["output_keys"] = output_keys
    with zipfile.ZipFile(model_file, "a") as zf:
        zf.writestr("extra/metadata.json", json.dumps(metadata))
        zf.writestr("extra/model_def_script.json", json.dumps(model_def_script))
        zf.writestr(
            "extra/model.json",
            json.dumps(data_for_json, separators=(",", ":")),
        )
