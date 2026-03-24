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

    Returns
    -------
    tuple
        (ext_coord, ext_atype, nlist, mapping, fparam, aparam)
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

    return ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam


def _build_dynamic_shapes(
    _ext_coord: torch.Tensor,
    _ext_atype: torch.Tensor,
    _nlist: torch.Tensor,
    _mapping: torch.Tensor,
    fparam: torch.Tensor | None,
    aparam: torch.Tensor | None,
) -> tuple:
    """Build dynamic shape specifications for torch.export.

    Marks nframes, nloc and nall as dynamic dimensions so the exported
    program handles arbitrary frame and atom counts.

    Returns a tuple (not dict) to match positional args of the make_fx
    traced module, whose arg names may have suffixes like ``_1``.
    """
    nframes_dim = torch.export.Dim("nframes", min=1)
    nall_dim = torch.export.Dim("nall", min=1)
    nloc_dim = torch.export.Dim("nloc", min=1)

    return (
        {0: nframes_dim, 1: nall_dim},  # extended_coord: (nframes, nall, 3)
        {0: nframes_dim, 1: nall_dim},  # extended_atype: (nframes, nall)
        {0: nframes_dim, 1: nloc_dim},  # nlist: (nframes, nloc, nnei)
        {0: nframes_dim, 1: nall_dim},  # mapping: (nframes, nall)
        {0: nframes_dim} if fparam is not None else None,  # fparam
        {0: nframes_dim, 1: nloc_dim} if aparam is not None else None,  # aparam
    )


def _collect_metadata(model: torch.nn.Module) -> dict:
    """Collect metadata from the model for storage in .pte extra_files."""
    # Serialize the fitting output definitions so that ModelOutputDef
    # can be reconstructed at inference time without loading the full model.
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
    return {
        "type_map": model.get_type_map(),
        "rcut": model.get_rcut(),
        "sel": model.get_sel(),
        "model_output_type": model.model_output_type(),
        "dim_fparam": model.get_dim_fparam(),
        "dim_aparam": model.get_dim_aparam(),
        "mixed_types": model.mixed_types(),
        "sel_type": model.get_sel_type(),
        "has_default_fparam": model.has_default_fparam(),
        "fitting_output_defs": fitting_output_defs,
    }


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
        ``model_params.json``, it is included under the
        ``"model_params"`` key.
    """
    if model_file.endswith(".pt2"):
        return _serialize_from_file_pt2(model_file)
    else:
        return _serialize_from_file_pte(model_file)


def _serialize_from_file_pte(model_file: str) -> dict:
    """Serialize a .pte model file to a dictionary."""
    extra_files = {"model.json": "", "model_params.json": ""}
    torch.export.load(model_file, extra_files=extra_files)
    model_dict = json.loads(extra_files["model.json"])
    model_dict = _json_to_numpy(model_dict)
    if extra_files["model_params.json"]:
        model_dict["model_params"] = json.loads(extra_files["model_params.json"])
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
        model_params_json = ""
        if "extra/model_params.json" in zf.namelist():
            model_params_json = zf.read("extra/model_params.json").decode("utf-8")
    model_dict = json.loads(model_json)
    model_dict = _json_to_numpy(model_dict)
    if model_params_json:
        model_dict["model_params"] = json.loads(model_params_json)
    return model_dict


def deserialize_to_file(
    model_file: str,
    data: dict,
    model_params: dict | None = None,
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
    model_params : dict or None
        Original model config (the dict passed to ``get_model``).
        If provided, embedded in the .pte so that ``--use-pretrain-script``
        can extract descriptor/fitting params at finetune time.
    model_json_override : dict or None
        If provided, this dict is stored in model.json instead of ``data``.
        Used by ``dp compress`` to store the compressed model dict while
        tracing the uncompressed model (make_fx cannot trace custom ops).
    """
    if model_file.endswith(".pt2"):
        _deserialize_to_file_pt2(model_file, data, model_json_override, model_params)
    else:
        _deserialize_to_file_pte(model_file, data, model_json_override, model_params)


def _trace_and_export(
    data: dict,
    model_json_override: dict | None = None,
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

    # 1. Deserialize model on CPU for make_fx tracing.
    # make_fx with _allow_non_fake_inputs=True keeps real model parameters;
    # on CUDA the autograd engine requires CUDA streams for those real
    # tensors during torch.autograd.grad, but proxy-tensor dispatch doesn't
    # set streams up → assertion failure.  Tracing on CPU avoids this.
    model = BaseModel.deserialize(data["model"])
    model.to("cpu")
    model.eval()

    # 2. Collect metadata
    metadata = _collect_metadata(model)

    # 3. Create sample inputs on CPU for tracing
    # Use nframes=2 so make_fx doesn't specialize on nframes=1
    _orig_device = _env.DEVICE
    _env.DEVICE = torch.device("cpu")
    try:
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = _make_sample_inputs(
            model, nframes=2
        )
    finally:
        _env.DEVICE = _orig_device

    # 4. Trace via make_fx on CPU.
    # This decomposes torch.autograd.grad into aten ops so the resulting
    # GraphModule no longer contains autograd calls.
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
    sample_out = traced(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
    output_keys = list(sample_out.keys())

    # 6. Export on CPU.
    # make_fx on CPU bakes device='cpu' into tensor-creation ops in the
    # graph.  Exporting on CPU keeps devices consistent; we move the
    # ExportedProgram to the target device afterwards via the official
    # move_to_device_pass (avoids FakeTensor device-propagation errors).
    dynamic_shapes = _build_dynamic_shapes(
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam
    )
    exported = torch.export.export(
        traced,
        (ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam),
        dynamic_shapes=dynamic_shapes,
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

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
    model_params: dict | None = None,
) -> None:
    """Deserialize a dictionary to a .pte model file."""
    exported, metadata, data_for_json, _output_keys = _trace_and_export(
        data, model_json_override
    )

    extra_files = {
        "model_def_script.json": json.dumps(metadata),
        "model.json": json.dumps(data_for_json, separators=(",", ":")),
    }
    if model_params is not None:
        extra_files["model_params.json"] = json.dumps(model_params)

    torch.export.save(exported, model_file, extra_files=extra_files)


def _deserialize_to_file_pt2(
    model_file: str,
    data: dict,
    model_json_override: dict | None = None,
    model_params: dict | None = None,
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
        data, model_json_override
    )

    # Compile via AOTInductor into a .pt2 package
    aoti_compile_and_package(exported, package_path=model_file)

    # Embed metadata into the .pt2 ZIP archive
    with zipfile.ZipFile(model_file, "a") as zf:
        zf.writestr("extra/model_def_script.json", json.dumps(metadata))
        zf.writestr("extra/output_keys.json", json.dumps(output_keys))
        zf.writestr(
            "extra/model.json",
            json.dumps(data_for_json, separators=(",", ":")),
        )
        if model_params is not None:
            zf.writestr("extra/model_params.json", json.dumps(model_params))
