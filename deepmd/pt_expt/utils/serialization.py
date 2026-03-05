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
    nloc: int = 2,
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
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )

    ext_coord = torch.tensor(extended_coord, dtype=torch.float64, device=DEVICE)
    ext_atype = torch.tensor(extended_atype, dtype=torch.int64, device=DEVICE)
    nlist_t = torch.tensor(nlist, dtype=torch.int64, device=DEVICE)
    mapping_t = torch.tensor(mapping, dtype=torch.int64, device=DEVICE)

    if dim_fparam > 0:
        fparam = torch.zeros(nframes, dim_fparam, dtype=torch.float64, device=DEVICE)
    else:
        fparam = None

    if dim_aparam > 0:
        aparam = torch.zeros(
            nframes, nloc, dim_aparam, dtype=torch.float64, device=DEVICE
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
        "fitting_output_defs": fitting_output_defs,
    }


def serialize_from_file(model_file: str) -> dict:
    """Serialize a .pte model file to a dictionary.

    Reads the model dict stored in the extra_files of the .pte archive.

    Parameters
    ----------
    model_file : str
        The .pte model file to be serialized.

    Returns
    -------
    dict
        The serialized model data.
    """
    extra_files = {"model.json": ""}
    torch.export.load(model_file, extra_files=extra_files)
    model_dict = json.loads(extra_files["model.json"])
    model_dict = _json_to_numpy(model_dict)
    return model_dict


def deserialize_to_file(model_file: str, data: dict) -> None:
    """Deserialize a dictionary to a .pte model file.

    Builds a pt_expt model from the dict, traces it via make_fx,
    exports with dynamic shapes, and saves using torch.export.save.

    Parameters
    ----------
    model_file : str
        The .pte model file to be saved.
    data : dict
        The dictionary to be deserialized (same format as dpmodel's
        serialize output, with "model" and optionally "model_def_script" keys).
    """
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )

    # 1. Deserialize into a pt_expt model
    model = BaseModel.deserialize(data["model"])
    model.eval()

    # 2. Collect metadata
    metadata = _collect_metadata(model)

    # 3. Create sample inputs for tracing
    # Use nframes=2 so make_fx doesn't specialize on nframes=1
    ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = _make_sample_inputs(
        model, nframes=2
    )

    # 4. Trace via forward_common_lower_exportable (make_fx)
    # Uses internal keys (energy, energy_redu, energy_derv_r, etc.)
    # so that communicate_extended_output can be applied at inference time.
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

    # 5. Build dynamic shapes and export
    dynamic_shapes = _build_dynamic_shapes(
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam
    )
    exported = torch.export.export(
        traced,
        (ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    # 6. Prepare extra files
    # Serialize the full model dict for cross-backend conversion
    from copy import (
        deepcopy,
    )

    data_for_json = deepcopy(data)
    data_for_json = _numpy_to_json_serializable(data_for_json)

    extra_files = {
        "model_def_script.json": json.dumps(metadata),
        "model.json": json.dumps(data_for_json, separators=(",", ":")),
    }

    # 7. Save
    torch.export.save(exported, model_file, extra_files=extra_files)
