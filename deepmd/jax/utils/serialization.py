# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import orbax.checkpoint as ocp

from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
    save_dp_model,
)
from deepmd.jax.env import (
    jax,
    jax_export,
    jnp,
    nnx,
)
from deepmd.jax.model.model import (
    BaseModel,
    get_model,
)


def _convert_str_to_int_key(item: dict) -> None:
    """Convert Orbax-restored numeric index keys from strings back to ints."""
    for key, value in item.copy().items():
        if isinstance(value, dict):
            _convert_str_to_int_key(value)
        if isinstance(key, str) and key.isdigit():
            item[int(key)] = item.pop(key)


def _normalize_restored_state_keys(
    state: dict,
    model_def_script: dict,
) -> None:
    """Normalize restored state keys while preserving multi-task branch names."""
    if "model_dict" in model_def_script:
        state_by_model = state.get("models", state)
        for model_key in model_def_script["model_dict"]:
            if model_key in state_by_model and isinstance(
                state_by_model[model_key], dict
            ):
                _convert_str_to_int_key(state_by_model[model_key])
        return
    _convert_str_to_int_key(state)


def _state_sequence_to_numpy_list(state_value: Any) -> list[np.ndarray]:
    """Convert an Orbax-restored list/dict sequence to NumPy arrays."""
    if isinstance(state_value, dict):
        values = [state_value[key] for key in sorted(state_value)]
    else:
        values = state_value
    return [np.asarray(getattr(value, "value", value)) for value in values]


def _state_value_to_numpy(state_value: Any) -> np.ndarray:
    """Convert an Orbax-restored state value to a NumPy array."""
    return np.asarray(getattr(state_value, "value", state_value))


def _restore_compression_slots_from_state(obj: Any, state: Any) -> None:
    """Create compression variable slots before replacing an NNX state.

    A compressed ``.jax`` checkpoint stores tabulation arrays in the NNX state,
    while ``model_def_script`` still describes the original uncompressed model.
    Build the corresponding descriptor attributes first so Flax can match the
    restored state keys.
    """
    if not isinstance(state, dict):
        return
    if (
        (hasattr(obj, "compress") or hasattr(obj, "geo_compress"))
        and "compress_data" in state
        and "compress_info" in state
    ):
        obj.compress_data = _state_sequence_to_numpy_list(state["compress_data"])
        obj.compress_info = _state_sequence_to_numpy_list(state["compress_info"])
        if hasattr(obj, "compress"):
            obj.compress = True
        if hasattr(obj, "geo_compress"):
            obj.geo_compress = True
        if hasattr(obj, "se_atten"):
            obj.se_atten.compress_data = obj.compress_data
            obj.se_atten.compress_info = obj.compress_info
            if hasattr(obj.se_atten, "geo_compress"):
                obj.se_atten.geo_compress = True
    if (
        hasattr(obj, "compress") or hasattr(obj, "tebd_compress")
    ) and "type_embd_data" in state:
        obj.type_embd_data = _state_value_to_numpy(state["type_embd_data"])
        if hasattr(obj, "compress"):
            obj.compress = True
        if hasattr(obj, "tebd_compress"):
            obj.tebd_compress = True
        if hasattr(obj, "geo_compress"):
            obj.geo_compress = "compress_data" in state and "compress_info" in state
        if hasattr(obj, "se_atten"):
            obj.se_atten.type_embd_data = obj.type_embd_data
            obj.se_atten.tebd_compress = True
            if hasattr(obj.se_atten, "geo_compress"):
                obj.se_atten.geo_compress = getattr(obj, "geo_compress", False)
            if getattr(obj, "geo_compress", False):
                obj.se_atten.compress_data = obj.compress_data
                obj.se_atten.compress_info = obj.compress_info
    for name, child_state in state.items():
        if not isinstance(child_state, dict):
            continue
        if isinstance(name, int):
            try:
                child = obj[name]
            except (IndexError, KeyError, TypeError):
                continue
        else:
            if not hasattr(obj, name):
                continue
            child = getattr(obj, name)
        _restore_compression_slots_from_state(child, child_state)
        if name == "se_atten" and hasattr(obj, "compress"):
            child_type_embd_data = getattr(child, "type_embd_data", None)
            if child_type_embd_data is not None:
                obj.type_embd_data = child_type_embd_data
                obj.tebd_compress = getattr(child, "tebd_compress", True)
                obj.compress = True
            if getattr(child, "geo_compress", False):
                obj.geo_compress = True
                obj.compress_data = child.compress_data
                obj.compress_info = child.compress_info


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(np.asarray(getattr(value, "value", value)))


def _set_model_min_nbor_dist_from_data(model: BaseModel, data: dict) -> None:
    if model.get_min_nbor_dist() is not None:
        return
    min_nbor_dist = _to_optional_float(data.get("min_nbor_dist"))
    if min_nbor_dist is None:
        min_nbor_dist = _to_optional_float(
            data.get("constants", {}).get("min_nbor_dist")
        )
    if min_nbor_dist is not None:
        model.min_nbor_dist = min_nbor_dist


def _find_compressed_type_two_side_descriptors(data: Any) -> list[str]:
    """Find compressed descriptors whose JAX HLO path is not exportable."""
    if not isinstance(data, dict):
        return []
    matches = []
    descriptor_type = data.get("type")
    if (
        descriptor_type in {"se_e2_a", "se_a", "dpa1", "se_atten"}
        and "compress" in data
        and data.get("type_one_side") is False
    ):
        matches.append(descriptor_type)
    for value in data.values():
        if isinstance(value, dict):
            matches.extend(_find_compressed_type_two_side_descriptors(value))
        elif isinstance(value, list):
            for item in value:
                matches.extend(_find_compressed_type_two_side_descriptors(item))
    return matches


def _check_compressed_hlo_exportable(data: dict) -> None:
    """Reject compressed descriptors that cannot be traced to StableHLO."""
    descriptor_types = _find_compressed_type_two_side_descriptors(data.get("model", {}))
    if descriptor_types:
        names = ", ".join(sorted(set(descriptor_types)))
        raise ValueError(
            "Compressed JAX HLO export does not support type_one_side=False for "
            f"{names} descriptors because the compressed path uses data-dependent "
            "type slices that cannot be traced. Use type_one_side=True for HLO "
            "export, or write a .jax checkpoint instead."
        )


def deserialize_to_file(model_file: str, data: dict) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    model_file : str
        The model file to be saved.
    data : dict
        The dictionary to be deserialized.
    """
    if model_file.endswith(".jax"):
        model_def_script = data["model_def_script"].copy()
        min_nbor_dist = _to_optional_float(data.get("min_nbor_dist"))
        if min_nbor_dist is None:
            min_nbor_dist = _to_optional_float(
                data.get("constants", {}).get("min_nbor_dist")
            )
        if min_nbor_dist is not None:
            model_def_script["_min_nbor_dist"] = min_nbor_dist
        if "model_dict" in model_def_script:
            models = {
                model_key: BaseModel.deserialize(data["model"]["model_dict"][model_key])
                for model_key in model_def_script["model_dict"]
            }
            state = {
                "models": {
                    model_key: nnx.split(model)[1].to_pure_dict()
                    for model_key, model in models.items()
                }
            }
        else:
            model = BaseModel.deserialize(data["model"])
            _, state = nnx.split(model)
            state = state.to_pure_dict()
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                Path(model_file).absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    model_def_script=ocp.args.JsonSave(model_def_script),
                ),
            )
    elif model_file.endswith(".hlo"):
        _check_compressed_hlo_exportable(data)
        model = BaseModel.deserialize(data["model"])
        _set_model_min_nbor_dist_from_data(model, data)
        model_def_script = data["model_def_script"]
        call_lower = model.call_common_lower

        nf, nloc, nghost = jax_export.symbolic_shape("nf, nloc, nghost")

        def exported_whether_do_atomic_virial(
            do_atomic_virial: bool, has_ghost_atoms: bool
        ) -> "jax_export.Exported":
            def call_lower_with_fixed_do_atomic_virial(
                coord: jnp.ndarray,
                atype: jnp.ndarray,
                nlist: jnp.ndarray,
                mapping: jnp.ndarray,
                fparam: jnp.ndarray,
                aparam: jnp.ndarray,
            ) -> dict[str, jnp.ndarray]:
                return call_lower(
                    coord,
                    atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    do_atomic_virial=do_atomic_virial,
                )

            if has_ghost_atoms:
                nghost_ = nghost
            else:
                nghost_ = 0

            return jax_export.export(jax.jit(call_lower_with_fixed_do_atomic_virial))(
                jax.ShapeDtypeStruct(
                    (nf, nloc + nghost_, 3), jnp.float64
                ),  # extended_coord
                jax.ShapeDtypeStruct((nf, nloc + nghost_), jnp.int32),  # extended_atype
                jax.ShapeDtypeStruct((nf, nloc, model.get_nnei()), jnp.int64),  # nlist
                jax.ShapeDtypeStruct((nf, nloc + nghost_), jnp.int64),  # mapping
                jax.ShapeDtypeStruct((nf, model.get_dim_fparam()), jnp.float64)
                if model.get_dim_fparam()
                else None,  # fparam
                jax.ShapeDtypeStruct((nf, nloc, model.get_dim_aparam()), jnp.float64)
                if model.get_dim_aparam()
                else None,  # aparam
            )

        exported = exported_whether_do_atomic_virial(
            do_atomic_virial=False, has_ghost_atoms=True
        )
        exported_atomic_virial = exported_whether_do_atomic_virial(
            do_atomic_virial=True, has_ghost_atoms=True
        )
        serialized: bytearray = exported.serialize()
        serialized_atomic_virial = exported_atomic_virial.serialize()

        exported_no_ghost = exported_whether_do_atomic_virial(
            do_atomic_virial=False, has_ghost_atoms=False
        )
        exported_atomic_virial_no_ghost = exported_whether_do_atomic_virial(
            do_atomic_virial=True, has_ghost_atoms=False
        )
        serialized_no_ghost: bytearray = exported_no_ghost.serialize()
        serialized_atomic_virial_no_ghost = exported_atomic_virial_no_ghost.serialize()

        data = data.copy()
        data.setdefault("@variables", {})
        data["@variables"]["stablehlo"] = np.void(serialized)
        data["@variables"]["stablehlo_atomic_virial"] = np.void(
            serialized_atomic_virial
        )
        data["@variables"]["stablehlo_no_ghost"] = np.void(serialized_no_ghost)
        data["@variables"]["stablehlo_atomic_virial_no_ghost"] = np.void(
            serialized_atomic_virial_no_ghost
        )
        data["constants"] = {
            "type_map": model.get_type_map(),
            "rcut": model.get_rcut(),
            "dim_fparam": model.get_dim_fparam(),
            "dim_aparam": model.get_dim_aparam(),
            "sel_type": model.get_sel_type(),
            "is_aparam_nall": model.is_aparam_nall(),
            "model_output_type": model.model_output_type(),
            "mixed_types": model.mixed_types(),
            "min_nbor_dist": model.get_min_nbor_dist(),
            "sel": model.get_sel(),
            "has_default_fparam": model.has_default_fparam(),
            "default_fparam": model.get_default_fparam(),
        }
        save_dp_model(filename=model_file, model_dict=data)
    elif model_file.endswith(".savedmodel"):
        # Keep the historical JAX/JAX2TF meaning of ".savedmodel": this
        # exporter must lower the JAX model through jax2tf and preserve
        # XlaCallModule ops in the SavedModel. The TF2 eager SavedModel
        # exporter owns the ".savedmodeltf" suffix.
        from deepmd.jax.jax2tf.serialization import (
            deserialize_to_file as deserialize_to_savedmodel,
        )

        return deserialize_to_savedmodel(model_file, data)
    else:
        raise ValueError("Unsupported file extension")


def serialize_from_file(model_file: str) -> dict:
    """Serialize the model file to a dictionary.

    Parameters
    ----------
    model_file : str
        The model file to be serialized.

    Returns
    -------
    dict
        The serialized model data.
    """
    if model_file.endswith(".jax"):
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            data = checkpointer.restore(
                Path(model_file).absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardRestore(),
                    model_def_script=ocp.args.JsonRestore(),
                ),
            )
        state = data.state
        model_def_script = data.model_def_script
        _normalize_restored_state_keys(state, model_def_script)
        min_nbor_dist = None

        def restore_model(model_params: dict, model_state: dict) -> BaseModel:
            abstract_model = get_model(model_params)
            _restore_compression_slots_from_state(abstract_model, model_state)
            graphdef, abstract_state = nnx.split(abstract_model)
            abstract_state.replace_by_pure_dict(model_state)
            return nnx.merge(graphdef, abstract_state)

        if "model_dict" in model_def_script:
            state_by_model = state.get("models", state)
            model_dict = {"model_dict": {}}
            for model_key, model_params in model_def_script["model_dict"].items():
                model = restore_model(model_params, state_by_model[model_key])
                model_dict["model_dict"][model_key] = model.serialize()
        else:
            model = restore_model(model_def_script, state)
            model_dict = model.serialize()
            min_nbor_dist = _to_optional_float(model.get_min_nbor_dist())
            if min_nbor_dist is None:
                min_nbor_dist = _to_optional_float(
                    model_def_script.get("_min_nbor_dist")
                )
        data = {
            "backend": "JAX",
            "jax_version": jax.__version__,
            "model": model_dict,
            "model_def_script": model_def_script,
            "@variables": {},
        }
        if min_nbor_dist is not None:
            data["min_nbor_dist"] = min_nbor_dist
        return data
    elif model_file.endswith(".hlo"):
        data = load_dp_model(model_file)
        data.pop("constants")
        data["@variables"].pop("stablehlo")
        return data
    elif model_file.endswith(".savedmodel"):
        raise ValueError(
            "JAX SavedModel does not support lossless file serialization. "
            "Use DeepEval.serialize() for a structure-only model tree."
        )
    else:
        raise ValueError(
            "JAX backend only supports lossless file serialization for .jax "
            "directory and .hlo."
        )
