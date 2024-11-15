# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
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
        model = BaseModel.deserialize(data["model"])
        model_def_script = data["model_def_script"]
        _, state = nnx.split(model)
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                Path(model_file).absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state.to_pure_dict()),
                    model_def_script=ocp.args.JsonSave(model_def_script),
                ),
            )
    elif model_file.endswith(".hlo"):
        model = BaseModel.deserialize(data["model"])
        model_def_script = data["model_def_script"]
        call_lower = model.call_lower

        nf, nloc, nghost = jax_export.symbolic_shape("nf, nloc, nghost")

        def exported_whether_do_atomic_virial(
            do_atomic_virial: bool, has_ghost_atoms: bool
        ):
            def call_lower_with_fixed_do_atomic_virial(
                coord, atype, nlist, mapping, fparam, aparam
            ):
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
        }
        save_dp_model(filename=model_file, model_dict=data)
    elif model_file.endswith(".savedmodel"):
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

        # convert str "1" to int 1 key
        def convert_str_to_int_key(item: dict) -> None:
            for key, value in item.copy().items():
                if isinstance(value, dict):
                    convert_str_to_int_key(value)
                if key.isdigit():
                    item[int(key)] = item.pop(key)

        convert_str_to_int_key(state)

        model_def_script = data.model_def_script
        abstract_model = get_model(model_def_script)
        graphdef, abstract_state = nnx.split(abstract_model)
        abstract_state.replace_by_pure_dict(state)
        model = nnx.merge(graphdef, abstract_state)
        model_dict = model.serialize()
        data = {
            "backend": "JAX",
            "jax_version": jax.__version__,
            "model": model_dict,
            "model_def_script": model_def_script,
            "@variables": {},
        }
        return data
    elif model_file.endswith(".hlo"):
        data = load_dp_model(model_file)
        data.pop("constants")
        data["@variables"].pop("stablehlo")
        return data
    else:
        raise ValueError("JAX backend only supports converting .jax directory")
