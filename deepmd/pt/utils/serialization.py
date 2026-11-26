# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from copy import (
    deepcopy,
)

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.dpa4_serialization import (
    is_dpa4_model,
    to_pt_dpa4_model,
)


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
    training = None
    if model_file.endswith(".pth"):
        saved_model = torch.jit.load(model_file, map_location="cpu")
        model_def_script = json.loads(saved_model.model_def_script)
        model = get_model(model_def_script)
        model.load_state_dict(saved_model.state_dict())
    elif model_file.endswith(".pt"):
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_def_script = state_dict["_extra_state"]["model_params"]
        multi_task = "model_dict" in model_def_script
        model = (
            {
                key: get_model(params)
                for key, params in model_def_script["model_dict"].items()
            }
            if multi_task
            else get_model(model_def_script)
        )
        branches = model if multi_task else {"Default": model}
        definitions = (
            model_def_script["model_dict"]
            if multi_task
            else {"Default": model_def_script}
        )
        for key, branch in branches.items():
            lora = definitions[key].get("lora")
            if lora is not None:
                from deepmd.pt.model.descriptor.sezm_nn.lora import (
                    apply_lora_to_sezm,
                )

                apply_lora_to_sezm(
                    branch, rank=int(lora["rank"]), alpha=lora.get("alpha")
                )
        modelwrapper = ModelWrapper(model)
        shared_links = checkpoint.get("shared_links") or state_dict["_extra_state"].get(
            "shared_links"
        )
        if multi_task and not shared_links:
            from deepmd.pt.utils.checkpoint import (
                infer_shared_links,
            )

            shared_links = infer_shared_links(modelwrapper, state_dict)
        if shared_links:
            modelwrapper.share_params(
                shared_links,
                dict.fromkeys(modelwrapper.model, 1.0),
                data_stat_protect=1e-2,
                resume=True,
            )
        modelwrapper.load_state_dict(state_dict)
        training = dict(modelwrapper.train_infos)
        if modelwrapper.shared_links:
            training["shared_links"] = modelwrapper.shared_links
        optimizer_groups = checkpoint.get("optimizer", {}).get("param_groups", [])
        if optimizer_groups and "muon_mode" in optimizer_groups[0]:
            from deepmd.pt.utils.checkpoint import (
                export_training_state,
            )

            training = export_training_state(checkpoint, modelwrapper)
        model = modelwrapper.model if multi_task else modelwrapper.model["Default"]
    else:
        raise ValueError("PyTorch backend only supports converting .pth or .pt file")

    multi_task = "model_dict" in model_def_script
    model_dict = (
        {"model_dict": {key: item.serialize() for key, item in model.items()}}
        if multi_task
        else model.serialize()
    )
    branches = model.values() if multi_task else (model,)
    lower_kinds = {branch.export_lower_input_kind() for branch in branches}
    if len(lower_kinds) != 1:
        raise ValueError(
            "The branches of a multi-task model must share one neighbor input "
            f"kind; got {sorted(lower_kinds)}."
        )
    data = {
        "backend": "PyTorch",
        "pt_version": str(torch.__version__),
        "model": model_dict,
        "model_def_script": model_def_script,
        "lower_input_kind": lower_kinds.pop(),
        "@variables": {},
    }
    if training is not None:
        data["training"] = training
    if not multi_task and model.get_min_nbor_dist() is not None:
        data["@variables"]["min_nbor_dist"] = model.get_min_nbor_dist()
    return data


def deserialize_to_file(
    model_file: str, data: dict, *, lower_kind: str = "auto"
) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    model_file : str
        The model file to be saved.
    data : dict
        The dictionary to be deserialized.
    lower_kind : str, default="auto"
        Required neighbor execution semantics of the source model.
    """
    if model_file.endswith(".pt"):
        model_definition = deepcopy(data["model_def_script"])
        if "model_dict" in model_definition:
            models = {}
            for key, model_data in data["model"]["model_dict"].items():
                models[key], model_definition["model_dict"][key] = (
                    _deserialize_checkpoint_model(
                        model_data, model_definition["model_dict"][key]
                    )
                )
            model = models
        else:
            model, model_definition = _deserialize_checkpoint_model(
                data["model"], model_definition
            )
        wrapper = ModelWrapper(model, model_params=model_definition)
        minimum = data.get(
            "min_nbor_dist", data.get("@variables", {}).get("min_nbor_dist")
        )
        if minimum is not None:
            for branch in wrapper.model.values():
                branch.min_nbor_dist.fill_(float(minimum))
        if data.get("training", {}).get("shared_links"):
            wrapper.share_params(
                data["training"]["shared_links"],
                dict.fromkeys(wrapper.model, 1.0),
                data_stat_protect=1e-2,
                resume=True,
            )
        for item in wrapper.model.values():
            _validate_lower_kind(item, lower_kind)
        from deepmd.pt.utils.checkpoint import (
            import_training_state,
        )

        checkpoint = import_training_state(data.get("training", {}), wrapper)
        torch.save(checkpoint, model_file)
        return
    if not model_file.endswith(".pth"):
        raise ValueError(
            "PyTorch backend supports converting .pt checkpoints or .pth frozen models."
        )
    model, model_definition = _deserialize_checkpoint_model(
        data["model"], data["model_def_script"]
    )
    _validate_lower_kind(model, lower_kind)
    # JIT will happy in this way...
    model.model_def_script = json.dumps(model_definition)
    if "min_nbor_dist" in data.get("@variables", {}):
        model.min_nbor_dist = torch.tensor(
            float(data["@variables"]["min_nbor_dist"]),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
    model = torch.jit.script(model)
    torch.jit.save(model, model_file)


def _validate_lower_kind(model: BaseModel, lower_kind: str) -> None:
    """Compare neighbor-selection semantics across graph ABI spellings."""
    canonical = {"edge_vec": "graph"}
    actual = model.export_lower_input_kind()
    if lower_kind != "auto" and canonical.get(actual, actual) != canonical.get(
        lower_kind, lower_kind
    ):
        raise ValueError(
            f"Cannot preserve {lower_kind!r} neighbor semantics in the target PyTorch model."
        )


def _deserialize_checkpoint_model(
    model_data: dict, model_definition: dict
) -> tuple[BaseModel, dict]:
    """Restore a model while retaining the DPA4 native-spin and bridging contract."""
    if is_dpa4_model(model_data):
        model_data, model_definition = to_pt_dpa4_model(model_data, model_definition)
    if model_data.get("type") == "spin_ener":
        from deepmd.pt.model.model.spin_model import (
            SpinEnergyModel,
        )

        model = SpinEnergyModel.deserialize(model_data)
    elif model_data.get("type") == "sezm_native_spin":
        from deepmd.pt.model.model.sezm_native_spin_model import (
            SeZMNativeSpinModel,
        )

        model = SeZMNativeSpinModel.deserialize(model_data)
    else:
        model = BaseModel.deserialize(model_data)
    return model, model_definition
