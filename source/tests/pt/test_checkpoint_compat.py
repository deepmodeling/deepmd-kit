# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for PyTorch checkpoint compatibility."""

from copy import (
    deepcopy,
)

import torch

from deepmd.pt.infer.inference import Tester as PTTester
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)


def test_legacy_property_checkpoint_defaults_to_type_agnostic_stats(tmp_path) -> None:
    """Legacy property checkpoints retain their pre-#5322 statistics behavior."""
    model_params = {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4, 4],
            "rcut": 3.0,
            "rcut_smth": 0.5,
            "neuron": [4],
            "axis_neuron": 2,
            "seed": 1,
        },
        "fitting_net": {
            "type": "property",
            "property_name": "property",
            "task_dim": 1,
            "intensive": True,
            "neuron": [4],
            "seed": 1,
            "distinguish_types": True,
        },
    }
    model = get_model(deepcopy(model_params)).to("cpu")
    wrapper = ModelWrapper(model, model_params=deepcopy(model_params))
    state_dict = wrapper.state_dict()
    state_dict["_extra_state"]["model_params"]["fitting_net"].pop("distinguish_types")
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": state_dict}, checkpoint)

    loaded = PTTester(str(checkpoint))

    assert loaded.model.atomic_model.fitting_net.get_distinguish_types() is False


def test_legacy_checkpoint_load_keeps_current_model_metadata() -> None:
    """Loading legacy weights does not replace explicit current metadata."""
    model_params = {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4, 4],
            "rcut": 3.0,
            "rcut_smth": 0.5,
            "neuron": [4],
            "axis_neuron": 2,
            "seed": 1,
        },
        "fitting_net": {
            "type": "property",
            "property_name": "property",
            "task_dim": 1,
            "intensive": True,
            "neuron": [4],
            "seed": 1,
            "distinguish_types": True,
        },
    }
    source = ModelWrapper(
        get_model(deepcopy(model_params)).to("cpu"),
        model_params=deepcopy(model_params),
    )
    legacy_state = source.state_dict()
    legacy_state["_extra_state"]["model_params"]["fitting_net"].pop("distinguish_types")

    target = ModelWrapper(
        get_model(deepcopy(model_params)).to("cpu"),
        model_params=deepcopy(model_params),
    )
    legacy_state["_extra_state"] = target.state_dict()["_extra_state"]
    target.load_state_dict(legacy_state)

    assert (
        target.state_dict()["_extra_state"]["model_params"]["fitting_net"][
            "distinguish_types"
        ]
        is True
    )
